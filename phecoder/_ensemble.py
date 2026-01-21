# phecoder/_ensemble.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Callable, Sequence
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

from .utils import _sanitize_model_name, _ensure_dir, _now, _df_fingerprint

# ───────────────────────────── helpers ───────────────────────────── #


def _align_weights(
    columns: Sequence[str], weights: Optional[Dict[str, float]]
) -> np.ndarray:
    """Return a weight vector aligned to `columns` (model names)."""
    if not weights:
        return np.ones(len(columns), dtype=float)
    return np.array([float(weights.get(m, 1.0)) for m in columns], dtype=float)


def _minmax_norm_array(x: np.ndarray) -> np.ndarray:
    """Column-wise min-max normalization [0,1]."""
    xmin = np.nanmin(x, axis=0)
    xmax = np.nanmax(x, axis=0)
    rng = np.where(xmax > xmin, xmax - xmin, np.nan)
    return (x - xmin) / rng


def _z_norm_array(x: np.ndarray) -> np.ndarray:
    """Column-wise z-score normalization."""
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd > 0, sd, np.nan)
    return (x - mu) / sd


def _wide_matrix(
    sub: pd.DataFrame, value: str, agg: str = "min"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized pivot: return (rows, cols, matrix).
    value : 'rank' or 'score'
    agg   : 'min' (for ranks) or 'max' (for scores)
    """
    # Temporarily convert 'model' to category for faster unstack
    sub = sub.copy()
    sub["model"] = sub["model"].astype("category")

    # Build wide table
    wide = sub.set_index(["icd_code", "model"])[value].unstack("model")

    # Group by icd_code index if duplicates
    if agg == "min":
        wide = wide.groupby(level=0).min()
    elif agg == "max":
        wide = wide.groupby(level=0).max()

    rows = wide.index.to_numpy()
    # Convert categories back to string to ensure downstream compatibility
    cols = wide.columns.astype(str).to_numpy()
    mat = wide.to_numpy(dtype=float)

    return rows, cols, mat


def _order_desc(score_vec: np.ndarray) -> np.ndarray:
    return np.argsort(-score_vec)


# ───────────────────────────── core fusers ───────────────────────────── #


def _rrf_fuse(sub: pd.DataFrame, k: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, r = _wide_matrix(sub, "rank", agg="min")
    max_rank = sub["rank"].max()
    r = np.where(np.isnan(r), max_rank + 1000, r)
    fused = (1.0 / (k + r)).sum(axis=1)
    order = _order_desc(fused)
    return rows[order], fused[order]


def _mean_rank_fuse(
    sub: pd.DataFrame, weights: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, r = _wide_matrix(sub, "rank", agg="min")
    max_rank = sub["rank"].max()
    r = np.where(np.isnan(r), max_rank + 1000, r)
    w = _align_weights(cols, weights)
    mean_r = np.average(r, axis=1, weights=w)
    score = -mean_r
    order = _order_desc(score)
    return rows[order], score[order]


def _median_rank_fuse(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, r = _wide_matrix(sub, "rank", agg="min")
    med = np.nanmedian(r, axis=1)
    score = -med
    order = _order_desc(score)
    return rows[order], score[order]


def _rra_fuse(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, r = _wide_matrix(sub, "rank", agg="min")
    max_rank = sub["rank"].max()
    r = np.where(np.isnan(r), max_rank + 1000, r)
    N = r.shape[0]
    pvals = np.clip(r / float(N + 1), 1e-12, 1 - 1e-12)
    Z = norm.isf(pvals)
    counts = np.sum(~np.isnan(Z), axis=1)
    denom = np.sqrt(np.maximum(counts, 1))
    zsum = np.nansum(Z, axis=1) / denom
    order = _order_desc(zsum)
    return rows[order], zsum[order]


def _fisher_fuse(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, s = _wide_matrix(sub, "score", agg="max")
    # Vectorized ECDF per column
    pvals = np.full_like(s, np.nan)
    for j in range(s.shape[1]):
        col = s[:, j]
        mask = np.isfinite(col)
        if not np.any(mask):
            continue
        x = np.sort(col[mask])
        ranks = np.searchsorted(x, col[mask], side="right")
        F = ranks / float(x.size)
        pvals[mask, j] = 1.0 - F
    with np.errstate(divide="ignore"):
        T = -2.0 * np.log(np.clip(pvals, 1e-300, 1))
    stat = np.nansum(T, axis=1)
    order = _order_desc(stat)
    return rows[order], stat[order]


def _zsum_fuse(
    sub: pd.DataFrame, weights: Optional[Dict[str, float]] = None, average: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, s = _wide_matrix(sub, "score", agg="max")
    z = _z_norm_array(s)
    w = _align_weights(cols, weights)
    zsum = np.nansum(z * w, axis=1)
    if average:
        valid = np.isfinite(z)
        denom = (valid * w).sum(axis=1)
        denom = np.where(denom > 0, denom, 1.0)
        zsum = zsum / denom
    order = _order_desc(zsum)
    return rows[order], zsum[order]


def _combsum_minmax_fuse(
    sub: pd.DataFrame, weights: Optional[Dict[str, float]] = None, average: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, s = _wide_matrix(sub, "score", agg="max")
    normed = _minmax_norm_array(s)
    w = _align_weights(cols, weights)
    fused = np.nansum(normed * w, axis=1)
    if average:
        valid = np.isfinite(normed)
        denom = (valid * w).sum(axis=1)
        denom = np.where(denom > 0, denom, 1.0)
        fused = fused / denom
    order = _order_desc(fused)
    return rows[order], fused[order]


def _combmnz_minmax_fuse(
    sub: pd.DataFrame, weights: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, s = _wide_matrix(sub, "score", agg="max")
    normed = _minmax_norm_array(s)
    w = _align_weights(cols, weights)
    sums = np.nansum(normed * w, axis=1)
    nnz = np.sum(np.isfinite(normed), axis=1).astype(float)
    fused = sums * nnz
    order = _order_desc(fused)
    return rows[order], fused[order]


# Registry
_FUSERS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
    "rrf": _rrf_fuse,
    "borda": _mean_rank_fuse,
    "mean_rank": _mean_rank_fuse,
    "median_rank": _median_rank_fuse,
    "rra": _rra_fuse,
    "zsum": _zsum_fuse,
    "zavg": lambda sub, **kw: _zsum_fuse(sub, average=True, **kw),
    "combsum": _combsum_minmax_fuse,
    "combmnz": _combmnz_minmax_fuse,
    "avg_minmax": lambda sub, **kw: _combsum_minmax_fuse(sub, average=True, **kw),
    "fisher": _fisher_fuse,
}


def _apply_fuser(fuser, sub, method_kwargs):
    try:
        return fuser(sub, **method_kwargs)
    except TypeError:
        return fuser(sub)


# ───────────────────────── orchestrator ─────────────────── #


def _build_ensemble_from_runs(
    *,
    output_dir,
    phecode_df: pd.DataFrame,
    icd_df: pd.DataFrame,
    model_to_run_dir: Dict[str, str],
    method: str = "rrf",
    method_kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    run_hash: Optional[str] = None,
    overwrite: bool = True,
) -> None:
    """
    Reads per-model similarity.parquet files (already computed),
    fuses per phecode, and writes a new synthetic run that looks like a normal model run.

    Parameters
    ----------
    method : one of the keys in _FUSERS, e.g.
        - 'rrf' (kwargs: k=60)
        - 'borda' / 'mean_rank' (kwargs: weights: Dict[str,float])
        - 'median_rank'
        - 'rra'
        - 'zsum' / 'zavg' (kwargs: weights: Dict[str,float], average: bool)
        - 'combsum' / 'avg_minmax' (kwargs: weights: Dict[str,float], average: bool)
        - 'combmnz' (kwargs: weights: Dict[str,float])
        - 'fisher'
    method_kwargs : dict of hyperparameters/weights for selected method.
        For weighted methods, pass {'weights': {'modelA': 1.5, 'modelB': 0.8, ...}}
    """
    method = method.lower()
    if method not in _FUSERS:
        raise ValueError(f"Unknown ensemble method: {method}")
    method_kwargs = dict(method_kwargs or {})

    ens_name = name or f"ensemble:{method}"
    safe = _sanitize_model_name(ens_name)

    output_dir = Path(output_dir)
    model_dir = output_dir / safe
    _ensure_dir(model_dir)

    run_dir = model_dir / "runs" / (run_hash or "CURRENT")
    _ensure_dir(run_dir)

    sim_path = run_dir / "similarity.parquet"
    manifest_path = run_dir / "manifest.json"

    # ── Overwrite / reuse guard (no silent staleness) ──
    if sim_path.exists() and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        same_inputs = (
            manifest.get("method") == method
            and (manifest.get("method_kwargs") or {}) == (method_kwargs or {})
            and set(manifest.get("source_models") or []) == set(model_to_run_dir.keys())
            and manifest.get("icd_fp")
            == _df_fingerprint(icd_df[["icd_code", "icd_string"]])
            and manifest.get("phecode_fp")
            == _df_fingerprint(phecode_df[["phecode", "phecode_string"]])
        )

        if same_inputs:
            if not overwrite:
                # identical inputs → reuse
                return
        else:
            if not overwrite:
                raise RuntimeError(
                    f"Existing ensemble at {run_dir} differs from current configuration, "
                    f"and overwrite=False."
                )
            # else: overwrite=True → fall through and rebuild

    # Load minimal columns from each model's run
    dfs = []
    for m, rdir in model_to_run_dir.items():
        parquet_file = Path(rdir) / "similarity.parquet"
        if not parquet_file.exists():
            continue
        df = pd.read_parquet(
            parquet_file, columns=["phecode", "icd_code", "score", "rank"]
        )
        df["model"] = m
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No similarity.parquet files found for the provided models.")

    all_sim = pd.concat(dfs, ignore_index=True)
    all_sim["phecode"] = all_sim["phecode"].astype(str)
    all_sim["icd_code"] = all_sim["icd_code"].astype(str)

    # ── compute min-of-max rank per phecode across models ──
    max_rank_per_model = all_sim.groupby(["phecode", "model"])["rank"].max()
    min_of_max_per_phe = max_rank_per_model.groupby("phecode").min().to_dict()

    # Lookups for labels
    icd_lookup = (
        icd_df[["icd_code", "icd_string"]]
        .copy()
        .assign(icd_code=lambda d: d["icd_code"].astype(str))
        .set_index("icd_code")
    )
    phe_lookup = (
        phecode_df[["phecode", "phecode_string"]]
        .copy()
        .assign(phecode=lambda d: d["phecode"].astype(str))
        .set_index("phecode")
    )

    rows = []
    n_icd = len(icd_lookup)
    n_phe = len(phe_lookup)
    created_at = _now()

    fuser = _FUSERS[method]
    for phe, sub in all_sim.groupby("phecode", sort=False):
        icds_sorted, scores_sorted = _apply_fuser(fuser, sub, method_kwargs)

        # truncate fused list to the shallowest model's cutoff for this phecode
        max_allowed = min_of_max_per_phe.get(phe, len(icds_sorted))
        icds_sorted = icds_sorted[:max_allowed]
        scores_sorted = scores_sorted[:max_allowed]

        phe_row = phe_lookup.loc[str(phe)]
        for rank, (icd, s) in enumerate(zip(icds_sorted, scores_sorted), start=1):
            icd_row = icd_lookup.loc[icd]
            rows.append(
                (
                    ens_name,
                    str(phe),
                    phe_row.phecode_string,
                    icd,
                    icd_row.icd_string,
                    float(s),
                    rank,
                    n_icd,
                    n_phe,
                    created_at,
                )
            )

    ens_df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "phecode",
            "phecode_string",
            "icd_code",
            "icd_string",
            "score",
            "rank",
            "n_icd",
            "n_phecodes",
            "created_at",
        ],
    )
    ens_df.to_parquet(sim_path, index=False)

    # Manifest
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "model_name": ens_name,
                "source_models": list(model_to_run_dir.keys()),
                "method": method,
                "method_kwargs": method_kwargs,
                "icd_fp": _df_fingerprint(icd_df[["icd_code", "icd_string"]]),
                "phecode_fp": _df_fingerprint(
                    phecode_df[["phecode", "phecode_string"]]
                ),
                "created_at": created_at,
            },
            f,
            indent=2,
        )
