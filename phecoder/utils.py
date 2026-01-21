from __future__ import annotations
from pathlib import Path
import pandas as pd
import hashlib
import datetime
import re
import json
from typing import Any, Dict, Iterator, Tuple, Optional, Iterable, Union


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _sanitize_model_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", s)


def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Stable, order-sensitive fingerprint of the dataframe contents & columns.
    """
    h = hashlib.sha256()
    h.update(("|".join(df.columns)).encode())
    # use itertuples for predictable ordering
    for row in df.itertuples(index=False, name=None):
        h.update(("|".join("" if x is None else str(x) for x in row)).encode())
        h.update(b"\n")
    return h.hexdigest()


def _now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _clean_kwargs(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Drop keys with None (so ST uses its internal defaults).
    """
    if not d:
        return {}
    out = dict(d)
    for k in list(out.keys()):
        if out[k] is None:
            out.pop(k)
    return out


def _resolve_model_dir(dir: Path, model: str, run_hash: str) -> Optional[Path]:
    """
    Return the run directory for `model` and `run_hash`, trying both the raw name
    and its sanitized folder name. None if not found.
    """
    safe = _sanitize_model_name(model)
    candidates = [
        dir / model / "runs" / run_hash,
        dir / safe / "runs" / run_hash,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_model_key(found: Dict[str, Path], query: str) -> Optional[str]:
    """
    Given a {model_name -> Path} mapping, find the key matching `query` via:
      1) exact match
      2) case-insensitive match
      3) equality after _sanitize_model_name
    """
    if query in found:
        return query
    ql = query.casefold()
    for k in found.keys():
        if k.casefold() == ql:
            return k
    qsafe = _sanitize_model_name(query)
    for k in found.keys():
        if _sanitize_model_name(k) == qsafe:
            return k
    return None


def _find_all_models(
    dir: Path,
    run_hash: str,
    include_ensembles: bool = True,
) -> Dict[str, Path]:
    """
    Find all models (and optionally ensembles) that have
    similarity.parquet under: dir/*/runs/<run_hash>.

    Returns {model_name -> run_dir Path}.
    """
    found: Dict[str, Path] = {}
    if not dir.exists():
        return found

    for d in sorted(dir.iterdir()):
        if not d.is_dir():
            continue
        rdir = d / "runs" / run_hash
        sim_path = rdir / "similarity.parquet"
        if not sim_path.exists():
            continue

        # Determine display model_name (prefer manifest → parquet → folder)
        manifest_path = rdir / "manifest.json"
        model_name: Optional[str] = None

        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                model_name = m.get("model_name")
            except Exception:
                model_name = None

        if not model_name:
            try:
                mcol = pd.read_parquet(sim_path, columns=["model"])
                if not mcol.empty:
                    model_name = mcol["model"].mode().iat[0]
            except Exception:
                model_name = None

        if not model_name:
            model_name = d.name  # fallback

        # Optional ensemble filtering (support "ens:*" and "ensemble:*")
        if not include_ensembles:
            mlow = str(model_name).lower()
            if mlow.startswith(("ens", "ensemble")):
                continue

        found[str(model_name)] = rdir

    return found


def _iter_run_dirs(
    dir: Path,
    models: Optional[Union[str, Iterable[str]]] = None,
) -> Iterator[Tuple[str, str, Path]]:
    """
    Yield (model_name, run_hash, run_dir Path) for every run with similarity.parquet.
    If `models` is provided (str or iterable of str), only iterate those models' runs.
    Matching is done against both raw and sanitized folder names.
    """
    if not dir.exists():
        return

    # Build the set of model directories to scan
    if models is None:
        model_dirs = [d for d in sorted(dir.iterdir()) if d.is_dir()]
    else:
        requested = [models] if isinstance(models, str) else list(models)
        cand_paths = []
        seen = set()
        for m in requested:
            raw = dir / str(m)
            safe = dir / _sanitize_model_name(str(m))
            for p in (raw, safe):
                if p.is_dir():
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        cand_paths.append(p)
        model_dirs = cand_paths

    for md in model_dirs:
        runs_dir = md / "runs"
        if not runs_dir.is_dir():
            continue
        for rd in sorted(runs_dir.iterdir()):
            if not rd.is_dir():
                continue
            sim_path = rd / "similarity.parquet"
            if not sim_path.exists():
                continue

            manifest_path = rd / "manifest.json"
            model_name: Optional[str] = None

            if manifest_path.exists():
                try:
                    m = json.loads(manifest_path.read_text())
                    model_name = m.get("model_name")
                except Exception:
                    model_name = None

            if not model_name:
                try:
                    mcol = pd.read_parquet(sim_path, columns=["model"])
                    if not mcol.empty:
                        model_name = mcol["model"].mode().iat[0]
                except Exception:
                    model_name = None

            if not model_name:
                model_name = md.name

            yield (str(model_name), rd.name, rd)


def _annotate_known_icds(
    results: pd.DataFrame,
    phecode_ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Add `is_known` (1 if ICD belongs to ground truth for phecode, else 0)."""
    results = results.copy()
    results["icd_code"] = results["icd_code"].astype(str)
    phecode_ground_truth = phecode_ground_truth.copy()
    phecode_ground_truth["icd_code"] = phecode_ground_truth["icd_code"].astype(str)

    # fast membership via merge
    merged = results.merge(
        phecode_ground_truth[["phecode", "icd_code"]],
        on=["phecode", "icd_code"],
        how="left",
        indicator=True,
    )

    # mark known/novel
    merged["is_known"] = (merged["_merge"] == "both").astype("int8")
    merged.drop(columns=["_merge"], inplace=True)

    return merged


def list_runs(
    dir: Path | str,
    models: Optional[Union[str, Iterable[str]]] = None,
    include_ensembles: bool = True,
) -> pd.DataFrame:
    """
    Enumerate stored model runs under an output directory.

    Returns columns: ['model', 'run_hash', 'created_at', 'top_k', 'run_dir']
    """
    dir = Path(dir)

    rows = []
    for model_name, run_hash, run_dir in _iter_run_dirs(dir, models=models):
        # filter out ensembles if requested (support both "ens:*" and "ensemble:*")
        if not include_ensembles:
            mlow = str(model_name).lower()
            if mlow.startswith(("ens", "ensemble")):
                continue

        man = run_dir / "manifest.json"
        created_at, top_k = None, None
        if man.exists():
            try:
                m = json.loads(man.read_text())
                created_at = m.get("created_at")
                top_k = (m.get("search_kwargs") or {}).get("top_k")
                if top_k is None:
                    top_k = m.get("fusion_top_k")
            except Exception:
                pass

        rows.append(
            {
                "model": model_name,
                "run_hash": run_hash,
                "created_at": created_at,
                "top_k": top_k,
                "run_dir": str(run_dir),
            }
        )

    df = pd.DataFrame(
        rows, columns=["model", "run_hash", "created_at", "top_k", "run_dir"]
    )
    if not df.empty:
        df = df.sort_values(
            ["created_at", "model", "run_hash"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
    return df


def load_results(
    dir: Path | str,
    phecode: Union[str, Iterable[str], None] = None,
    phecode_string: Union[str, Iterable[str], None] = None,
    models: Union[str, Iterable[str], None] = None,
    phecode_ground_truth: pd.DataFrame | None = None,
    include_ensembles: bool = False,
    run_hash: str | None = None,
) -> pd.DataFrame:
    """
    Load existing similarity.parquet outputs for one or more models,
    optionally filtering by phecode(s) or phecode_string(s), and annotating
    with a gold standard ICD→Phecode map.
    """
    dir = Path(dir)

    # --- normalize input types ---
    if isinstance(phecode, str):
        phecodes = [phecode]
    elif phecode is not None:
        phecodes = list(phecode)
    else:
        phecodes = None

    if isinstance(phecode_string, str):
        phecode_strings = [phecode_string.lower()]
    elif phecode_string is not None:
        phecode_strings = [s.lower() for s in phecode_string]
    else:
        phecode_strings = None

    # --- pick a run hash if not provided ---
    if run_hash is None:
        runs_df = list_runs(dir, include_ensembles=include_ensembles)
        if runs_df.empty:
            raise FileNotFoundError(f"No runs found under {dir}.")
        run_hash = runs_df.iloc[0]["run_hash"]

    # --- decide which models to read ---
    found_models: Optional[Dict[str, Path]]
    if models is None:
        found_models = _find_all_models(
            dir, run_hash, include_ensembles=include_ensembles
        )
        models_to_use = list(found_models.keys())
    else:
        models_to_use = [models] if isinstance(models, str) else list(models)
        found_models = None

    frames = []
    for m in models_to_use:
        # resolve model directory
        if found_models is not None:
            key = m if m in found_models else _resolve_model_key(found_models, m)
            if key is None:
                continue
            subdir = Path(found_models[key])
        else:
            subdir = _resolve_model_dir(dir, m, run_hash)
            if subdir is None:
                continue

        f = subdir / "similarity.parquet"
        if not f.exists():
            continue

        df = pd.read_parquet(f)

        # --- apply filters ---
        if phecodes is not None:
            df = df[df["phecode"].astype(str).isin(map(str, phecodes))]

        if phecode_strings is not None:
            df = df[df["phecode_string"].str.lower().isin(phecode_strings)]

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(
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
            ]
        )

    results = pd.concat(frames, ignore_index=True)

    if phecode_ground_truth is not None and not phecode_ground_truth.empty:
        results = _annotate_known_icds(results, phecode_ground_truth)

    return results
