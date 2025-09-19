from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import Iterable, Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch

from ._embed import build_st_model, encode_texts
from ._sim import semantic_search_topk
from ._eval import rank_metrics_for_ks
from ._util import (
    sanitize_model_name,
    ensure_dir,
    df_fingerprint,
    now_iso,
    clean_kwargs,
)

class Phecoder:
    """
    Main user-facing API.

    Inputs
    ------
    icd_df : DataFrame with columns ['icd_code', 'icd_string']
    phecodes : DataFrame[['phecode','phecode_string']] OR str OR list[str]
    models : str or list[str] of SentenceTransformer model IDs
    output_dir : base path for all outputs

    Optional
    --------
    hf_home : set HF_HOME cache path (string)
    device : 'cuda', 'cpu', or None (auto-detect if None)
    dtype : 'float16' or 'float32' (storage for .npz embeddings)
    st_encode_kwargs : dict of kwargs passed to model.encode() (global)
    st_search_kwargs : dict of kwargs passed to util.semantic_search(). Default: top_k=1000.
    per_model_encode_kwargs : dict[str, dict] overrides for specific models
    """

    def __init__(
        self,
        icd_df: pd.DataFrame,
        phecodes: Union[pd.DataFrame, str, List[str]],
        models: Union[str, List[str]],
        output_dir: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float16",
        st_encode_kwargs: Optional[Dict[str, Any]] = None,
        st_search_kwargs: Optional[Dict[str, Any]] = None,
        per_model_encode_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        # Validate ICD
        req_icd = {"icd_code", "icd_string"}
        miss = req_icd - set(icd_df.columns)
        if miss:
            raise ValueError(f"icd_df missing required columns: {miss}")
        self.icd_df = icd_df[["icd_code", "icd_string"]].copy()

        # Normalize phecodes
        self.phecode_df = self._normalize_phecodes(phecodes)

        # Models
        self.models = [models] if isinstance(models, str) else list(models)

        # Paths / env
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        if hf_home:
            os.environ["HF_HOME"] = str(hf_home)

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Storage dtype
        if dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be 'float16' or 'float32'")
        self.storage_dtype = np.float16 if dtype == "float16" else np.float32

        # Encode/search kwargs
        self.st_encode_kwargs = clean_kwargs(st_encode_kwargs)
        self.per_model_encode_kwargs = per_model_encode_kwargs or {}
        self.st_search_kwargs = clean_kwargs(st_search_kwargs)
        # Sensible defaults if not provided
        self.st_encode_kwargs.setdefault("normalize_embeddings", True)
        self.st_encode_kwargs.setdefault("convert_to_numpy", True)
        self.st_encode_kwargs.setdefault("show_progress_bar", True)
        self.st_encode_kwargs.setdefault("trust_remote_code", True)
        # We own device placement at class level; ignore any user-supplied 'device'
        self.st_encode_kwargs.pop("device", None)

        # Fingerprints for skip-logic
        self.icd_fp = df_fingerprint(self.icd_df)
        self.phecode_fp = df_fingerprint(self.phecode_df)
        self.phecode_hash = hashlib.sha256(self.phecode_fp.encode()).hexdigest()[:16]

    # ─────────────────────────── public API ────────────────────────────
    def run(self, overwrite: bool = False):
        """
        Compute (or reuse) embeddings and semantic-search results for all models.
        """
        for model_name in self.models:
            self._run_one_model(model_name, overwrite=overwrite)

    def get_results(
        self,
        model: str | None = None,
        phecode: str | None = None,
    ) -> pd.DataFrame:
        """
        Unified results accessor.

        - If both `model` and `phecode` are None: return concatenated results across all models
        for the *current* phecode set (same behavior as old `get_all_results()`).
        - If `model` is provided: restrict to that model.
        - If `phecode` is provided: restrict to that phecode ID.
        Convenience: if `phecode` matches a row in `self.phecode_df['phecode_string']`
        (case-insensitive), it will be mapped to the internal phecode ID.

        Returns an empty DataFrame if nothing is found.
        """
        # determine which models to read
        if model is None:
            models_to_use = list(self.models)
            model_filter = None
        else:
            models_to_use = [model]
            model_filter = sanitize_model_name(model)

        # optional phecode resolution: allow passing the *string* label
        phe_filter = None
        if phecode is not None:
            # if user passed exact ID, keep; else try to map from phecode_string
            if (self.phecode_df is not None) and (phecode not in set(self.phecode_df["phecode"])):
                # try case-insensitive match on phecode_string
                m = self.phecode_df[self.phecode_df["phecode_string"].str.lower() == str(phecode).lower()]
                if not m.empty:
                    phe_filter = m.iloc[0]["phecode"]
                else:
                    phe_filter = phecode  # keep as-is; may still match older runs
            else:
                phe_filter = phecode

        frames: list[pd.DataFrame] = []
        for m in models_to_use:
            subdir = self._run_dir(m)
            p = subdir / "similarity.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            # ensure we only return rows that belong to the model directory
            if model_filter is not None:
                df = df[df["model"] == model_filter]
            if phe_filter is not None:
                df = df[df["phecode"] == phe_filter]
            if not df.empty:
                frames.append(df)

        if not frames:
            # standard column order if nothing found
            return pd.DataFrame(columns=[
                "model","phecode","phecode_string","icd_code","icd_string",
                "score","rank","n_icd","n_phecodes","created_at"
            ])

        return pd.concat(frames, ignore_index=True)


    def evaluate(
        self,
        phecode_ground_truth: pd.DataFrame,
        models: Optional[list[str]] = None,
        k: Optional[Union[int, Iterable[int]]] = None,
        include_curves: bool = False,
        drop_missing: bool = True,
        run_hash: Optional[str] = None,
    ):
        """
        Rank-based evaluation per phecode (per model) against a gold long-table.

        Parameters
        ----------
        phecode_ground_truth : DataFrame with columns ['phecode','icd_code'] (long format).
        models : optional subset of models to evaluate; defaults to self.models.
        k : None | int | list[int]
            - None  → evaluate over the full stored ranking (i.e., whatever top_k you saved).
            - int   → compute AUPRC@k, P@k, R@k.
            - list  → compute metrics for each k in the list and stack rows.
        include_curves : if True, returns (metrics_df, curves_df); else returns metrics_df.
        drop_missing : if True (default), only return results for phecodes present in BOTH phecode_ground_truth and the run.
                    If False, emit placeholder rows for phecodes missing in the run (n_considered=0, auprc=0, etc.).
        run_hash : provide run_hash if you wish to evaluate a previous run, and not the most recent.

        Returns
        -------
        if include_curves=False: metrics_df
        if include_curves=True : (metrics_df, curves_df)

        metrics_df columns:
        ['model','phecode','k','n_considered','n_gold_pos','auprc','precision_at_k','recall_at_k']

        curves_df (only if include_curves=True):
        ['model','phecode','curve_precision','curve_recall']
        (one row per (model, phecode); curves correspond to largest effective K across requested ks)
        """
        # validate gold
        req = {"phecode", "icd_code"}
        miss = req - set(phecode_ground_truth.columns)
        if miss:
            raise ValueError(f"phecode_ground_truth missing required columns: {miss}")
        phecode_ground_truth = phecode_ground_truth[["phecode", "icd_code"]].dropna().drop_duplicates()

        # normalize k(s)
        if k is None:
            k_values: list[Optional[int]] = [None]
        elif isinstance(k, int):
            k_values = [k]
        else:
            seen = set()
            k_values = []
            for kk in k:
                kk_norm = None if kk is None else int(kk)
                key = ("None" if kk_norm is None else kk_norm)
                if key not in seen:
                    seen.add(key)
                    k_values.append(kk_norm)

        # map gold phecode → set(ICDs)
        gold_map = {p: set(g["icd_code"].tolist())
                    for p, g in phecode_ground_truth.groupby("phecode", sort=False)}

        models_to_use = self.models if models is None else list(models)

        metrics_rows = []
        curves_rows = []

        for model_name in models_to_use:
            safe_model = sanitize_model_name(model_name)

            # Choose which run to read from
            if run_hash is None:
                # current run for this instance's phecode set
                subdir = self._run_dir(model_name)
            else:
                # explicit previous run
                subdir = self.output_dir / safe_model / "runs" / run_hash

            sim_path = subdir / "similarity.parquet"
            if not sim_path.exists():
                continue

            sim = pd.read_parquet(sim_path)
            sim = sim[sim["model"] == safe_model]

            # Determine phecodes available in this run (for this model)
            available_phecodes = set(sim["phecode"].unique())

            # Choose which phecodes to evaluate
            if drop_missing:
                target_phecodes = sorted(set(gold_map.keys()) & available_phecodes)
            else:
                target_phecodes = sorted(gold_map.keys())

            for phe in target_phecodes:
                gold_set = gold_map[phe]
                sub = sim[sim["phecode"] == phe][["icd_code", "rank"]]

                if sub.empty:
                    # Only hit when drop_missing=False
                    rows_for_k = []
                    for k_val in k_values:
                        eff_k = None if (k_val is None) else 0
                        rows_for_k.append({
                            "k": eff_k,
                            "n_considered": 0,
                            "n_gold_pos": int(len(gold_set)),
                            "auprc": 0.0,
                            "precision_at_k": float("nan") if eff_k == 0 else float("nan"),
                            "recall_at_k": 0.0,
                        })
                    for r in rows_for_k:
                        metrics_rows.append({
                            "model": safe_model,
                            "phecode": phe,
                            **r
                        })
                    if include_curves:
                        curves_rows.append({
                            "model": safe_model,
                            "phecode": phe,
                            "curve_precision": np.zeros(0, dtype=float),
                            "curve_recall": np.zeros(0, dtype=float),
                        })
                    continue

                # Normal path: we have results for this phecode
                rows_for_k, curve_P, curve_R, K_star = rank_metrics_for_ks(sub, gold_set, k_values)

                for r in rows_for_k:
                    metrics_rows.append({
                        "model": safe_model,
                        "phecode": phe,
                        **r
                    })
                if include_curves:
                    curves_rows.append({
                        "model": safe_model,
                        "phecode": phe,
                        "curve_precision": curve_P,
                        "curve_recall": curve_R,
                    })

        metrics_df = pd.DataFrame(
            metrics_rows,
            columns=["model","phecode","k","n_considered","n_gold_pos","auprc","precision_at_k","recall_at_k"]
        )

        if include_curves:
            curves_df = pd.DataFrame(
                curves_rows,
                columns=["model","phecode","curve_precision","curve_recall"]
            )
            return metrics_df, curves_df

        return metrics_df


    # ───────────────────────── internal methods ────────────────────────
    def _run_one_model(self, model_name: str, overwrite: bool):
        safe = sanitize_model_name(model_name)
        model_dir = self.output_dir / safe
        ensure_dir(model_dir)

        icd_idx_pq = model_dir / "icd_index.parquet"
        icd_npz = model_dir / "icd_embeds.npz"

        run_dir = model_dir / "runs" / self.phecode_hash
        ensure_dir(run_dir)
        manifest_path = run_dir / "manifest.json"
        sim_path = run_dir / "similarity.parquet"
        phe_idx_pq = run_dir / "phecode_index.parquet"
        phe_npz = run_dir / "phecode_embeds.npz"

        # Skip if already computed and fingerprints match
        if sim_path.exists() and manifest_path.exists() and not overwrite:
            with open(manifest_path) as f:
                man = json.load(f)
            if man.get("icd_fp") == self.icd_fp and man.get("phecode_fp") == self.phecode_fp:
                return

        # Build/load model
        model = build_st_model(model_name, device=self.device)

        # ICD embeddings (shared per model)
        if not (icd_npz.exists() and icd_idx_pq.exists()) or overwrite:
            enc_kwargs = self._encode_kwargs_for_model(model_name)
            icd_vecs = encode_texts(
                model=model,
                texts=self.icd_df["icd_string"].tolist(),
                encode_kwargs=enc_kwargs,
            ).astype(self.storage_dtype)
            self.icd_df.to_parquet(icd_idx_pq, index=False)
            np.savez_compressed(icd_npz, X=icd_vecs)

        # Phecode embeddings (per run / phecode set)
        enc_kwargs = self._encode_kwargs_for_model(model_name)
        phe_vecs = encode_texts(
            model=model,
            texts=self.phecode_df["phecode_string"].tolist(),
            encode_kwargs=enc_kwargs,
        ).astype(self.storage_dtype)
        self.phecode_df.to_parquet(phe_idx_pq, index=False)
        np.savez_compressed(phe_npz, X=phe_vecs)

        # Similarity via SentenceTransformers util
        icd_vecs32 = np.load(icd_npz)["X"].astype(np.float32)
        phe_vecs32 = np.load(phe_npz)["X"].astype(np.float32)

        # Default top_k = 1000 unless user specified; cap at corpus size
        search_kwargs = dict(self.st_search_kwargs)  # user-supplied overrides
        if "top_k" not in search_kwargs or search_kwargs["top_k"] is None:
            search_kwargs["top_k"] = min(1000, icd_vecs32.shape[0])
        else:
            # ensure int + safe cap
            search_kwargs["top_k"] = min(int(search_kwargs["top_k"]), icd_vecs32.shape[0])

        # Semantic search
        scores_list, idx_list = semantic_search_topk(
            query=phe_vecs32,
            corpus=icd_vecs32,
            device=self.device,
            st_search_kwargs=search_kwargs,
        )

        # Build long table
        rows = []
        safe_model = sanitize_model_name(model_name)
        n_icd = icd_vecs32.shape[0]
        n_phe = phe_vecs32.shape[0]
        for i, phe in enumerate(self.phecode_df.itertuples(index=False)):
            top_idx = idx_list[i]
            top_scores = scores_list[i]
            for rank, (j, s) in enumerate(zip(top_idx, top_scores), start=1):
                rows.append((
                    safe_model,
                    phe.phecode,
                    phe.phecode_string,
                    self.icd_df.iloc[j].icd_code,
                    self.icd_df.iloc[j].icd_string,
                    float(s),
                    rank,
                    n_icd,
                    n_phe,
                    now_iso()
                ))
        sim_df = pd.DataFrame(rows, columns=[
            "model","phecode","phecode_string","icd_code","icd_string",
            "score","rank","n_icd","n_phecodes","created_at"
        ])
        sim_df.to_parquet(sim_path, index=False)

        # Manifest for reproducibility/skip-logic
        manifest = {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "run_dir": str(run_dir),
            "icd_fp": self.icd_fp,
            "phecode_fp": self.phecode_fp,
            "storage_dtype": "float16" if self.storage_dtype==np.float16 else "float32",
            "device": self.device,
            "encode_kwargs": self._encode_kwargs_for_model(model_name),
            "search_kwargs": search_kwargs,   # <-- record effective top_k
            "created_at": now_iso(),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _encode_kwargs_for_model(self, model_name: str) -> Dict[str, Any]:
        """
        Merge global encode kwargs with per-model overrides.
        Never pass 'device' here; we control device at class level.
        """
        per = clean_kwargs(self.per_model_encode_kwargs.get(model_name))
        merged = {**self.st_encode_kwargs, **per}
        merged.pop("device", None)
        # IMPORTANT: do not pass batch_size unless explicitly set by user
        if "batch_size" in merged and merged["batch_size"] is None:
            merged.pop("batch_size")
        return merged

    def _run_dir(self, model_name: str) -> Path:
        safe = sanitize_model_name(model_name)
        return self.output_dir / safe / "runs" / self.phecode_hash

    @staticmethod
    def _normalize_phecodes(phecodes) -> pd.DataFrame:
        if isinstance(phecodes, pd.DataFrame):
            req = {"phecode", "phecode_string"}
            miss = req - set(phecodes.columns)
            if miss:
                raise ValueError(f"phecode df missing columns: {miss}")
            return phecodes[["phecode","phecode_string"]].copy()
        if isinstance(phecodes, str):
            return pd.DataFrame({"phecode": ["PHEC_0001"], "phecode_string": [phecodes]})
        if isinstance(phecodes, list):
            rows = [(f"PHEC_{i+1:04d}", s) for i, s in enumerate(phecodes)]
            return pd.DataFrame(rows, columns=["phecode","phecode_string"])
        raise TypeError("phecodes must be DataFrame, str, or list[str]")

    def list_runs(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        List stored runs.

        If `model` is provided, returns runs only for that model.
        If `model` is None, returns runs for **all** models found under `output_dir`.

        Returns a DataFrame with columns:
        ['model', 'run_hash', 'created_at', 'top_k', 'run_dir']
        """
        runs_dirs = []

        if model is None:
            # scan all model dirs under output_dir that contain a 'runs' subdir
            if self.output_dir.exists():
                for d in sorted(self.output_dir.iterdir()):
                    if d.is_dir():
                        r = d / "runs"
                        if r.is_dir():
                            runs_dirs.append(r)
        else:
            safe = sanitize_model_name(model)
            r = self.output_dir / safe / "runs"
            if r.is_dir():
                runs_dirs.append(r)

        rows = []
        for rdir in runs_dirs:
            model_safe = rdir.parent.name  # directory name above 'runs'
            for rd in sorted(rdir.iterdir()):
                if not rd.is_dir():
                    continue
                man = rd / "manifest.json"
                created_at = None
                top_k = None
                try:
                    if man.exists():
                        m = json.loads(man.read_text())
                        created_at = m.get("created_at")
                        top_k = (m.get("search_kwargs") or {}).get("top_k")
                except Exception:
                    # keep row with whatever we have
                    pass
                rows.append({
                    "model": model_safe,
                    "run_hash": rd.name,
                    "created_at": created_at,
                    "top_k": top_k,
                    "run_dir": str(rd),
                })

        df = pd.DataFrame(rows, columns=["model","run_hash","created_at","top_k","run_dir"])
        if not df.empty:
            # newest first; then model; then hash
            df = df.sort_values(["created_at","model","run_hash"], ascending=[False, True, True], na_position="last")
            df = df.reset_index(drop=True)
        return df

    def load_results_from_hash(self, model: str, run_hash: str) -> pd.DataFrame:
        safe = sanitize_model_name(model)
        p = self.output_dir / safe / "runs" / run_hash / "similarity.parquet"
        return pd.read_parquet(p)