from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import Iterable, Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch

from ._embed import build_st_model, encode_texts
from ._sim import semantic_search_topk
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
    st_search_kwargs : dict of kwargs passed to util.semantic_search()
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

    def get_ranked_icds(self, model: str, phecode: str) -> pd.DataFrame:
        """
        Return ranked ICD list for a single phecode+model (long-format DataFrame).
        """
        subdir = self._run_dir(model)
        p = subdir / "similarity.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        safe_model = sanitize_model_name(model)
        return (
            df[(df["model"] == safe_model) & (df["phecode"] == phecode)]
            .sort_values("rank")
            .reset_index(drop=True)
        )

    def get_all_results(self) -> pd.DataFrame:
        """
        Concatenate all similarity results across models (if present).
        """
        frames = []
        for m in self.models:
            subdir = self._run_dir(m)
            p = subdir / "similarity.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ───────────────────────── internal methods ────────────────────────
    def _run_one_model(self, model_name: str, overwrite: bool):
        safe = sanitize_model_name(model_name)
        model_dir = self.output_dir / "embeddings" / safe
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

        # Similarity via SentenceTransformers util.semantic_search
        icd_vecs32 = np.load(icd_npz)["X"].astype(np.float32)
        phe_vecs32 = np.load(phe_npz)["X"].astype(np.float32)
        scores_list, idx_list = semantic_search_topk(
            query=phe_vecs32,
            corpus=icd_vecs32,
            device=self.device,
            st_search_kwargs=self.st_search_kwargs,
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
            "search_kwargs": self.st_search_kwargs,
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
        return self.output_dir / "embeddings" / safe / "runs" / self.phecode_hash

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
