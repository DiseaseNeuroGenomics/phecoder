from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Iterable, Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
import gc

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
    icd_cache_dir : str, optional
        Alternate base path for ICD embeddings/manifests (per model).
        Defaults to output_dir if not provided.
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
        icd_cache_dir: Optional[str] = None,
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
        self.icd_cache_dir = Path(icd_cache_dir) if icd_cache_dir else None

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

    def download_models(self):
        """
        Downloads models, nothing else. 
        """
        for model_name in self.models:
            snapshot_download(repo_id=model_name)     

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
            model_filter = model

        # optional phecode resolution: allow passing the *string* label
        phe_filter = None
        if phecode is not None:
            # if user passed exact ID, keep; else try to map from phecode_string
            if (self.phecode_df is not None) and (
                phecode not in set(self.phecode_df["phecode"])
            ):
                # try case-insensitive match on phecode_string
                m = self.phecode_df[
                    self.phecode_df["phecode_string"].str.lower()
                    == str(phecode).lower()
                ]
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

        return pd.concat(frames, ignore_index=True)

    def evaluate(
        self,
        phecode_ground_truth: pd.DataFrame,
        models: Optional[list[str]] = None,
        k: Optional[Union[int, Iterable[int]]] = None,
        include_curves: bool = False,
        run_hash: Optional[str] = None,
    ):
        """
        Rank-based evaluation per phecode (per model) against a gold long-table.

        Changes vs. earlier version:
        - Always drops phecodes missing from either side.
        - Prints phecodes only in self.phecode_df vs only in gold.
        - If k=None is passed, replaces with max rank for each phecode.
        - Faster: minimal columns, prefilter, presort, single groupby pass.
        """
        # ---- validate & normalize gold ----
        req = {"phecode", "icd_code"}
        miss = req - set(phecode_ground_truth.columns)
        if miss:
            raise ValueError(f"phecode_ground_truth missing required columns: {miss}")
        gold_df = (
            phecode_ground_truth[["phecode", "icd_code"]]
            .dropna()
            .drop_duplicates()
        )

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
                key = "None" if kk_norm is None else kk_norm
                if key not in seen:
                    seen.add(key)
                    k_values.append(kk_norm)

        # --- One-time reporting: set diffs between run's phecode_df and gold ---
        run_phecodes = set(self.phecode_df["phecode"].astype(str))
        gold_phecodes = set(gold_df["phecode"].astype(str))

        only_in_run = sorted(run_phecodes - gold_phecodes)
        only_in_gold = sorted(gold_phecodes - run_phecodes)

        if len(only_in_run)>0:
            print(f"[evaluate] phecodes in phecode_df but NOT in phecode_ground_truth (count={len(only_in_run)}): {only_in_run}", flush=True)
        if len(only_in_gold)>0:
            print(f"[evaluate] phecodes in phecode_ground_truth but NOT in phecode_df (count={len(only_in_gold)}): {only_in_gold}", flush=True)

        # Map gold phecode → set(ICDs)
        gold_map: Dict[str, set] = {
            str(p): set(g["icd_code"].tolist())
            for p, g in gold_df.groupby("phecode", sort=False)
        }

        models_to_use = self.models if models is None else list(models)

        metrics_rows: list[Dict[str, Any]] = []
        curves_rows: list[Dict[str, Any]] = []

        # ---- per model ----
        for model_name in models_to_use:
            safe_model = sanitize_model_name(model_name)

            # choose run dir
            if run_hash is None:
                subdir = self._run_dir(model_name)
            else:
                subdir = self.output_dir / safe_model / "runs" / run_hash

            sim_path = subdir / "similarity.parquet"
            if not sim_path.exists():
                continue

            # read only needed columns
            sim = pd.read_parquet(sim_path, columns=["model", "phecode", "icd_code", "rank"])
            sim = sim[sim["model"] == model_name]
            if sim.empty:
                continue

            # intersection of gold phecodes and available phecodes
            available_phecodes = set(sim["phecode"].astype(str).unique())
            target_phecodes = gold_phecodes & available_phecodes
            if not target_phecodes:
                continue

            sim = sim[sim["phecode"].astype(str).isin(target_phecodes)]
            sim = sim.sort_values(["phecode", "rank", "icd_code"], kind="mergesort")

            # grouped evaluation
            for phe, sub in sim.groupby("phecode", sort=False):
                phe_str = str(phe)
                gold_set = gold_map.get(phe_str, set())

                max_rank = int(sub["rank"].max())
                k_eff = [max_rank if kk is None else kk for kk in k_values]

                rows_for_k, curve_P, curve_R, K_star = rank_metrics_for_ks(
                    sub[["icd_code", "rank"]], gold_set, k_eff
                )
                for r in rows_for_k:
                    metrics_rows.append({"model": model_name, "phecode": phe_str, **r})
                if include_curves:
                    curves_rows.append(
                        {
                            "model": model_name,
                            "phecode": phe_str,
                            "curve_precision": curve_P,
                            "curve_recall": curve_R,
                        }
                    )

        metrics_df = pd.DataFrame(
            metrics_rows,
            columns=[
                "model",
                "phecode",
                "k",
                "n_considered",
                "n_gold_pos",
                "AP@k",
                "P@k",
                "R@k",
            ],
        )

        if include_curves:
            curves_df = pd.DataFrame(
                curves_rows,
                columns=["model", "phecode", "curve_precision", "curve_recall"],
            )
            return metrics_df, curves_df

        return metrics_df


    # ───────────────────────── internal methods ────────────────────────
    def _run_one_model(self, model_name: str, overwrite: bool):
        safe = sanitize_model_name(model_name)
        model_dir = self.output_dir / safe
        ensure_dir(model_dir)

        # Determine ICD embedding base directory
        icd_base_dir = (self.icd_cache_dir / safe) if self.icd_cache_dir else model_dir
        ensure_dir(icd_base_dir)

        # ICD-level artifacts (shared per model)
        icd_index_path = icd_base_dir / "icd_index.parquet"
        icd_embeddings_path = icd_base_dir / "icd_embeds.npz"
        icd_manifest_path = icd_base_dir / "icd_manifest.json"

        # Run-specific artifacts (per phecode set)
        run_dir = model_dir / "runs" / self.phecode_hash
        ensure_dir(run_dir)
        run_manifest_path = run_dir / "manifest.json"
        sim_path = run_dir / "similarity.parquet"
        phe_index_path = run_dir / "phecode_index.parquet"
        phe_embeddings_path = run_dir / "phecode_embeds.npz"

        # Skip run entirely if similarity already computed and manifests match
        if sim_path.exists() and run_manifest_path.exists() and not overwrite:
            with open(run_manifest_path) as f:
                man = json.load(f)
            if (
                man.get("icd_fp") == self.icd_fp
                and man.get("phecode_fp") == self.phecode_fp
            ):
                return

        # Build/load model
        model = build_st_model(model_name, device=self.device)

        # ---------- ICD embeddings (shared per model) ----------
        enc_kwargs_eff = self._encode_kwargs_for_model(model_name)
        storage_tag = "float16" if self.storage_dtype is np.float16 else "float32"

        need_build = False
        mismatch_reasons: list[str] = []

        # files present?
        have_files = icd_embeddings_path.exists() and icd_index_path.exists()

        if have_files and icd_manifest_path.exists():
            try:
                mf = json.loads(icd_manifest_path.read_text())
                if mf.get("icd_fp") != self.icd_fp:
                    mismatch_reasons.append("icd_fp differs (ICD corpus changed)")
                if mf.get("storage_dtype") != storage_tag:
                    mismatch_reasons.append(
                        f"storage_dtype differs (was {mf.get('storage_dtype')}, now {storage_tag})"
                    )
                if mf.get("encode_kwargs") != enc_kwargs_eff:
                    mismatch_reasons.append("encode_kwargs differ")
            except Exception as e:
                mismatch_reasons.append(f"manifest unreadable: {e!r}")

        if not have_files:
            need_build = True
            mismatch_reasons = []  # not a mismatch; just missing
        elif mismatch_reasons:
            if not overwrite:
                reasons = "; ".join(mismatch_reasons)
                raise RuntimeError(
                    "ICD embeddings exist but are incompatible with current settings, "
                    f"and overwrite=False. Refusing to rebuild.\nReasons: {reasons}\n"
                    f"Files:\n  index: {icd_index_path}\n  embeds: {icd_embeddings_path}\n  manifest: {icd_manifest_path}"
                )
            else:
                need_build = True

        if need_build:
            icd_vecs = encode_texts(
                model=model,
                texts=self.icd_df["icd_string"].tolist(),
                encode_kwargs=enc_kwargs_eff,
            ).astype(self.storage_dtype)
            tmp_icd = icd_embeddings_path.with_name(
                f"{icd_embeddings_path.stem}.tmp.npz"
            )
            np.savez_compressed(tmp_icd, X=icd_vecs)
            tmp_icd.replace(icd_embeddings_path)
            self.icd_df.to_parquet(icd_index_path, index=False)
            with open(icd_manifest_path, "w") as f:
                json.dump(
                    {
                        "model_name": model_name,
                        "icd_fp": self.icd_fp,
                        "storage_dtype": storage_tag,
                        "encode_kwargs": enc_kwargs_eff,
                        "created_at": now_iso(),
                    },
                    f,
                    indent=2,
                )

        # ---------- Phecode embeddings (per run) ----------
        phe_vecs = encode_texts(
            model=model,
            texts=self.phecode_df["phecode_string"].tolist(),
            encode_kwargs=enc_kwargs_eff,
        ).astype(self.storage_dtype)
        tmp_phe = phe_embeddings_path.with_name(f"{phe_embeddings_path.stem}.tmp.npz")
        np.savez_compressed(tmp_phe, X=phe_vecs)
        tmp_phe.replace(phe_embeddings_path)
        self.phecode_df.to_parquet(phe_index_path, index=False)

        # ---------- Delete model ----------
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------- Similarity search ----------
        icd_embs = np.load(icd_embeddings_path)["X"]
        phe_embs = np.load(phe_embeddings_path)["X"]

        search_kwargs = dict(self.st_search_kwargs)
        if "top_k" not in search_kwargs or search_kwargs["top_k"] is None:
            search_kwargs["top_k"] = min(1000, icd_embs.shape[0])
        else:
            search_kwargs["top_k"] = min(
                int(search_kwargs["top_k"]), icd_embs.shape[0]
            )

        scores_list, idx_list = semantic_search_topk(
            query=phe_embs,
            corpus=icd_embs,
            device=self.device,
            st_search_kwargs=search_kwargs,
        )

        # ---------- Build similarity long-table ----------
        rows = []
        n_icd = icd_embs.shape[0]
        n_phe = phe_embs.shape[0]
        for i, phe in enumerate(self.phecode_df.itertuples(index=False)):
            top_idx = idx_list[i]
            top_scores = scores_list[i]
            for rank, (j, s) in enumerate(zip(top_idx, top_scores), start=1):
                rows.append(
                    (
                        model_name,
                        phe.phecode,
                        phe.phecode_string,
                        self.icd_df.iloc[j].icd_code,
                        self.icd_df.iloc[j].icd_string,
                        float(s),
                        rank,
                        n_icd,
                        n_phe,
                        now_iso(),
                    )
                )
        sim_df = pd.DataFrame(
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
        sim_df.to_parquet(sim_path, index=False)

        # ---------- Run-level manifest ----------
        run_manifest = {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "run_dir": str(run_dir),
            "icd_fp": self.icd_fp,
            "phecode_fp": self.phecode_fp,
            "storage_dtype": storage_tag,
            "device": self.device,
            "encode_kwargs": enc_kwargs_eff,
            "search_kwargs": search_kwargs,
            "created_at": now_iso(),
            # Paths for reproducibility
            "icd_index_path": str(icd_index_path),
            "icd_embeddings_path": str(icd_embeddings_path),
            "icd_manifest_path": str(icd_manifest_path),
        }
        with open(run_manifest_path, "w") as f:
            json.dump(run_manifest, f, indent=2)

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
            return phecodes[["phecode", "phecode_string"]].copy()
        if isinstance(phecodes, str):
            return pd.DataFrame(
                {"phecode": ["PHEC_0001"], "phecode_string": [phecodes]}
            )
        if isinstance(phecodes, list):
            rows = [(f"PHEC_{i + 1:04d}", s) for i, s in enumerate(phecodes)]
            return pd.DataFrame(rows, columns=["phecode", "phecode_string"])
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
                rows.append(
                    {
                        "model": model,
                        "run_hash": rd.name,
                        "created_at": created_at,
                        "top_k": top_k,
                        "run_dir": str(rd),
                    }
                )

        df = pd.DataFrame(
            rows, columns=["model", "run_hash", "created_at", "top_k", "run_dir"]
        )
        if not df.empty:
            # newest first; then model; then hash
            df = df.sort_values(
                ["created_at", "model", "run_hash"],
                ascending=[False, True, True],
                na_position="last",
            )
            df = df.reset_index(drop=True)
        return df

    def load_results_from_hash(self, model: str, run_hash: str) -> pd.DataFrame:
        safe = sanitize_model_name(model)
        p = self.output_dir / safe / "runs" / run_hash / "similarity.parquet"
        return pd.read_parquet(p)
