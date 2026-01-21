from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Iterable, Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
import gc

from ._embed import _build_st_model, _encode_texts
from ._similarity import _semantic_search_topk
from ._evaluate import (
    _normalize_k_values,
    _vectorized_metrics_for_model,
    _warn_missing_codes,
)

from ._ensemble import _build_ensemble_from_runs
from .utils import (
    _sanitize_model_name,
    _ensure_dir,
    _df_fingerprint,
    _now,
    _clean_kwargs,
    _find_all_models,
    _resolve_model_dir,
)
from .utils import list_runs as list_runs_utils
from .utils import load_results as load_results_utils
from ._defaults import (
    MODEL_PRESETS,
    DEFAULT_MODEL_PRESET,
    DEFAULT_ENSEMBLE,
    DEFAULT_ENSEMBLE_KWARGS,
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
    st_search_kwargs : dict of kwargs passed to util._semantic_search(). Default: top_k=1000.
    per_model_encode_kwargs : dict[str, dict] overrides for specific models
    """

    def __init__(
        self,
        icd_df: pd.DataFrame,
        phecodes: Union[pd.DataFrame, str, List[str]],
        output_dir: str,
        models: Union[str, List[str]] = None,
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

        # Models + preset per-model kwargs
        preset_pm: Dict[str, Dict[str, Any]] = {}

        if models is None:
            preset = MODEL_PRESETS[DEFAULT_MODEL_PRESET]
            self.models = list(preset["models"])
            preset_pm = dict(preset["per_model_encode_kwargs"])
        elif isinstance(models, str) and models in MODEL_PRESETS:
            preset = MODEL_PRESETS[models]
            self.models = list(preset["models"])
            preset_pm = dict(preset["per_model_encode_kwargs"])
        else:
            self.models = [models] if isinstance(models, str) else list(models)

        if not self.models:
            raise ValueError("models cannot be empty")

        # per-model overrides: preset first, then user overrides
        self.per_model_encode_kwargs = {**preset_pm, **(per_model_encode_kwargs or {})}

        # Paths / env
        self.output_dir = Path(output_dir)
        _ensure_dir(self.output_dir)
        self.icd_cache_dir = Path(icd_cache_dir) if icd_cache_dir else None

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Storage dtype
        if dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be 'float16' or 'float32'")
        self.storage_dtype = np.float16 if dtype == "float16" else np.float32

        # Encode/search kwargs
        self.st_encode_kwargs = _clean_kwargs(st_encode_kwargs)
        self.per_model_encode_kwargs = per_model_encode_kwargs or {}
        self.st_search_kwargs = _clean_kwargs(st_search_kwargs)
        # Sensible defaults if not provided
        self.st_encode_kwargs.setdefault("normalize_embeddings", True)
        self.st_encode_kwargs.setdefault("convert_to_numpy", True)
        self.st_encode_kwargs.setdefault("show_progress_bar", True)
        self.st_encode_kwargs.setdefault("trust_remote_code", True)
        # We own device placement at class level; ignore any user-supplied 'device'
        self.st_encode_kwargs.pop("device", None)

        # Fingerprints for skip-logic
        self.icd_fp = _df_fingerprint(self.icd_df)
        self.phecode_fp = _df_fingerprint(self.phecode_df)
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

    def load_results(
        self,
        models: Union[str, Iterable[str], None] = None,
        phecode: str | None = None,
        phecode_ground_truth: pd.DataFrame | None = None,
        include_ensembles: bool = True,
    ) -> pd.DataFrame:
        return load_results_utils(
            output_dir=self.output_dir,
            phecode_hash=self.phecode_hash,
            phecode_df=self.phecode_df,
            models=models,
            phecode=phecode,
            phecode_ground_truth=phecode_ground_truth,
            include_ensembles=include_ensembles,
        )

    def build_ensemble(
        self,
        models: Optional[list[str]] = None,
        method: str = "default",
        method_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        run_hash: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Create an unsupervised ensemble from existing per-model runs and store it as a virtual model.
        """

        if isinstance(method, str) and method.lower() == "default":
            method = DEFAULT_ENSEMBLE
            method_kwargs = DEFAULT_ENSEMBLE_KWARGS

        models_to_use = list(self.models) if models is None else list(models)

        model_to_run_dir: Dict[str, str] = {}
        for m in models_to_use:
            subdir = (
                self._run_dir(m)
                if run_hash is None
                else (self.output_dir / _sanitize_model_name(m) / "runs" / run_hash)
            )
            if (subdir / "similarity.parquet").exists():
                model_to_run_dir[m] = str(subdir)

        if not model_to_run_dir:
            raise RuntimeError("No input runs found for the requested models/method.")

        rhash = run_hash or self.phecode_hash

        _build_ensemble_from_runs(
            output_dir=self.output_dir,
            phecode_df=self.phecode_df,
            icd_df=self.icd_df,
            model_to_run_dir=model_to_run_dir,
            method=method,
            method_kwargs=method_kwargs,
            name=name or f"ensemble:{method}",
            run_hash=rhash,
            overwrite=overwrite,
        )

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
        """
        # ---- validate & normalize gold ----
        req = {"phecode", "icd_code"}
        miss = req - set(phecode_ground_truth.columns)
        if miss:
            raise ValueError(f"phecode_ground_truth missing required columns: {miss}")
        gold_df = (
            phecode_ground_truth[["phecode", "icd_code"]].dropna().drop_duplicates()
        )
        gold_df["phecode"] = gold_df["phecode"].astype(str)

        k_values = _normalize_k_values(k)

        run_phecodes = set(self.phecode_df["phecode"].astype(str))
        gold_phecodes = set(gold_df["phecode"].astype(str))
        only_in_run = sorted(run_phecodes - gold_phecodes)
        only_in_gold = sorted(gold_phecodes - run_phecodes)
        if only_in_run:
            print(
                f"[evaluate] phecodes in phecode_df but NOT in phecode_ground_truth (count={len(only_in_run)}): {only_in_run}",
                flush=True,
            )
        if only_in_gold:
            print(
                f"[evaluate] phecodes in phecode_ground_truth but NOT in phecode_df (count={len(only_in_gold)}): {only_in_gold}",
                flush=True,
            )

        if models is None:
            found_models = _find_all_models(
                self.output_dir, (run_hash or self.phecode_hash)
            )
            models_to_use = list(found_models.keys())
        else:
            found_models = None
            models_to_use = list(models)

        metrics_blocks, curves_blocks = [], []
        icd_universe = set(self.icd_df["icd_code"])

        # ---- per model ----
        for i, model_name in enumerate(models_to_use):
            if found_models is not None:
                subdir = Path(found_models[model_name])
            elif run_hash is None:
                subdir = self._run_dir(model_name)
            else:
                subdir = _resolve_model_dir(self.output_dir, model_name, run_hash)
                if subdir is None:
                    continue

            # parquet loading handled here
            sim_path = subdir / "similarity.parquet"
            if not sim_path.exists():
                continue

            sim = pd.read_parquet(
                sim_path,
                columns=["model", "phecode", "icd_code", "rank"],
                engine="pyarrow",
                filters=[("model", "=", model_name)],
            )
            if sim.empty:
                continue

            available_phe = set(sim["phecode"].astype(str).unique())
            target_phe = gold_phecodes & available_phe
            if not target_phe:
                continue
            sim = sim[sim["phecode"].astype(str).isin(target_phe)]

            # warn/save missing Phecodes and ICDs during first loop
            if i == 0:
                missing_df = _warn_missing_codes(gold_df, icd_universe, target_phe)
                if missing_df is not None:
                    missing_df.to_csv(
                        self.output_dir / "excluded_codes.csv", index=False
                    )

            m_block, c_block = _vectorized_metrics_for_model(
                sim, gold_df, k_values, include_curves, model_name
            )
            if not m_block.empty:
                metrics_blocks.append(m_block)
            if include_curves and c_block is not None and not c_block.empty:
                curves_blocks.append(c_block)

        # ---- write outputs ----
        if metrics_blocks:
            metrics_df = pd.concat(metrics_blocks, ignore_index=True)
        else:
            metrics_df = pd.DataFrame(
                columns=[
                    "model",
                    "phecode",
                    "k",
                    "n_considered",
                    "n_gold_pos",
                    "AP@k",
                    "P@k",
                    "R@k",
                ]
            )
        metrics_df.to_parquet(self.output_dir / "metrics.parquet", index=False)

        if include_curves:
            if curves_blocks:
                curves_df = pd.concat(curves_blocks, ignore_index=True)
            else:
                curves_df = pd.DataFrame(
                    columns=["model", "phecode", "curve_precision", "curve_recall"]
                )
            curves_df.to_parquet(self.output_dir / "pr_curves.parquet", index=False)

    def list_runs(
        self,
        models: Optional[Union[str, Iterable[str]]] = None,
        include_ensembles: bool = True,
    ) -> pd.DataFrame:
        """
        List stored runs (base models + optionally ensembles).

        Parameters
        ----------
        models : str or iterable of str, optional
            Restrict to a specific model or set of models.
        include_ensembles : bool, default=True
            Whether to include ensemble model folders (starting with 'ens').

        Returns
        -------
        DataFrame
            Columns: ['model', 'run_hash', 'created_at', 'top_k', 'run_dir']
        """
        return list_runs_utils(
            output_dir=self.output_dir,
            models=models,
            include_ensembles=include_ensembles,
        )

    # ───────────────────────── internal methods ────────────────────────
    def _run_one_model(self, model_name: str, overwrite: bool):
        safe = _sanitize_model_name(model_name)
        model_dir = self.output_dir / safe
        _ensure_dir(model_dir)

        # Determine ICD embedding base directory
        icd_base_dir = (self.icd_cache_dir / safe) if self.icd_cache_dir else model_dir
        _ensure_dir(icd_base_dir)

        # ICD-level artifacts (shared per model)
        icd_index_path = icd_base_dir / "icd_index.parquet"
        icd_embeddings_path = icd_base_dir / "icd_embeds.npz"
        icd_manifest_path = icd_base_dir / "icd_manifest.json"

        # Run-specific artifacts (per phecode set)
        run_dir = model_dir / "runs" / self.phecode_hash
        _ensure_dir(run_dir)
        run_manifest_path = run_dir / "manifest.json"
        sim_path = run_dir / "similarity.parquet"
        phe_index_path = run_dir / "phecode_index.parquet"
        phe_embeddings_path = run_dir / "phecode_embeds.npz"

        # Skip or raise based on manifest consistency
        if sim_path.exists() and run_manifest_path.exists():
            with open(run_manifest_path) as f:
                man = json.load(f)

            same_inputs = (
                man.get("icd_fp") == self.icd_fp
                and man.get("phecode_fp") == self.phecode_fp
            )

            if same_inputs:
                if not overwrite:
                    # Reuse: nothing to do
                    return
                # Overwrite=True → rebuild anyway (fresh run)
            else:
                if not overwrite:
                    reasons = []
                    if man.get("icd_fp") != self.icd_fp:
                        reasons.append("ICD fingerprint changed")
                    if man.get("phecode_fp") != self.phecode_fp:
                        reasons.append("Phecode fingerprint changed")
                    raise RuntimeError(
                        f"Existing run found at {run_dir} but inputs differ and overwrite=False.\n"
                        f"Reasons: {', '.join(reasons)}"
                    )

        # Build/load model
        model = _build_st_model(model_name, device=self.device)

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
            icd_vecs = _encode_texts(
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
                        "created_at": _now(),
                    },
                    f,
                    indent=2,
                )

        # ---------- Phecode embeddings (per run) ----------
        phe_vecs = _encode_texts(
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
            search_kwargs["top_k"] = min(int(search_kwargs["top_k"]), icd_embs.shape[0])

        scores_list, idx_list = _semantic_search_topk(
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
                        _now(),
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
            "created_at": _now(),
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
        per = _clean_kwargs(self.per_model_encode_kwargs.get(model_name))
        merged = {**self.st_encode_kwargs, **per}
        merged.pop("device", None)
        # IMPORTANT: do not pass batch_size unless explicitly set by user
        if "batch_size" in merged and merged["batch_size"] is None:
            merged.pop("batch_size")
        return merged

    def _run_dir(self, model_name: str) -> Path:
        safe = _sanitize_model_name(model_name)
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
