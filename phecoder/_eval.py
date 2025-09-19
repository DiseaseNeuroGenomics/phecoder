# phecoder/_eval.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Iterable, Set
import numpy as np
import pandas as pd

def _build_rank_binary(codes: np.ndarray, gold_set: Set[str]) -> np.ndarray:
    # vectorized membership aligned to rank order
    return np.fromiter((c in gold_set for c in codes), dtype=np.int8, count=codes.size)

def _precompute_arrays(y: np.ndarray, n_pos_total: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """precision@r, recall@r, and indices of positive ranks (0-based)."""
    n = y.size
    if n == 0:
        Z = np.zeros(0, dtype=np.float64)
        return Z, Z, np.zeros(0, dtype=np.int64)
    cum_tp = np.cumsum(y, dtype=np.int64)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    precision = (cum_tp / ranks).astype(np.float64)
    recall = (cum_tp / float(n_pos_total)).astype(np.float64) if n_pos_total > 0 else np.zeros_like(precision)
    pos_idx = np.flatnonzero(y == 1).astype(np.int64)
    return precision, recall, pos_idx

def _auprc_at_k(precision: np.ndarray, pos_idx: np.ndarray, n_pos_total: int, k: Optional[int]) -> float:
    """Rank-aware AP (AUPRC). If k is provided, consider positives with rank ≤ k."""
    if n_pos_total == 0 or pos_idx.size == 0:
        return 0.0
    if k is None:
        return float(precision[pos_idx].sum() / n_pos_total)
    mask = pos_idx < int(k)
    if not np.any(mask):
        return 0.0
    return float(precision[pos_idx[mask]].sum() / n_pos_total)

def _precision_at_k(precision: np.ndarray, k: int) -> float:
    if precision.size == 0:
        return float("nan")
    k_eff = max(1, min(int(k), precision.size))
    return float(precision[k_eff - 1])

def _recall_at_k(recall: np.ndarray, n_pos_total: int, k: int) -> float:
    if n_pos_total == 0 or recall.size == 0:
        return 0.0
    k_eff = max(1, min(int(k), recall.size))
    return float(recall[k_eff - 1])

def rank_metrics_for_ks(
    ranked_df: pd.DataFrame,
    gold_icds: Iterable[str],
    ks: list[Optional[int]],
) -> Tuple[list[Dict[str, Any]], np.ndarray, np.ndarray, int]:
    """
    Core evaluator: compute metrics for multiple K in one pass, and return
    curves (precision[], recall[]) for the largest effective K as arrays.
    """
    # Sort to ensure deterministic rank order
    if not ranked_df.empty:
        ranked_df = ranked_df.sort_values(["rank", "icd_code"], kind="mergesort")

    codes = ranked_df["icd_code"].to_numpy() if not ranked_df.empty else np.array([], dtype=object)
    gold_set = set(gold_icds)
    y = _build_rank_binary(codes, gold_set)
    n_total_pos = len(gold_set)
    prec_arr, rec_arr, pos_idx = _precompute_arrays(y, n_total_pos)

    N = y.size
    metrics_rows = []
    # Pick one K* to produce a single pair of curves (largest effective K)
    eff_Ks = [(None if k is None else min(int(k), N)) for k in ks]
    K_star = max([ek for ek in eff_Ks if ek is not None], default=N)

    for k, eff_k in zip(ks, eff_Ks):
        if k is None:
            auprc = _auprc_at_k(prec_arr, pos_idx, n_total_pos, None)
            n_considered = N
            p_at_k = _precision_at_k(prec_arr, N) if N > 0 else float("nan")
            r_at_k = _recall_at_k(rec_arr, n_total_pos, N) if N > 0 else 0.0
            k_out = None
        else:
            auprc = _auprc_at_k(prec_arr, pos_idx, n_total_pos, eff_k)
            n_considered = eff_k
            p_at_k = _precision_at_k(prec_arr, eff_k) if eff_k > 0 else float("nan")
            r_at_k = _recall_at_k(rec_arr, n_total_pos, eff_k) if eff_k > 0 else 0.0
            k_out = eff_k

        metrics_rows.append({
            "k": k_out,
            "n_considered": int(n_considered),
            "n_gold_pos": int(n_total_pos),
            "auprc": float(auprc),
            "precision_at_k": float(p_at_k),
            "recall_at_k": float(r_at_k),
        })

    # Curves arrays for the chosen K*
    curve_P = prec_arr[:K_star].copy() if K_star > 0 else np.zeros(0, dtype=np.float64)
    curve_R = rec_arr[:K_star].copy() if K_star > 0 else np.zeros(0, dtype=np.float64)
    return metrics_rows, curve_P, curve_R, K_star
