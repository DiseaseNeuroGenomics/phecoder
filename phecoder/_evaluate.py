# phecoder/_eval.py
from __future__ import annotations
from typing import Optional, Tuple, Iterable, Set, List
import numpy as np
import pandas as pd


def _normalize_k_values(k: Optional[Iterable[int] | int]) -> List[Optional[int]]:
    if k is None:
        return [None]
    if isinstance(k, int):
        return [k]
    seen, out = set(), []
    for kk in k:
        kk_norm = None if kk is None else int(kk)
        key = "None" if kk_norm is None else kk_norm
        if key not in seen:
            seen.add(key)
            out.append(kk_norm)
    return out


def _vectorized_metrics_for_model(
    sim: pd.DataFrame,
    gold_df: pd.DataFrame,
    Ks: List[Optional[int]],
    include_curves: bool,
    model_name: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Fast, single-pass evaluation for one model (no per-phecode Python loop).
    """
    if sim.empty:
        return (
            pd.DataFrame(
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
            ),
            None if include_curves else None,
        )

    sim["phecode"] = sim["phecode"].astype(str)
    gold_phe = set(gold_df["phecode"].astype(str).unique())
    sim = sim[sim["phecode"].isin(gold_phe)]
    if sim.empty:
        return (
            pd.DataFrame(
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
            ),
            None if include_curves else None,
        )

    gold = gold_df.assign(label=1)
    sim = sim.merge(gold, on=["phecode", "icd_code"], how="left")
    sim["label"] = sim["label"].fillna(0).astype("int8")
    sim = sim.sort_values(["phecode", "rank", "icd_code"], kind="mergesort")

    sim["idx"] = sim.groupby("phecode", sort=False).cumcount().astype("int32") + 1
    sim["tp_cum"] = sim.groupby("phecode", sort=False)["label"].cumsum().astype("int32")

    n_gold = gold.groupby("phecode", sort=False)["label"].sum().astype("int32")
    sim = sim.join(n_gold.rename("n_gold_pos"), on="phecode")

    sim["prec_step"] = (sim["tp_cum"] / sim["idx"]).astype("float32")
    sim["prec_hit"] = (sim["prec_step"] * sim["label"]).astype("float32")
    sim["prec_hit_cum"] = sim.groupby("phecode", sort=False)["prec_hit"].cumsum()

    max_idx = sim.groupby("phecode", sort=False)["idx"].transform("max").astype("int32")

    blocks = []
    for kk in Ks:
        if kk is None:
            take_k = max_idx
            k_out = None
        else:
            take_k = np.minimum(int(kk), max_idx).astype("int32")
            k_out = int(kk)

        at_k = sim.loc[
            sim["idx"].eq(take_k),
            ["phecode", "idx", "tp_cum", "n_gold_pos", "prec_hit_cum"],
        ].copy()
        at_k.rename(columns={"idx": "n_considered"}, inplace=True)

        denom_gold = at_k["n_gold_pos"].to_numpy()
        Pk = (at_k["tp_cum"] / at_k["n_considered"]).astype("float32")
        Rk = np.where(denom_gold > 0, at_k["tp_cum"] / denom_gold, 0.0).astype(
            "float32"
        )

        denom_ap = np.where(denom_gold == 0, 1, denom_gold)
        APk = (at_k["prec_hit_cum"].to_numpy() / denom_ap).astype("float32")

        block = pd.DataFrame(
            {
                "model": model_name,
                "phecode": at_k["phecode"].astype(str).values,
                "k": (
                    at_k["n_considered"].to_numpy()
                    if k_out is None
                    else np.full(len(at_k), k_out, dtype=np.int32)
                ),
                "n_considered": at_k["n_considered"].to_numpy().astype("int32"),
                "n_gold_pos": at_k["n_gold_pos"].to_numpy().astype("int32"),
                "AP@k": APk,
                "P@k": Pk,
                "R@k": Rk,
            }
        )
        blocks.append(block)

    metrics_df = (
        pd.concat(blocks, ignore_index=True)
        if blocks
        else pd.DataFrame(
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
    )

    curves_df = None
    if include_curves:
        ladder = sim[["phecode", "idx", "tp_cum", "n_gold_pos"]].copy()
        ladder["precision"] = (ladder["tp_cum"] / ladder["idx"]).astype("float32")
        ladder["recall"] = np.where(
            ladder["n_gold_pos"] > 0, ladder["tp_cum"] / ladder["n_gold_pos"], 0.0
        ).astype("float32")
        groups = []
        for phecode, sub in ladder.groupby("phecode", sort=False):
            groups.append(
                {
                    "phecode": phecode,
                    "curve_precision": sub["precision"].to_numpy(dtype=np.float32),
                    "curve_recall": sub["recall"].to_numpy(dtype=np.float32),
                }
            )
        curves_df = pd.DataFrame(groups)
        curves_df.insert(0, "model", model_name)

    return metrics_df, curves_df


def _warn_missing_codes(
    gold_df: pd.DataFrame,
    icd_universe: Set[str],
    phecodes: Iterable[str],
) -> Optional[pd.DataFrame]:
    phecodes = set(map(str, phecodes))
    gold_df = gold_df.copy()
    gold_df["phecode"] = gold_df["phecode"].astype(str)

    # --- ICDs missing from ICD universe ---
    by_phe = gold_df.groupby("phecode")["icd_code"].apply(
        lambda s: set(s) - icd_universe
    )

    records = []
    for phe, miss in by_phe.items():
        for icd in sorted(miss):
            records.append({"phecode": phe, "missing_icd": icd})

    missing_icds_df = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["phecode", "missing_icd"])
    )

    # --- Entire phecodes with no overlap ---
    phe_with_overlap = {
        phe
        for phe, icds in gold_df.groupby("phecode")["icd_code"]
        if len(set(icds) & icd_universe) > 0
    }
    fully_excluded_phecodes = sorted(list(phecodes - phe_with_overlap))

    # --- Print summary ---
    msgs = []
    if not missing_icds_df.empty:
        by_phe_counts = missing_icds_df.groupby("phecode").size()
        for phe, n in by_phe_counts.items():
            some = (
                missing_icds_df.loc[missing_icds_df["phecode"] == phe, "missing_icd"]
                .head(10)
                .tolist()
            )
            msgs.append(f"{phe}: {n} missing (e.g., {some}{'...' if n > 10 else ''})")
    if fully_excluded_phecodes:
        msgs.append(
            f"Fully excluded phecodes: {fully_excluded_phecodes[:10]}{'...' if len(fully_excluded_phecodes) > 10 else ''}"
        )

    if msgs:
        print(
            "[evaluate] excluded ground-truth ICDs/phecodes not in ICD dataframe:\n  "
            + "\n  ".join(msgs),
            flush=True,
        )

    # --- Combine results ---
    if missing_icds_df.empty and not fully_excluded_phecodes:
        return None

    missing_icds_df["missing_type"] = "partial"
    missing_phe_df = pd.DataFrame(
        {
            "phecode": fully_excluded_phecodes,
            "missing_icd": None,
            "missing_type": "full",
        }
    )
    out_df = pd.concat([missing_icds_df, missing_phe_df], ignore_index=True)

    return out_df
