import os
import pandas as pd
from phecoder.utils import load_results

# ---------------------------------------------------------------------
# 1. Compute per-(model,phecode) PPV cutoff rank
# ---------------------------------------------------------------------

metrics_path = "/sc/arion/projects/va-biobank/jamie/phecoder/results/case_studies/metrics.parquet"
pr_curves_path = "/sc/arion/projects/va-biobank/jamie/phecoder/results/case_studies/pr_curves.parquet"

metrics = pd.read_parquet(metrics_path)
pr_curves = pd.read_parquet(pr_curves_path)

threshold = 0.5
records = []

for _, row in pr_curves.iterrows():
    precisions = row["curve_precision"]
    cutoff_idx = next(
        (i for i, p in enumerate(precisions, start=1) if i > 10 and p < threshold),
        None,
    )
    records.append({
        "model": row["model"],
        "phecode": row["phecode"],
        "cutoff_rank": cutoff_idx,
    })

ppv_df = pd.DataFrame(records)
ppv_df["model"] = ppv_df["model"].str.split(":").str[-1]


# ---------------------------------------------------------------------
# 2. Load annotation sheets (manual physician review)
# ---------------------------------------------------------------------

annotations_dict = pd.read_excel(
    "/sc/arion/projects/va-biobank/jamie/phecoder/validation/phecoder_validation_ST_final_annotated_fixed.xlsx",
    sheet_name=None,
    dtype={"icd_code": str},
)
annotations_all = pd.concat(annotations_dict.values(), ignore_index=True)
annotations_all = annotations_all.astype({
    "icd_code": "string[python]",
    "icd_string": "string[python]",
})


# ---------------------------------------------------------------------
# 3. Load phecoder results and ICD metadata
# ---------------------------------------------------------------------

results_dir = "/sc/arion/projects/va-biobank/jamie/phecoder/results/case_studies"
phecode_ground_truth = pd.read_parquet(
    "/sc/arion/projects/va-biobank/jamie/phecoder/data/processed/phecodeX/phecode_icd_pairs.parquet"
)
icd_info = pd.read_parquet(
    "/sc/arion/projects/va-biobank/jamie/phecoder/data/processed/phecodeX/icd_info.parquet"
)

models = [
    "ens:zsum",
    "FremyCompany/BioLORD-2023",
    "infly/inf-retriever-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/sentence-t5-xxl",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/all-MiniLM-L12-v2",
    "NeuML/pubmedbert-base-embeddings",
    "Qwen/Qwen3-Embedding-8B",
    "Qwen/Qwen3-Embedding-4B",
]

sim_df = load_results(dir=results_dir, phecode_ground_truth=phecode_ground_truth, models=models)
sim_df["model"] = sim_df["model"].str.split(":").str[-1]

# Merge with ICD flags and annotations
merged = (
    sim_df
    .merge(icd_info[["icd_code", "flag"]], on="icd_code", how="left")
    .merge(annotations_all, on=["phecode", "icd_code", "icd_string", "flag"], how="outer")
    .merge(ppv_df, on=["model", "phecode"], how="left")
)

# Define retrieval based on PPV cutoff
merged["retrieved"] = (
    merged["rank"]
    .fillna(float("inf"))
    .le(merged["cutoff_rank"])
    .astype(int)
)

# ---------------------------------------------------------------------
# 3b. Parent-aware non-triviality (W) from PhecodeX hierarchy
#      W := new to this phecode AND not present in any parent phecode
# ---------------------------------------------------------------------

# Load canonical PhecodeX pairs (explicit path you requested)
phecode_icd_pairs = pd.read_parquet(
    "/sc/arion/projects/va-biobank/jamie/phecoder/data/processed/phecodeX/phecode_icd_pairs.parquet"
)

phecode_icd_pairs["phecode"] = phecode_icd_pairs["phecode"].astype(str)
phecode_icd_pairs["icd_code"] = phecode_icd_pairs["icd_code"].astype(str)

# map: phecode -> set(ICD)
known_map = (
    phecode_icd_pairs
    .groupby("phecode")["icd_code"]
    .apply(set)
    .to_dict()
)

# precompute parent list per phecode
def phecode_parents(pc: str) -> list[str]:
    """
    Return all parent phecodes by trimming trailing segments.
    e.g. "MB_293.11" -> ["MB_293.1", "MB_293"]
    """
    parts = pc.split('.')
    parents = []
    for i in range(len(parts)-1, 0, -1):
        parent = '.'.join(parts[:i])
        parents.append(parent)
    return parents

# cache: phecode -> set of ICDs known in any parent
_parent_icd_cache: dict[str, set] = {}

def parent_known_icds(pc: str) -> set:
    if pc in _parent_icd_cache:
        return _parent_icd_cache[pc]
    s = set()
    for p in phecode_parents(pc):
        s |= known_map.get(p, set())
    _parent_icd_cache[pc] = s
    return s

# compute is_known_parent for each (phecode, icd) in merged
merged["phecode"] = merged["phecode"].astype(str)
merged["icd_code"] = merged["icd_code"].astype(str)

# boolean: ICD listed under any parent phecode of this phecode
merged["is_known_parent"] = merged.apply(
    lambda r: r["icd_code"] in parent_known_icds(r["phecode"]) if pd.notna(r["icd_code"]) and pd.notna(r["phecode"]) else False,
    axis=1
)
merged["is_known_parent"] = merged["is_known_parent"].astype(int)

# non-trivial new: not in this phecode's X AND not in any parent (i.e., "truly novel" to branch)
merged["is_nontrivial_new"] = (merged["is_known"].eq(0)) & (~merged["is_known_parent"])
merged["is_nontrivial_new"] = merged["is_nontrivial_new"].astype(int)

# ---------------------------------------------------------------------
# 4. Define static set descriptions
# ---------------------------------------------------------------------

SET_DESCRIPTIONS = pd.DataFrame([
    # Base retrieval / labeling
    ("X", "ICDs assigned to the phecode by PhecodeX."),
    ("R", "ICDs retrieved for this phecode (rank ≤ per-model, per-phecode cutoff)."),
    ("R2", "Retrieved ICDs adjudicated as strict matches (reviewer label = 2)."),
    ("R1", "Retrieved ICDs adjudicated as extended/related (reviewer label = 1)."),
    ("R0", "Retrieved ICDs adjudicated as not relevant (reviewer label = 0)."),

    # Unions frequently referenced
    ("R2_or_R1", "Union of reviewer-approved retrieved ICDs (strict or extended)."),
    ("R2_or_X", "Union of PhecodeX with reviewer-strict retrieved ICDs."),
    ("R2_or_R1_or_X", "Union of PhecodeX with all reviewer-approved retrieved ICDs."),

    # Intersections / complements w.r.t. X
    ("X_and_R2", "PhecodeX ICDs that were retrieved and adjudicated strict."),
    ("X_and_R1", "PhecodeX ICDs that were retrieved and adjudicated extended/related."),
    ("X_and_R2_or_R1", "PhecodeX ICDs that were retrieved and adjudicated strict or extended."),
    ("X_not_R", "PhecodeX ICDs not retrieved at the model-specific cutoff."),

    ("R2_not_X", "Reviewer-strict retrieved ICDs absent from PhecodeX (strict extension)."),
    ("R2_or_R1_not_X", "Reviewer-approved retrieved ICDs absent from PhecodeX (approved extension)."),

    ("X_and_R0", "PhecodeX ICDs retrieved but adjudicated not relevant (disagreement with PhecodeX)."),
    ("R0_not_X", "Retrieved ICDs absent from PhecodeX and adjudicated not relevant (unrelated retrieval)."),

    # Non-trivial extension (parent-aware)
    ("W", "Non-trivial novel ICDs: not in PhecodeX for this phecode and not present under any parent phecode."),
    ("R2_and_W", "Reviewer-strict retrieved ICDs that are non-trivial (not in X and not in any parent)."),
    ("R1_and_W", "Reviewer-extended retrieved ICDs that are non-trivial (not in X and not in any parent)."),
    ("R2_or_R1_and_W", "Reviewer-approved retrieved ICDs that are non-trivial (not in X and not in any parent)."),
], columns=["set_name", "description"])


# ---------------------------------------------------------------------
# 5. Helper: construct set membership and sizes per phecode
# ---------------------------------------------------------------------

def build_phecode_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build set membership and summary counts for a single phecode.

    Parameters
    ----------
    df : pd.DataFrame
        Rows for a single phecode. Must contain:
        - 'phecode', 'icd_code', 'is_known', 'retrieved', 'relevant', 'is_known_parent', 'is_nontrivial_new',
          and optionally 'flag'.

    Returns
    -------
    sets_long : pd.DataFrame
        Long-format membership table: (phecode, set_name, icd_code, flag)
    sizes_wide : pd.DataFrame
        One-row summary of counts for each set.
    desc_df : pd.DataFrame
        Static set descriptions.
    """
    phecode_val = df["phecode"].iloc[0]

    # Base masks
    X  = df["is_known"].eq(1)
    R  = df["retrieved"].eq(1)
    rel = pd.to_numeric(df["relevant"], errors="coerce")

    R2 = R & rel.eq(2)
    R1 = R & rel.eq(1)
    R0 = R & rel.eq(0)

    # Parent-aware non-trivial novelty
    W  = df["is_nontrivial_new"].fillna(False).astype(bool)

    # Derived masks (include requested unions; avoid logically empty intersections like X & W)
    mask_dict = {
        # base
        "X": X,
        "R": R,
        "R2": R2,
        "R1": R1,
        "R0": R0,

        # reviewer-approved unions
        "R2_or_R1": (R2 | R1),

        # intersections with X
        "X_and_R2": X & R2,
        "X_and_R1": X & R1,
        "X_and_R2_or_R1": X & (R2 | R1),

        # complements wrt X
        "X_not_R": X & ~R,
        "R2_not_X": R2 & ~X,
        "R2_or_R1_not_X": (R2 | R1) & ~X,

        # broader unions with X (explicitly requested)
        "R2_or_X": (R2 | X),
        "R2_or_R1_or_X": (R2 | R1 | X),

        # disagreements / unrelated
        "X_and_R0": X & R0,
        "R0_not_X": R0 & ~X,

        # non-trivial (parent-aware) extensions
        "R2_and_W": R2 & W,                  # strict & non-trivial
        "R1_and_W": R1 & W,                  # extended & non-trivial
        "R2_or_R1_and_W": (R2 | R1) & W,     # any reviewer-approved & non-trivial
    }

    # Long-format membership
    sets_long = pd.concat([
        df.loc[mask, ["icd_code", "flag"]].assign(
            phecode=phecode_val,
            set_name=set_name
        )
        for set_name, mask in mask_dict.items()
    ], ignore_index=True)

    # One-row summary (counts per set)
    sizes_wide = pd.DataFrame([{
        "phecode": phecode_val,
        **{k: int(pd.Series(mask).sum()) for k, mask in mask_dict.items()}
    }])

    return sets_long, sizes_wide, SET_DESCRIPTIONS.copy()


# ---------------------------------------------------------------------
# 6. Build sets across all models and phecodes
# ---------------------------------------------------------------------

all_sets, all_sizes = [], []

for model, df_m in merged.groupby("model"):
    for phecode, df_p in df_m.groupby("phecode"):
        sets_p, sizes_p, desc_df = build_phecode_sets(df_p)
        sets_p["model"] = sizes_p["model"] = model
        all_sets.append(sets_p)
        all_sizes.append(sizes_p)

all_sets = pd.concat(all_sets, ignore_index=True)
all_sizes = pd.concat(all_sizes, ignore_index=True)

# Final formatting
all_sets["flag"] = all_sets["flag"].astype("Int64")
all_sets = all_sets[["model", "phecode", "set_name", "icd_code", "flag"]]
all_sizes = all_sizes[["model", "phecode"] + sorted(
    [c for c in all_sizes.columns if c not in ("model", "phecode")]
)]

# ---------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------

SAVE_DIR = 'validation/sets'

os.makedirs(SAVE_DIR, exist_ok=True)
all_sets.to_csv(SAVE_DIR + "/sets_tidy.csv", index=False)
all_sizes.to_csv(SAVE_DIR + "/set_sizes.csv", index=False)
desc_df.to_csv(SAVE_DIR + "/set_descriptions.csv", index=False)
merged.to_csv(SAVE_DIR + "/merged_phecoder_output.csv", index=False)