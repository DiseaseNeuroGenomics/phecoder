import os
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True  # safer assignment semantics

# ---- Paths ------------------------------------------------------------------
MAIN_DIR = "/sc/arion/projects/va-biobank/jamie/phecoder"
PATH_OUTPAT_COUNT   = os.path.join(MAIN_DIR, "data/raw/david/outpat_icd_summary_counts.txt")
PATH_OUTPAT_COUNTPAT   = os.path.join(MAIN_DIR, "data/raw/david/outpat_icd_summary_counts_patient.txt")
PATH_INPAT_COUNT   = os.path.join(MAIN_DIR, "data/raw/david/icd_summary_inpat_counts.txt")
PATH_INPAT_COUNTPAT   = os.path.join(MAIN_DIR, "data/raw/david/icd_summary_inpat_patcounts.txt")
PATH_UNROLLED = os.path.join(MAIN_DIR, "data/raw/phecodeX/phecodeX_unrolled_ICD_CM.csv")
PATH_INFO     = os.path.join(MAIN_DIR, "data/raw/phecodeX/phecodeX_info.csv")
SAVE_DIR      = os.path.join(MAIN_DIR, "data/processed/phecodeX")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- Load data --------------------------------------------------------------
icd_outpat_count_df = pd.read_csv(PATH_OUTPAT_COUNT, sep="|", dtype={"ICDCode": "string", "CodeType": "string"})
icd_outpat_countpat_df = pd.read_csv(PATH_OUTPAT_COUNTPAT, sep="|", dtype={"ICDCode": "string", "CodeType": "string"})
icd_inpat_count_df = pd.read_csv(PATH_INPAT_COUNT, sep="|", dtype={"ICDCode": "string", "CodeType": "string"})
icd_inpat_countpat_df = pd.read_csv(PATH_INPAT_COUNTPAT, sep="|", dtype={"ICDCode": "string", "CodeType": "string"})
phecode_info_df = pd.read_csv(PATH_INFO, dtype={"ICDCode": "string", "phecode": "string"})  # master phecode metadata
phecode_icd_df = pd.read_csv(PATH_UNROLLED, dtype={"ICD": "string", "phecode": "string"})  # contains all ICD↔phecode associations

keys = ['ICDCode', 'CodeType']

# ---- Depuplicate ICD codes with multiple descriptions -----------------------
others = [c for c in icd_outpat_count_df.columns if c not in keys + ['Count']]
icd_outpat_count_df = (
    icd_outpat_count_df.groupby(keys, as_index=False, sort=False, dropna=False)
      .agg(Count=('Count','sum'), **{col: (col, 'first') for col in others})
)
others = [c for c in icd_inpat_count_df.columns if c not in keys + ['Count']]
icd_inpat_count_df = (
    icd_inpat_count_df.groupby(keys, as_index=False, sort=False, dropna=False)
      .agg(Count=('Count','sum'), **{col: (col, 'first') for col in others})
)
others = [c for c in icd_outpat_countpat_df.columns if c not in keys + ['CountPat']]
icd_outpat_countpat_df = (
    icd_outpat_countpat_df.groupby(keys, as_index=False, sort=False, dropna=False)
      .agg(CountPat=('CountPat','sum'), **{col: (col, 'first') for col in others})
)
others = [c for c in icd_inpat_countpat_df.columns if c not in keys + ['CountPat']]
icd_inpat_countpat_df = (
    icd_inpat_countpat_df.groupby(keys, as_index=False, sort=False, dropna=False)
      .agg(CountPat=('CountPat','sum'), **{col: (col, 'first') for col in others})
)

# ---- Ensure consistant columns and names ------------------------------------

# Outpatient: counts + unique-patient counts
icd_outpat_df = (
    icd_outpat_count_df
      .merge(icd_outpat_countpat_df[keys + ['CountPat']], on=keys, how='outer', validate='one_to_one')
      .rename(columns={'Count': 'outpat_count', 'CountPat': 'outpat_patient_count'})
)

# Inpatient: counts + unique-patient counts
icd_inpat_df = (
    icd_inpat_count_df
      .merge(icd_inpat_countpat_df[keys + ['CountPat']], on=keys, how='outer', validate='one_to_one')
      .rename(columns={'Count': 'inpat_count', 'CountPat': 'inpat_patient_count'})
)

# Union inpatient + outpatient
icd_info_df = icd_outpat_df.merge(icd_inpat_df, on=keys, how='outer', suffixes=('', '_right'))
# If both sides carried the same descriptive cols, prefer the left (outpatient) version:
icd_info_df['ICDDescription'] = icd_info_df['ICDDescription'].combine_first(icd_info_df['ICDDescription_right'])
icd_info_df = icd_info_df.drop(columns=[c for c in icd_info_df.columns if c.endswith('_right')])

flag_map = {"ICD9": 9, "ICD10": 10}  # keep WHO just in case
icd_info_df['flag'] = icd_info_df['CodeType'].map(flag_map)
icd_info_df = icd_info_df.drop('CodeType', axis=1)

flag_map = {"ICD9CM": 9, "ICD10CM": 10}  # keep WHO just in case
phecode_icd_df['flag'] = phecode_icd_df['vocabulary_id'].map(flag_map)
phecode_icd_df = phecode_icd_df.drop('vocabulary_id', axis=1)

icd_info_df = icd_info_df.rename(
    columns= {
    'ICDCode': 'icd_code',
    'ICDDescription': 'icd_string',
}
)
phecode_info_df = phecode_info_df.rename(
    columns= {
    'ICDCode': 'icd_code',
    'ICDDescription': 'icd_string',
}
)
phecode_icd_df = phecode_icd_df.rename(
    columns= {
    'ICD': 'icd_code',
}
)

# Strip trailing '.' from icd dataframes. This is not present in PhecodeX and will cause downstream merge issues
icd_info_df["icd_code"] = icd_info_df["icd_code"].str.rstrip(".")

# Save
icd_info_df.to_parquet(os.path.join(SAVE_DIR, "icd_info.parquet"), index=False)
phecode_info_df.to_parquet(os.path.join(SAVE_DIR, "phecode_info.parquet"), index=False)
phecode_icd_df.to_parquet(os.path.join(SAVE_DIR, "phecode_icd_pairs.parquet"), index=False)