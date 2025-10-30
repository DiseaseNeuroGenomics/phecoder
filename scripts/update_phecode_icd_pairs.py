import pandas as pd

# Load original phecode_icd_pairs file
phecode_icd_pairs = pd.read_parquet('/sc/arion/projects/va-biobank/jamie/phecoder/data/processed/phecodeX/phecode_icd_pairs.parquet')

# Load physician annotated excel