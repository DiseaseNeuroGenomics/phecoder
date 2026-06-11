# Usage Guide

## Setup

```python
import os
import pandas as pd
from phecoder import Phecoder, load_icd_df

# Recommended: set Hugging Face cache directory
os.environ["HF_HOME"] = "./hf-home"

output_dir = "./results"
icd_cache_dir = "./icd_cache"  # optional; ICD embeddings are reused across runs
```

## ICD data

Phecoder ships with a bundled ICD reference. Use it directly or supply your own DataFrame:

```python
# Use bundled ICD data (default)
icd_df = load_icd_df()

# Or load your own — must have these columns:
```

| icd_code | icd_string |
|:---|:---|
| E11.9 | Type 2 diabetes mellitus without complications |
| I10 | Essential (primary) hypertension |
| J45.909 | Unspecified asthma, uncomplicated |

!!! tip "Best results"
    Use the actual ICD codes and descriptions from your biobank or EHR dataset. If your EHR uses specific phrasings or truncated descriptions, provide those exact strings — the ranked results will then directly correspond to codes available in your data.

## Define phenotypes

```python
# Single phenotype
phecodes = "Eating disorders"

# Multiple phenotypes
phecodes = ["Eating disorders", "Type 2 diabetes", "Hypertension"]

# DataFrame with phecode IDs and descriptions
phecodes = pd.DataFrame({
    "phecode": ["250.2", "401.1"],
    "phecode_string": ["Type 2 diabetes", "Hypertension"],
})
```

## Choose models

```python
# Light model — fast, ~80 MB
models = ["sentence-transformers/all-MiniLM-L6-v2"]

# Clinically-trained model — better for medical text, ~440 MB
models = ["FremyCompany/BioLORD-2023"]

# Multiple models for ensemble
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "FremyCompany/BioLORD-2023",
    "NeuML/pubmedbert-base-embeddings",
]

# Presets based on our evaluation
models = "preset:best_single"    # best single model
models = "preset:best_ensemble"  # best set of models for ensemble (default)
```

## Initialize and run

```python
pc = Phecoder(
    phecodes=phecodes,
    icd_df=icd_df,                        # omit to use bundled ICD data
    models=models,
    output_dir=output_dir,
    icd_cache_dir=icd_cache_dir,
    st_search_kwargs={"top_k": 100},      # return top 100 ICD codes per phenotype
)

# Option 1: run directly (models auto-download if needed)
pc.run()

# Option 2: pre-download models first (useful for batch jobs)
pc.download_models()
pc.run()
```

## Build an ensemble

```python
# Default ensemble (reciprocal rank fusion, recommended)
pc.build_ensemble()

# Custom ensemble
pc.build_ensemble(
    method="rrf",
    method_kwargs={"k": 60},
    name="ens:rrf60",
)
```

## Load results

```python
# All results (individual models + ensembles)
results = pc.load_results()

# Specific model or ensemble
results = pc.load_results(
    models=["ens:rrf60"],
    include_ensembles=True,
)
```

## Performance tips

- **First run is slower** — models download and ICD embeddings are computed.
- **Subsequent runs are fast** — ICD embeddings are cached and reused automatically.
- **Use `icd_cache_dir`** to share ICD embeddings across multiple projects.
- **Start with a light model** for testing, then switch to clinical models for production.
- **Ensembles typically outperform individual models.**
- **Pre-download with `pc.download_models()`** before batch jobs to separate download time from computation.

## ICD data preparation

For details on how the bundled ICD file was constructed, see [ICDDataPreparationREADME.md](https://github.com/DiseaseNeuroGenomics/phecoder/blob/main/ICDDataPreparationREADME.md).
