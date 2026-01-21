# Phecoder: semantic retrieval for auditing and expanding ICD-based phenotypes in EHR biobanks

## Overview

Phecoder maps clinical phenotypes (Phecodes) to diagnosis (ICD) codes using pretrained text embedding models. It evaluates multiple embedding models and ensemble methods to find the most relevant diagnosis codes for each phenotype.

<p align="center">
  <img src="figures/fig1.png" alt="Figure description" width="600">
</p>

## Installing Phecoder
Note : python >=3.10 is required

### As a user
 ```
python -m venv venv
source venv/bin/activate
pip install phecoder
 ```

#### PyTorch with CUDA

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version **before** installing Phecoder. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details on how to install PyTorch.

### As a developer
Phecoder is developed using Poetry. Follow [Poetry - Installation](https://python-poetry.org/docs/#installation) for further details on how to install Poetry. Then,
```
git clone https://github.com/DiseaseNeuroGenomics/phecoder.git
poetry install
```

# Quick Start

## Workflow example with default settings
The default settings in Phecoder allow you to use the best ensemble as per [our study](https://www.medrxiv.org/content/10.64898/2026.01.08.26343725v1).
```python
import os
import pandas as pd
from phecoder import Phecoder

# Setup
os.environ["HF_HOME"] = "./hf-home"

# Load ICD data
icd_df = pd.read_parquet("example-data/icd_info.parquet")

# Initialize
pc = Phecoder(
    icd_df=icd_df,
    phecodes=["Suicidal ideation", "Depression", "Anxiety"],
    output_dir="./results",
    icd_cache_dir="./icd_cache"
)

# Run pipeline
pc.run()
pc.build_ensemble()

# Load results into dataframe
results = pc.load_results()
```

## A more detailed example

### 1. Setup and Import
```python
import os
import pandas as pd
from phecoder import Phecoder

# Set Hugging Face cache directory (optional but recommended)
os.environ["HF_HOME"] = "./hf-home"
```

### 2. Define Directories
```python
output_dir = "./results"              # Results saved here
icd_cache_dir = "./icd_cache"         # ICD embeddings cached here (optional, reusable across runs)
```

### 3. Load ICD Codes

Your ICD data must have columns: `icd_code` and `icd_string`
```python
icd_df = pd.read_parquet("example-data/icd_info.parquet")
```

**Example format (essential columns):**

| icd_code     | icd_string |
|:-------------|:-------------|
| E11.9        | Type 2 diabetes mellitus without complications | 
| I10          | Essential (primary) hypertension |
| J45.909      | Unspecified asthma, uncomplicated |

### 4. Define Phenotype(s)
```python
# Single phenotype
phenotype = "Eating disorders"

# OR multiple phenotypes
phenotypes = ["Eating disorders", "Type 2 diabetes", "Hypertension"]

# OR DataFrame with phecode and description
phecode_df = pd.DataFrame({
    'phecode': ['250.2', '401.1'],
    'phecode_string': ['Type 2 diabetes', 'Hypertension']
})
```

### 5. Choose Models
```python
# Light model (fast, ~80MB)
models = ["sentence-transformers/all-MiniLM-L6-v2"]

# OR clinical-trained model (better for medical text, ~440MB)
models = ["FremyCompany/BioLORD-2023"]

# OR multiple models (for ensemble)
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "FremyCompany/BioLORD-2023",
    "NeuML/pubmedbert-base-embeddings"
]
# OR use a preset
models = "preset:best_single"  # best single model
models = "preset:best_ensemble"  # best set of models for ensemble (same as default)
```

### 6. Initialize Phecoder
```python
pc = Phecoder(
    icd_df=icd_df,
    phecodes=phenotype,                  # string, list of strings, or dataframe with "phecode" column
    models=models,
    output_dir=output_dir,
    icd_cache_dir=icd_cache_dir,         # Optional: cache ICD embeddings for reuse
    st_search_kwargs={
    "top_k": 100,
    }      # Return top 100 ICD codes per phenotype
)
```

### 7. Run Pipeline
```python
# Option 1: Run directly (models auto-download if needed)
pc.run()

# Option 2: Pre-download models, then run (useful for batch jobs)
pc.download_models()  # Optional: explicitly download models first
pc.run()

# Build ensemble (combines multiple models using reciprocal rank fusion)
pc.build_ensemble(
    method="rrf",
    method_kwargs={"k": 60},
    name="ens:rrf60"
)
```

### 8. Load Results
```python
# Load all results (individual models + ensemble)
results = pc.load_results(include_ensembles=True)

# Load ensemble results only
ensemble_results = pc.load_results(
    models=['ens:rrf60'],
    include_ensembles=True
)
```

---

---

## Tips

- **First run is slower** - Models download and embeddings are computed
- **Subsequent runs are fast** - ICD embeddings are cached and reused
- **Use `icd_cache_dir`** to share ICD embeddings across multiple projects
- **Start with light models** for testing, then use clinical models for production
- **Ensembles typically outperform individual models**
- **Pre-download models** with `pc.download_models()` for batch jobs to separate download time from computation

---
## See also
For more information on how the ICD file was created, see the [ICD Data Preparation](./ICDDataPreparationREADME.md).


**For best results, use the actual ICD codes and descriptions from your biobank/EHR dataset.** 

The semantic matching works best when it operates on the same code descriptions that exist in your data. If your EHR uses specific phrasings or truncated descriptions, provide those exact strings rather than standard reference descriptions. This ensures the ranked results directly correspond to codes available in your dataset.



__Support__: If you have any questions, feel free to post your question as a GitHub Issue here or send an email to jamie.bennett@mssm.edu.

## Citations 
If you use Phecoder in research, please cite our preprint on medRxiv:

> **Phecoder: semantic retrieval for auditing and expanding ICD-based phenotypes in EHR biobanks.** Jamie J. R. Bennett, Simone Tomasi, Sonali Gupta, VA Million Veteran Program, Georgios Voloudakis, Panos Roussos, David Burstein (2026). doi: [https://doi.org/10.64898/2026.01.08.26343725](https://www.medrxiv.org/content/10.64898/2026.01.08.26343725v1).
