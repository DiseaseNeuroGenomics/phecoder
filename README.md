# Phecoder: Semantic ICD code ranking for efficient Phecode curation.

PheCoder maps clinical phenotypes (Phecodes) to ICD-9/ICD-10 diagnosis codes using transformer-based semantic embeddings. It evaluates multiple embedding models and ensemble methods to find the most relevant diagnosis codes for each phenotype.

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.



**Installing Phecoder**

 *As a developer* <br>
 Note : python >=3.10 is required
```
git clone https://github.com/03bennej/phecoder.git
pip install pipx
pipx install poetry
pipx install poetry 
```

 *As a user*
 ```
pip install phecoder
 ```
# Quick Start

```
from phecoder import Phecoder
import pandas as pd

# Load your data
icd_df = pd.read_parquet("icd_info.parquet")
phecode_icd_lookup = pd.read_parquet("phecode_icd_pairs.parquet")

# Initialize
pc = Phecoder(
    icd_df=icd_df,
    phecodes=phecode_examples,
    models=["all-MiniLM-L6-v2", "FremyCompany/BioLORD-2023"],
    output_dir="results/",
)

# Run semantic search
pc.run()

# Build ensemble
pc.build_ensemble(method="rrf", method_kwargs={"k": 60})

# Evaluate
pc.evaluate(phecode_ground_truth=phecode_icd_lookup)
```
