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

# Encoding options
st_encode_kwargs = {
    "normalize_embeddings": True,
    "trust_remote_code": True,
    "show_progress_bar": True,
}

# Per-model customization
per_model_encode_kwargs = {
    "Alibaba-NLP/gte-Qwen2-7B-instruct": {
        "prompt_name": "query"
    }
}

models = [
    "FremyCompany/BioLORD-2023",  # model trained specifically on clinical sentences and biomedical concepts.
    "infly/inf-retriever-v1", # best model on MTEB leaderboard (Medical) for information retrieval
]

ensemble_methods = [
    ("rrf",        {"k": 60},   "ens:rrf60"),
    ("mean_rank",  {},          "ens:meanrank"),
]

phecode_examples = [
    'MB_293',
    'MB_287.1',
    'MB_286.1',
    'MB_280.2',
    'MB_280.11',
    'MB_296'
]

# Load your data
icd_df = pd.read_parquet("icd_info.parquet")

# Initialize
pc = Phecoder(
    icd_df=icd_df,
    phecodes=phecode_examples,
    models=models,
    output_dir="results/",
    st_encode_kwargs=st_encode_kwargs,
    per_model_encode_kwargs=per_model_encode_kwargs,
)

# Run semantic search
pc.run()

# build ensembles
for method, kwargs, name in ensemble_methods:
    pc.build_ensemble(
        method=method,
        method_kwargs=kwargs,
        name=name
    )
```

# Data Format
ICD DataFrame (icd_df):

icd: ICD code (e.g., "E11.9", "250.00")
icd_string: Description text
flag: ICD version (9 or 10)


