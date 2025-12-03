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

models = [
    "FremyCompany/BioLORD-2023",  # model trained specifically on clinical sentences and biomedical concepts.
    "infly/inf-retriever-v1", # best model on MTEB leaderboard (Medical) for information retrieval
    "sentence-transformers/all-MiniLM-L6-v2", # original SentenceTransformer model: best across all datasets tested. "all" models trained on all data
    "sentence-transformers/sentence-t5-xxl",  # original SentenceTransformer model: best for sentence embedding task. just trained on sentences data
    "sentence-transformers/multi-qa-mpnet-base-dot-v1", # original SentenceTransformer model: best for semantic search task. just trained on sentences data
    "sentence-transformers/all-MiniLM-L12-v2", # original SentenceTransformer model: best smaller model across both tasks
    "NeuML/pubmedbert-base-embeddings",  # trained on PubMed
    "Qwen/Qwen3-Embedding-8B", # best model on MTEB leaderboard for semantic text similarity
    "Qwen/Qwen3-Embedding-4B", # 2nd best model on MTEB leaderboard for semantic text similarity
]

ensemble_methods = [
    ("rrf",        {"k": 60},   "ens:rrf60"),
    ("mean_rank",  {},          "ens:meanrank"),
    ("median_rank",{},          "ens:medianrank"),
    ("rra",        {},          "ens:rra"),
    ("zsum",       {},          "ens:zsum"),
    ("combsum",    {},          "ens:combsum"),
    ("combmnz",    {},          "ens:combmnz"),
    ("fisher",     {},          "ens:fisher"),
]

# Load your data
icd_df = pd.read_parquet("icd_info.parquet")
phecode_icd_lookup = pd.read_parquet("phecode_icd_pairs.parquet")

# Initialize
pc = Phecoder(
    icd_df=icd_df,
    phecodes=phecode_examples,
    models=models,
    output_dir="results/",
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

# evaluate
pc.evaluate(phecode_ground_truth=phecode_icd_lookup_example, include_curves=True)
```
