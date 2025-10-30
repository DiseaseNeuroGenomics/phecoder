import os
HUGGINGFACE_PATH = "/sc/arion/projects/va-biobank/jamie/phecoder/hf-home"
os.environ['HF_HOME'] = HUGGINGFACE_PATH  # set cache dir for HF BEFORE Phecoder import
import pandas as pd
from phecoder import Phecoder

# directories
output_dir = "/sc/arion/projects/va-biobank/jamie/phecoder/results/phecodeX"
icd_cache_dir = "/sc/arion/projects/va-biobank/jamie/phecoder/results/icd_embeddings_cache"
hf_home = "/sc/arion/projects/va-biobank/jamie/phecoder/hf-home"

# icd and phecode data
icd_df = pd.read_parquet(
    "/sc/arion/projects/va-biobank/jamie/phecoder/data/processed/phecodeX/icd_info.parquet"
)
phecode_df = pd.read_parquet(
    "/sc/arion/projects/va-biobank/jamie/phecoder/data/processed/phecodeX/phecode_info.parquet"
)
# phecode_icd_lookup = pd.read_parquet(
#     "/sc/arion/projects/va-biobank/jamie/phecoder/data/processed/phecodeX/phecode_icd_pairs.parquet"
# )

# test code (comment for full analysis)
# icd_df = icd_df.iloc[:100, :]
# phecode_df = phecode_df[phecode_df["phecode"] == "MB_293"]

# global model/search options
st_encode_kwargs= {
        "normalize_embeddings": True,
        "trust_remote_code": True,
        "show_progress_bar": True,
}
st_search_kwargs = {
    "top_k": len(icd_df),  # don't truncate lists
    }

# per-model options
per_model_encode_kwargs = {
    "Alibaba-NLP/gte-Qwen2-7B-instruct": {
        "prompt_name": "query"
    }
}

# model for encodings
models = [
    # "google/embeddinggemma-300m",  # best model on MTEB leaderboard for semantic text similarity from google
    "FremyCompany/BioLORD-2023",  # model trained specifically on clinical sentences and biomedical concepts.
    "Alibaba-NLP/gte-Qwen2-7B-instruct", # best model on MTEB leaderboard (Medical)
    "infly/inf-retriever-v1", # best model on MTEB leaderboard (Medical) for information retrieval
    "sentence-transformers/all-MiniLM-L6-v2", # original SentenceTransformer model: best across all datasets tested. "all" models trained on all data
    "sentence-transformers/sentence-t5-xxl",  # original SentenceTransformer model: best for sentence embedding task. just trained on sentences data
    "sentence-transformers/multi-qa-mpnet-base-dot-v1", # original SentenceTransformer model: best for semantic search task. just trained on sentences data
    "sentence-transformers/all-MiniLM-L12-v2", # original SentenceTransformer model: best smaller model across both tasks
    "NeuML/pubmedbert-base-embeddings",  # trained on PubMed
    "Qwen/Qwen3-Embedding-8B", # best model on MTEB leaderboard for semantic text similarity
    "Qwen/Qwen3-Embedding-4B", # 2nd best model on MTEB leaderboard for semantic text similarity
]

# init Phecoder and do analysis
pc = Phecoder(
    icd_df=icd_df,
    phecodes=phecode_df,
    models=models,
    output_dir=output_dir,
    icd_cache_dir=icd_cache_dir,
    dtype="float16",
    st_encode_kwargs=st_encode_kwargs,
    st_search_kwargs=st_search_kwargs,
    per_model_encode_kwargs=per_model_encode_kwargs
)

pc.download_models()