# ─────────────────────────────────────────────────────────────────────────────
# Embed ICD and Phecode strings with multiple SentenceTransformer models
# and write per‑model Parquet outputs.
#
# Per your request, the code statements are **unchanged**; only comments
# and section headers have been added for structure and clarity.
# (Any potential issues are noted in comments but not fixed.)
# ─────────────────────────────────────────────────────────────────────────────

# ── Environment / cache setup (Hugging Face) ─────────────────────────────────
import os
HUGGINGFACE_PATH = "/sc/arion/projects/va-biobank/jamie/phenocoder/hf-home"
os.environ['HF_HOME'] = HUGGINGFACE_PATH  # set cache dir for HF before loading models

# ── Imports ──────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from pathlib import Path

# ── Pandas settings ─────────────────────────────────────────────────────────
pd.options.mode.copy_on_write = True  # safer assignment semantics

# ── Paths / configuration ───────────────────────────────────────────────────
MAIN_DIR = '/sc/arion/projects/va-biobank/jamie/phenocode-embeddings'
SAVE_DIR = '/sc/arion/projects/va-biobank/jamie/phenocode-embeddings/embeddings/phecodeX'

# ── Load input data ─────────────────────────────────────────────────────────
icd_df = pd.read_parquet(os.path.join(MAIN_DIR, 'data/processed/phecodeX/icd_info.parquet'))
phecode_df = pd.read_parquet(os.path.join(MAIN_DIR, 'data/processed/phecodeX/phecode_info.parquet'))

# ── Models to evaluate / embed with ─────────────────────────────────────────
models = [
    # "google/embeddinggemma-300m",  # best model on MTEB leaderboard for semantic text similarity from google
    "Qwen/Qwen3-Embedding-8B", # best model on MTEB leaderboard for semantic text similarity
    "Qwen/Qwen3-Embedding-4B", # 2nd best model on MTEB leaderboard for semantic text similarity
    "FremyCompany/BioLORD-2023",  # model trained specifically on clinical sentences and biomedical concepts.
    "Alibaba-NLP/gte-Qwen2-7B-instruct", # best model on MTEB leaderboard (Medical)
    "infly/inf-retriever-v1", # best model on MTEB leaderboard (Medical) for information retrieval
    "all-MiniLM-L6-v2", # original SentenceTransformer model: best across all datasets tested. "all" models trained on all data
    "sentence-t5-xxl",  # original SentenceTransformer model: best for sentence embedding task. just trained on sentences data
    "multi-qa-mpnet-base-dot-v1", # original SentenceTransformer model: best for semantic search task. just trained on sentences data
    "all-MiniLM-L12-v2", # original SentenceTransformer model: best smaller model across both tasks
    "NeuML/pubmedbert-base-embeddings",  # trained on PubMed
]

# ── Main embedding loop ─────────────────────────────────────────────────────
for model_name in models:

    # Build output subdirectory per model.
    SAVE_PATH_SUBDIR = os.path.join(SAVE_DIR, model_name.split('/')[-1]) 

    # Skip if outputs already exist.
    if not os.path.exists(SAVE_PATH_SUBDIR):

        # Work on copies to avoid mutating the original dataframes.
        icd_df_temp = icd_df.copy()
        phecode_df_temp = phecode_df.copy()

        # Instantiate model (explicitly on CUDA).
        # No device fallback is implemented here (left as‑is).
        model = SentenceTransformer(model_name, device="cuda")  

        # Compute embeddings for ICD strings.
        icd_embeddings = model.encode(icd_df["icd_string"].tolist(),
                                trust_remote_code=True,
                                show_progress_bar=True,
                                normalize_embeddings=True,
                                batch_size=512)  

        # Compute embeddings for Phecode strings.
        phecode_embeddings = model.encode(phecode_df["phecode_string"].tolist(),
                                trust_remote_code=True,
                                show_progress_bar=True,
                                normalize_embeddings=True,
                                batch_size=512)  

        icd_df_temp["icd_embedding"] = list(icd_embeddings)
        phecode_df_temp["phecode_embedding"] = list(phecode_embeddings)

        # Ensure per-model directory exists and save outputs.
        os.makedirs(SAVE_PATH_SUBDIR, exist_ok=True)

        # Save Parquet files.
        icd_df_temp.to_parquet(os.path.join(SAVE_PATH_SUBDIR, 'icd_embeddings.parquet'))
        phecode_df_temp.to_parquet(os.path.join(SAVE_PATH_SUBDIR, 'phecode_embeddings.parquet'))

# import torch

# print("PyTorch version:", torch.__version__)
# print("CUDA available :", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("Number of GPUs :", torch.cuda.device_count())
#     print("Current device :", torch.cuda.current_device())
#     print("Device name    :", torch.cuda.get_device_name(torch.cuda.current_device()))
#     print("Device capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))
# else:
#     print("⚠️ No CUDA-capable GPU detected or PyTorch not built with CUDA.")