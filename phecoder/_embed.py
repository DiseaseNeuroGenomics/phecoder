from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np


def _build_st_model(model_name: str, device: str = "cpu"):
    # Let SentenceTransformer manages local cache / downloads.
    return SentenceTransformer(model_name, device=device)


def _encode_texts(model, texts, encode_kwargs: dict) -> np.ndarray:
    """
    Thin wrapper that forwards kwargs to SentenceTransformer.encode().
    Do not inject batch_size unless the user provided it.
    """
    emb = model.encode(texts, **encode_kwargs)
    return np.asarray(emb, dtype=np.float32)
