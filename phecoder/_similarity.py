from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from sentence_transformers import util as st_util


def _semantic_search_topk(
    query: np.ndarray,
    corpus: np.ndarray,
    device: str,
    st_search_kwargs: Dict[str, Any] | None = None,
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Use SentenceTransformers' util.semantic_search to get top-k neighbors.
    Expects ALREADY-normalized embeddings if cosine similarity is desired.
    (If you encoded with normalize_embeddings=True, they are normalized.)

    Returns:
      scores_list: list over queries of [score_i...]
      idx_list:    list over queries of [corpus_id_i...]
    """
    st_search_kwargs = dict(st_search_kwargs or {})
    # Default top_k if not provided: return all (handled by util as full ranking)
    # But typically users will set 'top_k' in st_search_kwargs.

    Q = torch.from_numpy(query).to(device)
    X = torch.from_numpy(corpus).to(device)

    hits = st_util.semantic_search(
        query_embeddings=Q, corpus_embeddings=X, **st_search_kwargs
    )

    # Convert to plain Python lists on CPU
    scores_list, idx_list = [], []
    for per_query in hits:
        idx_list.append([int(r["corpus_id"]) for r in per_query])
        scores_list.append([float(r["score"]) for r in per_query])

    return scores_list, idx_list
