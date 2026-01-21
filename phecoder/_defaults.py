from typing import TypedDict, Any

class Preset(TypedDict):
    models: list[str]
    per_model_encode_kwargs: dict[str, dict[str, Any]]  # per-model overrides

MODEL_PRESETS: dict[str, Preset] = {

    # This preset generates a top-performing ensemble when used with zsum ensemble, see https://www.medrxiv.org/content/10.64898/2026.01.08.26343725v1
    "preset:best_ensemble": {
        "models": [
            "FremyCompany/BioLORD-2023",
            "infly/inf-retriever-v1",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/sentence-t5-xxl",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            "sentence-transformers/all-MiniLM-L12-v2",
            "NeuML/pubmedbert-base-embeddings",
            "Qwen/Qwen3-Embedding-8B",
            "Qwen/Qwen3-Embedding-4B",
        ],
        "per_model_encode_kwargs": {},
    },

    # this preset is for the best performing single text embedding model
    "preset:best_single": {
        "models": ["Qwen/Qwen3-Embedding-4B"],
        "per_model_encode_kwargs": {},
    },
}

# default to best model, i.e. ensemble
DEFAULT_MODEL_PRESET = 'preset:best_ensemble'

# this was the best ensemble method
DEFAULT_ENSEMBLE = 'zsum'

# no ensemble kwargs at present, this is a placeholder in case things change
DEFAULT_ENSEMBLE_KWARGS = None