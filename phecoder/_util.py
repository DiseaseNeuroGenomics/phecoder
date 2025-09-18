from __future__ import annotations
from pathlib import Path
import pandas as pd
import hashlib, datetime, re
from typing import Any, Dict, Optional

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def sanitize_model_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", s)

def df_fingerprint(df: pd.DataFrame) -> str:
    """
    Stable, order-sensitive fingerprint of the dataframe contents & columns.
    """
    h = hashlib.sha256()
    h.update(("|".join(df.columns)).encode())
    # use itertuples for predictable ordering
    for row in df.itertuples(index=False, name=None):
        h.update(("|".join("" if x is None else str(x) for x in row)).encode())
        h.update(b"\n")
    return h.hexdigest()

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def clean_kwargs(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Drop keys with None (so ST uses its internal defaults).
    """
    if not d:
        return {}
    out = dict(d)
    for k in list(out.keys()):
        if out[k] is None:
            out.pop(k)
    return out
