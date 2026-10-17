"""Where the interface stages read and write, overridable by env for the synthetic dataset.

GALLERY_DATA_DIR    items.parquet, summaries.jsonl, vectors.npy, vector_ids.json, layout.npy (default data/)
GALLERY_RETRIEVAL   retrieval config JSON (default config/retrieval.json)
GALLERY_ASSETS_DIR  front-end bundle root holding data/ and thumbs/ (default web/)
"""

from __future__ import annotations

import os
from pathlib import Path

from gallery.schema import ROOT

WEB = ROOT / "web"


def _resolve(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def data_dir() -> Path:
    return _resolve("GALLERY_DATA_DIR", ROOT / "data")


def retrieval_config() -> Path:
    return _resolve("GALLERY_RETRIEVAL", ROOT / "config" / "retrieval.json")


def assets_dir() -> Path:
    return _resolve("GALLERY_ASSETS_DIR", WEB)
