"""Unified item schema shared by every pipeline stage."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

COLUMNS = [
    "id",          # "<source>:<native id>", unique
    "source",      # aic | nasa | wiki | moon
    "modality",    # image | text | video | audio
    "title",
    "raw_text",    # description (image items) or paragraph (wiki)
    "media_path",  # image: <=768px JPEG fed to the models; video: the short silent MP4 loop the UI plays; "" for text
    "thumb_path",  # image items: repo-relative path to <=320px JPEG for the UI; "" for text
    "source_url",
    "license",
    "color_hex",   # "#rrggbb", lowercase
    "topic",       # department / keyword / article title
    "creator",     # artist / center / "" ; free text for display
    "date",        # display date string, may be ""
]

SOURCES = {"aic", "nasa", "wiki", "moon"}   # moon: the separate data_moon/ dataset
MODALITIES = {"image", "text", "video", "audio"}
WIKI_COLOR = "#9a958c"  # warm grey for text cards in the cloud
_HEX = re.compile(r"^#[0-9a-f]{6}$")


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame, check_files: bool = True) -> pd.DataFrame:
    """Raise SchemaError on the first broken invariant; return df unchanged otherwise."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if df["id"].duplicated().any():
        raise SchemaError(f"duplicate ids: {df.loc[df['id'].duplicated(), 'id'].head().tolist()}")
    bad = ~df["source"].isin(SOURCES)
    if bad.any():
        raise SchemaError(f"bad source values: {df.loc[bad, 'source'].unique().tolist()}")
    bad = ~df["modality"].isin(MODALITIES)
    if bad.any():
        raise SchemaError(f"bad modality values: {df.loc[bad, 'modality'].unique().tolist()}")
    bad = ~df["color_hex"].astype(str).str.match(_HEX)
    if bad.any():
        raise SchemaError(f"bad color_hex: {df.loc[bad, 'color_hex'].head().tolist()}")
    for col in ("title", "source_url", "raw_text"):
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty.any():
            raise SchemaError(f"empty {col} for ids {df.loc[empty, 'id'].head().tolist()}")
    if check_files:
        img = df[df["modality"].isin(["image", "video"])]
        for col in ("media_path", "thumb_path"):
            gone = [i for i, p in zip(img["id"], img[col]) if not p or not (ROOT / p).is_file()]
            if gone:
                raise SchemaError(f"{col} missing on disk for {len(gone)} items, e.g. {gone[:3]}")
        snd = df[df["modality"] == "audio"]   # sounds: a playable file, no thumbnail
        gone = [i for i, p in zip(snd["id"], snd["media_path"]) if not p or not (ROOT / p).is_file()]
        if gone:
            raise SchemaError(f"media_path missing on disk for {len(gone)} audio items, e.g. {gone[:3]}")
    return df
