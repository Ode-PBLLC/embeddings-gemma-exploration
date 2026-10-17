"""Embed every row of data/items.parquet with EmbeddingGemma 2.

    uv run python -m gallery.embed_corpus [--chunk 256] [--fresh]

Writes data/vectors.npy (float32, (N, 768), L2-normalized) and data/vector_ids.json, both in
items.parquet order. Image items use config/retrieval.json "image_input" ("image" default, or
"image+title"). Text items use raw_text in the model card's document format
("title: {title} | text: {raw_text}").

Resumable: each chunk is saved under data/embed_chunks/<mode>/ with its ids; a rerun reuses a
chunk only if its ids, text and media files (size, mtime) are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gallery.paths import data_dir, retrieval_config
from gallery.schema import ROOT
from gallery.ingest.nasa_audio import audio_paths
from gallery.ingest.nasa_video import frames_dir

DATA = data_dir()   # GALLERY_DATA_DIR, default data/
ITEMS = DATA / "items.parquet"
VECTORS = DATA / "vectors.npy"
VECTOR_IDS = DATA / "vector_ids.json"
RETRIEVAL_CFG = retrieval_config()
CHUNKS = DATA / "embed_chunks"

log = logging.getLogger("gallery.embed_corpus")


def image_input_mode() -> str:
    if not RETRIEVAL_CFG.exists():
        return "image"
    mode = json.loads(RETRIEVAL_CFG.read_text()).get("image_input", "image")
    if mode not in ("image", "image+title"):
        raise ValueError(f"config/retrieval.json image_input must be 'image' or 'image+title', got {mode!r}")
    return mode


def chunk_signature(df: pd.DataFrame, mode: str) -> str:
    """Hash of everything that feeds the embedding: ids, text, titles, media file size+mtime."""
    h = hashlib.sha256(mode.encode())
    for r in df[["id", "modality", "title", "raw_text", "media_path"]].itertuples(index=False):
        if r.modality == "image":
            st = (ROOT / r.media_path).stat()
        elif r.modality == "video":
            st = (frames_dir(r.id) / "f00.jpg").stat()
        elif r.modality == "audio":
            st = audio_paths(r.id)[0].stat()
        else:
            st = None
        h.update(json.dumps([r.id, r.modality, r.title, r.raw_text, r.media_path,
                             st.st_size if st else 0, st.st_mtime_ns if st else 0]).encode())
    return h.hexdigest()


def embed_chunk(emb, df: pd.DataFrame, mode: str) -> np.ndarray:
    out = np.zeros((len(df), 768), dtype=np.float32)
    is_img = (df["modality"] == "image").to_numpy()
    is_vid = (df["modality"] == "video").to_numpy()
    is_aud = (df["modality"] == "audio").to_numpy()
    if is_aud.any():
        aud = df[is_aud]
        titles = aud["title"].tolist() if mode == "image+title" else None   # same choice as for images
        out[is_aud] = emb.embed_audio([audio_paths(i)[0] for i in aud["id"]], titles=titles)
    if is_vid.any():
        vid = df[is_vid]
        titles = vid["title"].tolist() if mode == "image+title" else None   # same choice as for images
        out[is_vid] = emb.embed_videos([frames_dir(i) for i in vid["id"]], titles=titles)
    is_txt = (df["modality"] == "text").to_numpy()
    if is_img.any():
        img = df[is_img]
        paths = [ROOT / p for p in img["media_path"]]
        titles = img["title"].tolist() if mode == "image+title" else None
        out[is_img] = emb.embed_images(paths, titles=titles)
    if is_txt.any():
        txt = df[is_txt]
        out[is_txt] = emb.embed_texts(txt["raw_text"].tolist(), kind="document",
                                      titles=txt["title"].tolist())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--fresh", action="store_true", help="ignore cached chunks for this mode")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    items = pd.read_parquet(ITEMS)
    mode = image_input_mode()
    cdir = CHUNKS / mode.replace("+", "_")
    if args.fresh and cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    log.info("items=%d (image=%d text=%d) image_input=%s chunk=%d", len(items),
             int((items["modality"] == "image").sum()), int((items["modality"] == "text").sum()),
             mode, args.chunk)

    from gallery.embed import Embedder
    import torch

    t_load = time.perf_counter()
    emb = Embedder(with_audio=bool((items["modality"] == "audio").any()))
    log.info("model loaded in %.1fs", time.perf_counter() - t_load)

    parts = []
    t0 = time.perf_counter()
    n_new = 0
    for ci, start in enumerate(range(0, len(items), args.chunk)):
        df = items.iloc[start:start + args.chunk]
        ids = df["id"].tolist()
        sig = chunk_signature(df, mode)
        vf, idf = cdir / f"{ci:05d}.npy", cdir / f"{ci:05d}.ids.json"
        if vf.exists() and idf.exists() and json.loads(idf.read_text()) == {"ids": ids, "sig": sig}:
            parts.append(np.load(vf))
            continue
        tc = time.perf_counter()
        v = embed_chunk(emb, df, mode)
        np.save(vf, v)
        idf.write_text(json.dumps({"ids": ids, "sig": sig}))
        parts.append(v)
        n_new += len(df)
        dt = time.perf_counter() - tc
        log.info("chunk %d: rows %d-%d (%d img) %.1fs, %.1f items/s, done %d/%d",
                 ci, start, start + len(df) - 1, int((df["modality"] == "image").sum()), dt,
                 len(df) / dt, start + len(df), len(items))

    vecs = np.concatenate(parts).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    if vecs.shape != (len(items), 768) or not np.isfinite(vecs).all() or np.abs(norms - 1).max() > 1e-3:
        log.error("bad vectors: shape=%s finite=%s max|norm-1|=%.2e", vecs.shape,
                  np.isfinite(vecs).all(), np.abs(norms - 1).max())
        return 1
    np.save(VECTORS, vecs)
    VECTOR_IDS.write_text(json.dumps(items["id"].tolist()))
    total = time.perf_counter() - t0
    log.info("wrote %s %s; embedded %d new rows in %.1fs (%.1f rows/s); peak VRAM %.0f MiB",
             VECTORS, vecs.shape, n_new, total, n_new / total if total else 0,
             torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
