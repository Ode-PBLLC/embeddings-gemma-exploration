"""FastAPI app: POST /search, GET /healthz, and the static front end.

    uvicorn gallery.app:create_app --factory --port 8000     (see scripts/serve.sh)

Env:
  GALLERY_EMBEDDER    gemma (default; loads EmbeddingGemma once at startup) | fake (synthetic keyword embedder)
  GALLERY_DATA_DIR, GALLERY_RETRIEVAL, GALLERY_ASSETS_DIR   see gallery.paths
  GALLERY_MOUNTS      comma-separated extra datasets served beside the main one, sharing its embedder:
                      "moon" serves data_moon/ + web_moon/ (+ data_moon/retrieval.json) at /moon/
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gallery.paths import WEB, assets_dir, data_dir, retrieval_config
from gallery.schema import ROOT
from gallery.search import EmbedFn, KeywordEmbedder, QueryError, RetrievalConfig, Searcher, gemma_embed_fn

log = logging.getLogger("gallery.app")


class SearchRequest(BaseModel):
    query: str
    # deterministic metadata pre-filter, e.g. {"modality": ["text"], "source": ["nasa"]}
    filters: dict[str, list[str]] | None = None


def load_searcher(data: Path, assets: Path, retrieval: Path, embed: EmbedFn) -> Searcher:
    """Restrict the vectors to the items in the asset bundle, in asset order."""
    items = json.loads((assets / "data" / "items.json").read_text())
    blob = (assets / "data" / "points.bin").read_bytes()
    n = len(items)
    if len(blob) != n * 15:
        raise ValueError(f"points.bin is {len(blob)} bytes, expected {n * 15} for {n} items")
    xyz = np.frombuffer(blob, dtype="<f4", count=n * 3).reshape(n, 3)
    vectors = np.load(data / "vectors.npy", mmap_mode="r")
    vids = json.loads((data / "vector_ids.json").read_text())
    row_of = {v: i for i, v in enumerate(vids)}
    missing = [it["id"] for it in items if it["id"] not in row_of]
    if missing:
        raise ValueError(f"{len(missing)} asset items have no vector (e.g. {missing[:3]}); rebuild assets")
    rows = np.array([row_of[it["id"]] for it in items])
    return Searcher(
        vectors=np.asarray(vectors[rows]),
        layout=xyz,
        sources=np.array([it["source"] for it in items]),
        ids=[it["id"] for it in items],
        config=RetrievalConfig.load(retrieval),
        embed=embed,
        modalities=np.array([it["modality"] for it in items]),
    )


def default_embed() -> EmbedFn:
    kind = os.environ.get("GALLERY_EMBEDDER", "gemma")
    if kind == "fake":
        return KeywordEmbedder(data_dir() / "fake_topics.npz").embed
    if kind == "gemma":
        return gemma_embed_fn()
    raise ValueError(f"GALLERY_EMBEDDER must be gemma|fake, got {kind!r}")


def mount_names() -> list[str]:
    names = [n.strip() for n in os.environ.get("GALLERY_MOUNTS", "").split(",") if n.strip()]
    for n in names:
        if not n.isidentifier():
            raise ValueError(f"GALLERY_MOUNTS entry {n!r} is not a plain name")
    return names


def create_app(searcher: Searcher | None = None, assets: Path | None = None, data: Path | None = None,
               mounts: list[str] | None = None) -> FastAPI:
    assets = assets or assets_dir()
    data = data or data_dir()
    if searcher is None:
        searcher = load_searcher(data, assets, retrieval_config(), default_embed())
    app = FastAPI(title="Embedding Gallery")
    # extra datasets: their own vectors, layout and bundle; the one loaded model is shared
    for name in mount_names() if mounts is None else mounts:
        sub_data, sub_assets = ROOT / f"data_{name}", ROOT / f"web_{name}"
        sub = load_searcher(sub_data, sub_assets, sub_data / "retrieval.json", searcher.embed)
        app.mount(f"/{name}", create_app(sub, sub_assets, sub_data, mounts=[]), name=f"set-{name}")
        # the page uses relative URLs, so it must be opened as /<name>/
        app.add_api_route(f"/{name}", lambda name=name: RedirectResponse(f"/{name}/"), include_in_schema=False)
    app.state.searcher = searcher

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "items": len(searcher.ids), "mode": searcher.config.mode, "k": searcher.config.k}

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        t0 = time.perf_counter()
        try:
            result = searcher.search(req.query, req.filters)
        except QueryError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:  # model or numeric failure: surface it, with the traceback in the log
            log.exception("search failed for %r", req.query[:80])
            raise HTTPException(status_code=500, detail=f"search failed: {type(e).__name__}: {e}") from e
        result["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    app.mount("/data", StaticFiles(directory=assets / "data"), name="data")
    thumbs = assets / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    app.mount("/thumbs", StaticFiles(directory=thumbs), name="thumbs")
    for kind in ("videos", "audio"):
        d = assets / kind
        if d.is_dir():
            app.mount(f"/{kind}", StaticFiles(directory=d), name=kind)
    # larger (<=768 px) images for the detail card; same <source>/<file> layout as thumbs
    media = data / "media"
    if media.is_dir():
        app.mount("/media", StaticFiles(directory=media), name="media")
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app
