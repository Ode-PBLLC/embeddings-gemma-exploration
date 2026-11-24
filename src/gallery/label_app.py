"""The labelling page: the gallery cloud as a tool for building a label tree.

    uvicorn gallery.label_app:create_app --factory --port 8010     (see scripts/serve_label.sh)

Serves web_label/ at /label/ on top of the gallery's own front-end files (web/: the Cloud, fonts,
three.js, the bundle), which it only reads. The numeric steps are in gallery.labeling. EmbeddingGemma
is loaded in the background (it takes minutes from a cold disk) and is needed only for "split by words".
Labels are saved to GALLERY_LABELS_DIR (default: <data dir>/labels).

    GET  /label/api/health    {items, model: loading|ready|failed, error}
    GET  /label/api/state     the saved labelling state (data/labels/labels.json), or an empty one
    PUT  /label/api/state     save it (also writes label_rows.csv)
    POST /label/api/check     {seed, pool}            -> the groups of a selection that sit together
    POST /label/api/tree      {members}               -> Ward merge tree
    POST /label/api/names     {groups}                -> words and central works per group
    POST /label/api/split     {members, words}        -> similarity of each member to each phrase
    POST /label/api/suggest   {labels}                -> proposed labels with a neighbour-vote confidence
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gallery import labeling
from gallery.app import load_searcher
from gallery.paths import WEB, assets_dir, data_dir, retrieval_config
from gallery.schema import ROOT
from gallery.search import MAX_QUERY_CHARS

log = logging.getLogger("gallery.label_app")
WEB_LABEL = ROOT / "web_label"
MAX_WORDS = 8


class LazyEmbedder:
    """EmbeddingGemma, loaded on a background thread so the page is usable while it loads."""

    def __init__(self, factory=None):
        self.status, self.error, self.model = "loading", None, None
        self._factory = factory
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            if self._factory is None:
                from gallery.embed import Embedder
                self._factory = Embedder
            self.model = self._factory()
            self.status = "ready"
        except Exception as e:   # reported by /health and by every /split call; the traceback is in the log
            log.exception("could not load the embedding model")
            self.status, self.error = "failed", f"{type(e).__name__}: {e}"

    def queries(self, texts: list[str]) -> np.ndarray:
        if self.status == "loading":
            raise HTTPException(503, "The text model is still loading. Try again in a minute.")
        if self.status == "failed":
            raise HTTPException(500, f"The text model could not load: {self.error}")
        return np.asarray(self.model.embed_texts(texts, kind="query"), dtype=np.float32)


class CheckRequest(BaseModel):
    seed: list[int]
    pool: list[int] | None = None


class TreeRequest(BaseModel):
    members: list[int]


class NamesRequest(BaseModel):
    groups: list[list[int]]


class SplitRequest(BaseModel):
    members: list[int]
    words: list[str]


class SuggestLabel(BaseModel):
    node: str
    members: list[int]
    pool: list[int] | None = None
    rejected: list[int] = []
    group: str | None = None


class SuggestRequest(BaseModel):
    labels: list[SuggestLabel]


def create_app(embedder=None, assets: Path | None = None, data: Path | None = None,
               retrieval: Path | None = None, labels_dir: Path | None = None) -> FastAPI:
    assets = assets or assets_dir()
    data = data or data_dir()
    labels_dir = labels_dir or Path(os.environ.get("GALLERY_LABELS_DIR") or data / "labels")
    embedder = embedder or LazyEmbedder()
    # the searcher is reused only for its loading: the vectors in bundle order
    searcher = load_searcher(data, assets, retrieval or retrieval_config(), embed=lambda q: embedder.queries([q])[0])
    V = searcher.vectors
    items = json.loads((assets / "data" / "items.json").read_text())
    nb = labeling.Neighbours(V)
    dataset = {"n": len(items), "ids_sha1": hashlib.sha1("\n".join(searcher.ids).encode()).hexdigest()}
    backed_up = False
    lock = threading.Lock()
    app = FastAPI(title="Embedding Gallery: labelling")

    def guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ValueError as e:   # a bad request (empty set, index out of range, ...): say which
            raise HTTPException(422, str(e)) from e

    @app.get("/label/api/health")
    def health() -> dict:
        return {"ok": True, "items": len(items), "model": embedder.status, "error": embedder.error}

    @app.get("/label/api/state")
    def get_state() -> dict:
        return guard(labeling.load, labels_dir, items, dataset)

    @app.put("/label/api/state")
    def put_state(state: dict) -> dict:
        nonlocal backed_up
        with lock:
            f = labels_dir / "labels.json"
            if not backed_up and f.is_file():   # once per server run: keep the file the session started from
                saved = json.loads(f.read_text(encoding="utf-8")).get("saved_at", "unknown").replace(":", "")
                shutil.copy2(f, labels_dir / f"labels.before-{saved}.json")
            backed_up = True
            return guard(labeling.save, labels_dir, state, items, dataset)

    @app.post("/label/api/check")
    def check(req: CheckRequest) -> dict:
        return guard(labeling.check_selection, nb, req.seed, req.pool)

    @app.post("/label/api/tree")
    def tree(req: TreeRequest) -> dict:
        return guard(labeling.ward_tree, V, req.members)

    @app.post("/label/api/names")
    def names(req: NamesRequest) -> dict:
        for g in req.groups:
            guard(labeling._indices, g, len(items), "groups")
        return {"groups": guard(labeling.name_groups, V, items, req.groups)}

    @app.post("/label/api/split")
    def split(req: SplitRequest) -> dict:
        words = [w.strip() for w in req.words if w.strip()]
        if not 2 <= len(words) <= MAX_WORDS:
            raise HTTPException(422, f"give 2 to {MAX_WORDS} words or phrases, separated by commas")
        if any(len(w) > MAX_QUERY_CHARS for w in words):
            raise HTTPException(422, f"a phrase is longer than {MAX_QUERY_CHARS} characters")
        members = guard(labeling._indices, req.members, len(items), "members")
        Q = embedder.queries(words)
        sims = V[members] @ Q.T
        return {"members": members.tolist(), "words": words, "sims": np.round(sims, 4).tolist()}

    @app.post("/label/api/suggest")
    def suggest(req: SuggestRequest) -> dict:
        return {"suggestions": guard(labeling.suggest, nb, [lab.model_dump() for lab in req.labels])}

    app.add_api_route("/label", lambda: RedirectResponse("/label/"), include_in_schema=False)
    app.add_api_route("/", lambda: RedirectResponse("/label/"), include_in_schema=False)
    app.mount("/label", StaticFiles(directory=WEB_LABEL, html=True), name="label")
    app.mount("/data", StaticFiles(directory=assets / "data"), name="data")
    app.mount("/thumbs", StaticFiles(directory=assets / "thumbs"), name="thumbs")
    for kind in ("videos", "audio"):
        if (assets / kind).is_dir():
            app.mount(f"/{kind}", StaticFiles(directory=assets / kind), name=kind)
    if (data / "media").is_dir():
        app.mount("/media", StaticFiles(directory=data / "media"), name="media")
    for part in ("js", "css", "vendor"):   # the gallery's own front end, read-only
        app.mount(f"/{part}", StaticFiles(directory=WEB / part), name=part)
    return app
