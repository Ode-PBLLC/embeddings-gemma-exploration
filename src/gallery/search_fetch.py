"""Search/Fetch v1: the retrieval contract Ode agents call (ode-agent, src/ode_agent/contracts/search.py).

    POST /v1/search      {query, top_k, filters: {kind: [...]}}  -> {hits, weak}
    GET  /v1/fetch/{id}                                           -> {id, kind, title, text, media_url, ...}
    errors: non-2xx with {"error": {code, message}}

An agent reads `text`: the passage for Wikipedia items, and for images, clips and sounds the Gemma 4
label plus caption and transcript. Images also carry `media_url`, which an agent can pass to a model.
URLs are absolute, built from the request, so a browser or agent on another host can load them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from pydantic import BaseModel

from gallery.search import QueryError, Searcher

log = logging.getLogger("gallery.search_fetch")
KINDS = ("image", "text", "video", "audio")   # the gallery's modalities are already contract kinds


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    filters: dict[str, Any] = {}


class Failure(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _envelope(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status)


def _media(it: dict, data: Path) -> tuple[str | None, str | None]:
    """Path (relative to the app root) of the item's full media, and its type."""
    if it["modality"] == "video":
        return it["video"], "video/mp4"
    if it["modality"] == "audio":
        return it["audio"], "audio/mp4"
    if it["modality"] == "image" and it["thumb"]:
        larger = it["thumb"].removeprefix("thumbs/")   # data/media keeps the thumbs' <source>/<file> layout
        if (data / "media" / larger).is_file():
            return f"media/{larger}", "image/jpeg"
        return it["thumb"], "image/jpeg"
    return None, None


def _locator(it: dict) -> dict:
    loc: dict[str, Any] = {"url": it["source_url"]}
    if it["modality"] == "audio" and it.get("duration"):
        loc |= {"start_s": 0, "end_s": it["duration"]}
    if it["source"] == "wiki" and (para := it["id"].rsplit(":", 1)[-1]).isdigit():   # wiki:<page>:<paragraph>
        loc["paragraph"] = int(para)
    return loc


def _text(it: dict, raw_text: str) -> str:
    if it["modality"] == "text":
        return raw_text or it["summary"]
    lines = [" ".join(x for x in (it["title"] + ".", it.get("creator", ""), it.get("date", "")) if x)]
    if it.get("caption"):
        lines.append(f"Caption: {it['caption']}")
    lines.append(f"Description (written by Gemma 4 from the {it['modality']}): {it['summary']}")
    if it.get("words"):
        lines.append("Transcript: " + " ".join(w[0] for w in it["words"]))
    return "\n".join(lines)


def router(searcher: Searcher, assets: Path, data: Path) -> APIRouter:
    items = {it["id"]: it for it in json.loads((assets / "data" / "items.json").read_text())}
    texts = pd.read_parquet(data / "items.parquet", columns=["id", "raw_text"])
    raw = {i: t for i, t in zip(texts["id"], texts["raw_text"]) if i in items}
    r = APIRouter()

    def absolute(request: Request, path: str | None) -> str | None:
        # base_url ends in the mount, ".../v1/"; media is served by the parent app
        return str(request.base_url).removesuffix("v1/") + path if path else None

    @r.post("/search")
    def search(req: SearchRequest, request: Request) -> dict:
        kinds = req.filters.get("kind") or []
        if extra := set(req.filters) - {"kind"}:
            raise Failure(400, "INVALID_ARGUMENT", f"unsupported filters {sorted(extra)}; only 'kind'")
        if bad := sorted(set(kinds) - set(KINDS)):
            raise Failure(400, "INVALID_ARGUMENT", f"unknown kinds {bad}; this collection has {list(KINDS)}")
        if not 1 <= req.top_k <= searcher.config.k:
            raise Failure(400, "INVALID_ARGUMENT", f"top_k must be 1..{searcher.config.k}")
        try:
            out = searcher.search(req.query, {"modality": kinds} if kinds else None)
        except QueryError as e:
            raise Failure(400, "INVALID_ARGUMENT", str(e)) from e
        hits = []
        for n in out["neighbors"][: req.top_k]:
            it = items[n["id"]]
            media_url, _ = _media(it, data)
            hits.append({
                "id": it["id"], "kind": it["modality"], "title": it["title"], "snippet": it["summary"][:300],
                "score": round(n["similarity"], 4), "preview_url": absolute(request, it["thumb"]),
                "media_url": absolute(request, media_url), "locator": _locator(it),
                "metadata": {k: it[k] for k in ("source", "creator", "date") if it.get(k)},
            })
        return {"hits": hits, "weak": out["weak"]}

    @r.get("/fetch/{item_id:path}")
    def fetch(item_id: str, request: Request) -> dict:
        it = items.get(item_id)
        if it is None:
            raise Failure(404, "NOT_FOUND", f"no item {item_id!r}")
        media_url, media_type = _media(it, data)
        doc = {"id": it["id"], "kind": it["modality"], "title": it["title"], "text": _text(it, raw.get(item_id, "")),
               "media_url": absolute(request, media_url), "media_type": media_type, "locator": _locator(it)}
        return {k: v for k, v in doc.items() if v is not None}

    return r


def install(app, searcher: Searcher, assets: Path, data: Path) -> None:
    """Serve the contract at /v1 on `app`, as a sub-app so every /v1 error, including unknown paths and
    methods, leaves in the contract's envelope. The gallery's other routes are untouched."""
    v1 = FastAPI(title="Search/Fetch v1")
    v1.include_router(router(searcher, assets, data))

    @v1.exception_handler(Failure)
    async def failure(request: Request, e: Failure):
        return _envelope(e.status, e.code, e.message)

    @v1.exception_handler(RequestValidationError)
    async def invalid(request: Request, e: RequestValidationError):
        return _envelope(400, "INVALID_ARGUMENT", "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}"
                                                              for x in e.errors()))

    @v1.exception_handler(HTTPException)
    async def http_error(request: Request, e: HTTPException):
        code = {403: "PERMISSION_DENIED", 404: "NOT_FOUND", 429: "RATE_LIMITED"}.get(e.status_code, "INVALID_ARGUMENT")
        return _envelope(e.status_code, code, str(e.detail))

    @v1.exception_handler(Exception)
    async def unexpected(request: Request, e: Exception):
        log.exception("%s %s failed", request.method, request.url.path)
        return _envelope(500, "INTERNAL", "the search service failed; see its log")

    app.mount("/v1", v1)
