"""Search/Fetch v1 (/v1/search, /v1/fetch) on the tiny synthetic set: the contract Ode agents call."""
import json
from urllib.parse import quote

import pytest

from test_search_api import make_client
from _interface_data import tiny_dataset

HIT_KEYS = {"id", "kind", "title", "snippet", "score", "locator", "metadata"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = tiny_dataset(tmp_path_factory.mktemp("v1"))
    c, _ = make_client(data)
    return c, json.loads((data / "web" / "data" / "items.json").read_text())


def test_search_returns_contract_hits(client):
    c, items = client
    r = c.post("/v1/search", json={"query": "a storm at sea", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"hits", "weak"} and len(body["hits"]) == 5
    for h in body["hits"]:
        assert HIT_KEYS <= set(h) and h["kind"] in ("image", "text", "video", "audio")
        for url in (h.get("preview_url"), h.get("media_url")):
            assert url is None or (url.startswith("http://testserver/") and "/v1/" not in url)
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)


def test_kind_filter_is_exact(client):
    c, items = client
    kinds = {it["modality"] for it in items}
    for kind in kinds:
        hits = c.post("/v1/search", json={"query": "anything", "filters": {"kind": [kind]}}).json()["hits"]
        assert hits and {h["kind"] for h in hits} == {kind}


def test_fetch_every_kind(client):
    c, items = client
    for kind in {it["modality"] for it in items}:
        it = next(i for i in items if i["modality"] == kind)
        r = c.get("/v1/fetch/" + quote(it["id"], safe=""))
        assert r.status_code == 200, r.text
        doc = r.json()
        assert doc["id"] == it["id"] and doc["kind"] == kind and doc["text"]
        assert doc["locator"]["url"] == it["source_url"]


def test_errors_use_the_envelope(client):
    c, _ = client
    r = c.get("/v1/fetch/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    for body in ({"query": "x", "filters": {"kind": ["pdf"]}}, {"query": "x", "filters": {"year": [1, 2]}},
                 {"query": "x", "top_k": 0}, {"query": "  "}, {"top_k": 3}, {"query": "x", "top_k": "many"}):
        r = c.post("/v1/search", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_ARGUMENT", body


def test_unexpected_failures_use_the_envelope(client, monkeypatch):
    c, _ = client
    searcher = c.app.state.searcher
    monkeypatch.setattr(searcher, "search", lambda *a, **k: 1 / 0)
    r = c.post("/v1/search", json={"query": "a storm at sea"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL"
    r = c.get("/v1/nothing-here")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    r = c.get("/v1/search")  # wrong method
    assert r.status_code == 405 and "error" in r.json()
