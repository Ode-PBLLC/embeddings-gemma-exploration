"""The static site works end to end: front page, both collections, the suggested searches (precomputed
vectors, no model) and, with RUN_MODEL_E2E=1, free-text search with the model running in the browser."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

import numpy as np
import pytest

MODEL_E2E = os.environ.get("RUN_MODEL_E2E") == "1"
MODEL_TIMEOUT_MS = 20 * 60 * 1000   # first download over a slow line


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def expected_top(base: str, query: str) -> tuple[list[str], bool]:
    """Rank a suggested search in numpy from the published files: what the page must show."""
    items = json.loads(fetch(base + "data/items.json"))
    meta = json.loads(fetch(base + "data/meta.json"))
    vec = np.frombuffer(fetch(base + "data/vectors.bin"), dtype="<f4").reshape(len(items), 768)
    demo = {d["q"]: d["v"] for d in json.loads(fetch(base + "data/demo_queries.json"))}
    qv = np.frombuffer(base64.b64decode(demo[query]), dtype="<f4")
    sims = vec @ (qv / np.linalg.norm(qv))
    cfg = meta["static"]["retrieval"]
    top = np.argsort(-sims, kind="stable")[: cfg["k"]]
    return [items[i]["id"] for i in top], bool(sims.max() < cfg["weak_threshold"])


def open_set(page, url):
    page.goto(url)
    page.wait_for_function("window.__gallery && window.__gallery.ready", timeout=60000)


def ask(page, query, timeout=30000):
    page.fill("#q", query)
    page.press("#q", "Enter")
    page.wait_for_function(
        "q => window.__gallery.phase === 'wall' && window.__gallery.lastResult.query === q", arg=query, timeout=timeout)
    return page.evaluate("window.__gallery.lastResult")


def test_front_page(page, site_url):
    page.goto(site_url)
    assert page.title() == "The Embedding Gallery"
    body = page.text_content("body")
    assert "{{" not in page.content()
    assert "5,376 works" in body and "1,024 tiles" in body
    links = page.eval_on_selector_all("a.room", "els => els.map(e => e.getAttribute('href'))")
    assert links == ["gallery/", "moon/"]
    page.wait_for_function("[...document.images].every(i => i.complete && i.naturalWidth > 0)", timeout=20000)
    page.click("a.room[href='gallery/']")
    page.wait_for_function("window.__gallery && window.__gallery.ready", timeout=60000)
    assert page.url.endswith("/gallery/")


@pytest.mark.parametrize("collection,query", [("gallery", "fire and heat"), ("moon", "a huge impact basin")])
def test_suggested_search_without_the_model(page, site_url, collection, query):
    base = site_url + collection + "/"
    open_set(page, base)
    assert page.is_visible("#model-status")
    chips = page.eval_on_selector_all(".suggest-q", "els => els.map(e => e.textContent)")
    assert query in chips, chips
    page.click(f".suggest-q:text-is('{query}')")
    page.wait_for_function("q => window.__gallery.phase === 'wall' && window.__gallery.lastResult.query === q",
                           arg=query, timeout=30000)
    got = page.evaluate("window.__gallery.lastResult")
    ids, weak = expected_top(base, query)
    assert [n["id"] for n in got["neighbors"]] == ids
    assert got["weak"] is weak is False
    assert page.evaluate("document.querySelectorAll('#wall .piece, #wall [data-index]').length") >= 1
    assert not any("/audio/aic/" in u for u in page.requests)


def test_placard_uses_thumbnail_only(page, site_url):
    open_set(page, site_url + "gallery/")
    ask(page, "light on water")
    idx = page.evaluate("window.__gallery.lastResult.neighbors.find(n => n.id.startsWith('aic:')).index")
    page.evaluate("i => window.__cloud.clickCb(i)", idx)
    page.wait_for_function("i => window.__gallery.detailIndex === i", arg=idx, timeout=5000)
    page.wait_for_timeout(500)
    assert not any("/media/" in u for u in page.requests), [u for u in page.requests if "/media/" in u]


def test_free_text_before_the_model_is_ready_says_so(page, site_url):
    open_set(page, site_url + "gallery/")
    state = page.evaluate("window.__static.status.state")
    if state == "ready":
        pytest.skip("model already loaded from this browser's cache")
    page.fill("#q", "a lighthouse at dusk")
    page.press("#q", "Enter")
    page.wait_for_selector("#ask-error:not([hidden])", timeout=10000)
    assert "still loading" in page.inner_text("#ask-error")
    assert page.evaluate("window.__gallery.phase") == "idle"


@pytest.mark.skipif(not MODEL_E2E, reason="downloads the model (hundreds of MB); set RUN_MODEL_E2E=1")
def test_free_text_search_with_the_model(page, site_url):
    open_set(page, site_url + "gallery/")
    page.wait_for_function("window.__static.status.state === 'ready' || window.__static.status.state === 'failed'",
                           timeout=MODEL_TIMEOUT_MS)
    status = page.evaluate("window.__static.status")
    assert status["state"] == "ready", status
    assert "Search model ready" in page.inner_text("#model-text")
    got = ask(page, "a painting by Winslow Homer of the sea")
    # the server's answer (reports/static_parity.md), less the AIC audio guide the public site leaves out
    server = ["wiki:313284:0", "aic:16818", "wiki:313284:1", "aic:8971", "aic:110507", "aic:16815",
              "aic:84088", "aic:68792", "aic:14309", "aic:25865", "aic:16779"]
    ids = [n["id"] for n in got["neighbors"]]
    assert ids[0] == server[0]
    assert len(set(ids) & set(server)) >= 10, ids
    assert got["weak"] is False
    weak = ask(page, "quarterly tax filing deadlines")
    assert weak["weak"] is True
    assert page.is_visible("#weak-badge")
