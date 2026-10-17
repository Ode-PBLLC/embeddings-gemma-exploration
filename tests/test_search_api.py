"""US-014: POST /search against a tiny synthetic set with the keyword fake embedder."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from _interface_data import tiny_dataset
from gallery.app import create_app, load_searcher
from gallery.search import KeywordEmbedder, RetrievalConfig, Searcher


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return tiny_dataset(tmp_path_factory.mktemp("search"))


def make_client(data, embed=None, **cfg_over):
    cfg = json.loads((data / "retrieval.json").read_text()) | cfg_over
    path = data / f"retrieval_{abs(hash(json.dumps(cfg, sort_keys=True)))}.json"
    path.write_text(json.dumps(cfg))
    embed = embed or KeywordEmbedder(data / "fake_topics.npz").embed
    s = load_searcher(data, data / "web", path, embed)
    return TestClient(create_app(searcher=s, assets=data / "web"), raise_server_exceptions=False), s


@pytest.fixture(scope="module")
def client(data):
    return make_client(data)


def test_contract(client, data):
    c, s = client
    r = c.post("/search", json={"query": "a storm at sea"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"query", "position", "neighbors", "weak", "mode"}
    k = s.config.k
    assert len(body["neighbors"]) == k
    assert [n["rank"] for n in body["neighbors"]] == list(range(1, k + 1))
    sims = [n["similarity"] for n in body["neighbors"]]
    assert sims == sorted(sims, reverse=True)
    items = json.loads((data / "web" / "data" / "items.json").read_text())
    for n in body["neighbors"]:
        assert items[n["index"]]["id"] == n["id"]  # index is the asset index
    assert len(body["position"]) == 3 and body["mode"] == "raw"


def test_neighbors_are_true_top_k(client, data):
    c, s = client
    body = c.post("/search", json={"query": "rings of saturn"}).json()
    q = KeywordEmbedder(data / "fake_topics.npz").embed("rings of saturn")
    sims = s.vectors @ (q / np.linalg.norm(q))
    expect = set(np.argsort(-sims)[: s.config.k].tolist())
    assert {n["index"] for n in body["neighbors"]} == expect
    np.testing.assert_allclose([n["similarity"] for n in body["neighbors"]], np.sort(sims)[::-1][: s.config.k], atol=1e-5)


def test_position_is_weighted_centroid(client):
    c, s = client
    body = c.post("/search", json={"query": "the surface of mars"}).json()
    idx = [n["index"] for n in body["neighbors"]]
    w = np.maximum([n["similarity"] for n in body["neighbors"]], 0)
    expect = (w[:, None] * s.layout[idx]).sum(0) / w.sum()
    np.testing.assert_allclose(body["position"], expect, atol=1e-5)


def test_position_plain_mean_when_all_weights_zero(data):
    _, s = make_client(data)
    # non-negative item vectors and an all-negative query: every similarity is < 0, so every weight is 0
    pos = np.abs(s.vectors)
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    neg = Searcher(pos, s.layout, s.sources, s.ids, s.config, lambda q: -np.ones(pos.shape[1]))
    body = neg.search("anything")
    assert all(n["similarity"] < 0 for n in body["neighbors"])
    idx = [n["index"] for n in body["neighbors"]]
    np.testing.assert_allclose(body["position"], s.layout[idx].mean(0), atol=1e-5)


def test_weak_flag(client):
    c, s = client
    on = c.post("/search", json={"query": "water lilies in a garden pond"}).json()
    off = c.post("/search", json={"query": "quarterly tax filing deadlines"}).json()
    assert on["weak"] is False and on["neighbors"][0]["similarity"] >= s.config.weak_threshold
    assert off["weak"] is True and off["neighbors"][0]["similarity"] < s.config.weak_threshold


def test_quota_mode(data):
    c, s = make_client(data, mode="quota")
    body = c.post("/search", json={"query": "the surface of mars"}).json()
    assert body["mode"] == "quota"
    srcs = [s.sources[n["index"]] for n in body["neighbors"]]
    assert {x: srcs.count(x) for x in set(srcs)} == {"aic": 4, "nasa": 4, "wiki": 4}
    sims = [n["similarity"] for n in body["neighbors"]]
    assert sims == sorted(sims, reverse=True)
    assert [n["rank"] for n in body["neighbors"]] == list(range(1, 13))


@pytest.mark.parametrize("q", ["", "   ", "\n\t", "x" * 501])
def test_bad_queries_422(client, q):
    c, _ = client
    r = c.post("/search", json={"query": q})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_500_char_query_ok(client):
    c, _ = client
    assert c.post("/search", json={"query": "moon " * 100}).status_code == 200


def test_model_error_is_500_with_detail(data):
    def broken(q):
        raise RuntimeError("CUDA out of memory")
    c, _ = make_client(data, embed=broken)
    r = c.post("/search", json={"query": "anything"})
    assert r.status_code == 500
    assert "CUDA out of memory" in r.json()["detail"]


def test_bad_vector_is_500(data):
    c, _ = make_client(data, embed=lambda q: np.full(768, np.nan))
    r = c.post("/search", json={"query": "anything"})
    assert r.status_code == 500 and "bad vector" in r.json()["detail"]


def test_config_requires_threshold(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"mode": "raw", "k": 12}))
    with pytest.raises(ValueError, match="weak_threshold"):
        RetrievalConfig.load(p)
    with pytest.raises(FileNotFoundError):
        RetrievalConfig.load(tmp_path / "missing.json")


# ---------- deterministic metadata filters ----------

@pytest.fixture(scope="module")
def fc(client):
    return client[0]


def _items(data):
    return json.loads((data / "web" / "data" / "items.json").read_text())


def test_filter_text_only_returns_only_text(fc, data):
    items = _items(data)
    r = fc.post("/search", json={"query": "galaxy", "filters": {"modality": ["text"]}}).json()
    assert r["filters"] == {"modality": ["text"]}
    assert r["neighbors"] and all(items[n["index"]]["modality"] == "text" for n in r["neighbors"])
    assert r["eligible"] == sum(it["modality"] == "text" for it in items)


def test_filter_collection_is_exact_top_k_within_subset(fc, data):
    items = _items(data)
    r_all = fc.post("/search", json={"query": "galaxy"}).json()
    r = fc.post("/search", json={"query": "galaxy", "filters": {"source": ["nasa"]}}).json()
    assert all(items[n["index"]]["source"] == "nasa" for n in r["neighbors"])
    # deterministic: the same request gives the same answer, and nasa hits from the unfiltered
    # top-k keep their similarity and order inside the filtered result
    again = fc.post("/search", json={"query": "galaxy", "filters": {"source": ["nasa"]}}).json()
    assert [n["id"] for n in r["neighbors"]] == [n["id"] for n in again["neighbors"]]
    nasa_in_all = [n["id"] for n in r_all["neighbors"] if n["id"].startswith("nasa:")]
    assert [n["id"] for n in r["neighbors"]][:len(nasa_in_all)] == nasa_in_all
    sims = [n["similarity"] for n in r["neighbors"]]
    assert sims == sorted(sims, reverse=True)


def test_filters_combine(fc, data):
    items = _items(data)
    r = fc.post("/search", json={"query": "galaxy", "filters": {"source": ["aic", "nasa"], "modality": ["image"]}}).json()
    assert all(items[n["index"]]["source"] in ("aic", "nasa") and items[n["index"]]["modality"] == "image"
               for n in r["neighbors"])


@pytest.mark.parametrize("filters, msg", [
    ({"colour": ["red"]}, "unknown filter field"),
    ({"source": ["louvre"]}, "unknown source"),
    ({"modality": "text"}, "valid list"),
    ({"source": ["wiki"], "modality": ["image"]}, "no works match"),
])
def test_bad_or_empty_filters_422(fc, filters, msg):
    r = fc.post("/search", json={"query": "galaxy", "filters": filters})
    assert r.status_code == 422 and msg in json.dumps(r.json())


def test_empty_filter_lists_mean_no_filter(fc):
    r = fc.post("/search", json={"query": "galaxy", "filters": {"source": [], "modality": []}}).json()
    assert r["filters"] == {} and len(r["neighbors"]) == 12
