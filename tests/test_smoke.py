"""US-001: the app starts, /healthz answers, / serves the front end."""
import pytest
from fastapi.testclient import TestClient

from _interface_data import tiny_dataset
from gallery.app import create_app, load_searcher
from gallery.search import KeywordEmbedder


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tiny_dataset(tmp_path_factory.mktemp("smoke"))
    s = load_searcher(d, d / "web", d / "retrieval.json", KeywordEmbedder(d / "fake_topics.npz").embed)
    return TestClient(create_app(searcher=s, assets=d / "web"))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "The Embedding Gallery" in r.text
    assert 'id="q"' in r.text


def test_static_bundle_served(client):
    assert client.get("/data/items.json").status_code == 200
    assert client.get("/data/points.bin").status_code == 200
    assert client.get("/vendor/three/three.module.min.js").status_code == 200
    assert client.get("/js/main.js").status_code == 200


def test_no_cdn_at_runtime():
    """Everything the page loads is local (BUILD.md decision 3, SPEC N4)."""
    from gallery.paths import WEB
    for f in [WEB / "index.html", WEB / "css" / "gallery.css", *sorted((WEB / "js").glob("*.js"))]:
        text = f.read_text()
        assert "http://" not in text and "https://" not in text, f
