"""The Moon set: tile geometry, gazetteer naming, globe-posed layout, extra-dataset mount, globe bundle."""
import json
import math
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from _interface_data import tiny_dataset
from gallery import app as app_mod
from gallery.build_assets import build
from gallery.ingest.moon import describe, tile_bounds
from gallery.layout import align_similarity, globe_xyz, tile_centres
from gallery.search import KeywordEmbedder


def test_tile_grid_covers_mercator_world_without_gaps():
    n = 32
    for y in range(n):
        for x in range(n):
            w, e, s, nn = tile_bounds(x, y)
            assert e - w == pytest.approx(360 / n) and nn > s
            if x + 1 < n:
                assert tile_bounds(x + 1, y)[0] == pytest.approx(e)
            if y + 1 < n:
                assert tile_bounds(x, y + 1)[3] == pytest.approx(s)   # row y+1 sits directly south
    assert tile_bounds(0, 0)[3] == pytest.approx(85.0511, abs=1e-3)
    assert tile_bounds(0, n - 1)[2] == pytest.approx(-85.0511, abs=1e-3)
    assert tile_bounds(0, 0)[0] == -180 and tile_bounds(n - 1, 0)[1] == pytest.approx(180)


def _gazetteer(rows):
    df = pd.DataFrame(rows)
    df["kind"] = df["type"].str.split(",").str[0]
    for c in ("min_lon", "max_lon", "min_lat", "max_lat"):
        if c not in df:
            df[c] = 0.0
    return df.sort_values("diameter", ascending=False).reset_index(drop=True)


def test_describe_names_tile_after_largest_feature_in_it():
    # z5 tile (15, 20) spans 11.25W..0E, about 46S..37S (Tycho's tile)
    gz = _gazetteer([
        {"name": "Tycho", "type": "Crater, craters", "diameter": 85.0, "lon": -11.2, "lat": -43.3,
         "origin": "Tycho Brahe; Danish astronomer.", "link": "http://x/1"},
        {"name": "Pictet", "type": "Crater, craters", "diameter": 62.0, "lon": -7.4, "lat": -43.6,
         "origin": "Marc Auguste.", "link": "http://x/2"},
        {"name": "Far", "type": "Crater, craters", "diameter": 300.0, "lon": 120.0, "lat": 10.0,
         "origin": "", "link": "http://x/3"},
    ])
    d = describe(15, 20, gz, gz.iloc[0:0])
    assert d["title"] == "Tycho" and d["topic"] == "crater"
    assert d["source_url"] == "https://x/1"
    assert "Tycho (crater, 85 km): Tycho Brahe" in d["raw_text"] and "Pictet" in d["raw_text"]
    assert "Far" not in d["raw_text"]


def test_describe_empty_tile_names_nearest_feature():
    gz = _gazetteer([{"name": "Malapert", "type": "Crater, craters", "diameter": 69.0, "lon": 12.9, "lat": -84.9,
                      "origin": "", "link": "http://x/1"}])
    d = describe(16, 31, gz, gz.iloc[0:0])   # 0..11.25E, 85S..84S: Malapert's centre lies just east of it
    assert d["title"] == "Near Malapert" and d["topic"] == "terrain"
    assert "the nearest is Malapert" in d["raw_text"]


def test_align_matches_globe_spread_and_recovers_pose():
    rng = np.random.default_rng(0)
    d = globe_xyz(rng.uniform(-180, 180, 400), rng.uniform(-80, 80, 400))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    src = (d @ q.T) * 0.37 + np.array([2, -1, 0.5])
    out, fit = align_similarity(src, d)
    assert fit["rms"] < 1e-6
    # unrelated layout: still centred on the globe with the globe's RMS radius (no shrinking)
    out, fit = align_similarity(rng.normal(size=(400, 3)) * 5 + 3, d)
    assert np.allclose(out.mean(0), d.mean(0), atol=1e-5)
    assert np.sqrt(((out - out.mean(0)) ** 2).sum(1).mean()) == pytest.approx(
        np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean()), rel=1e-5)


def test_tile_centres_match_tile_bounds():
    globe = {"z": 5, "tiles": {"a": [15, 20], "b": [0, 0]}}
    c = tile_centres(globe, ["a", "b"])
    for (x, y), v in zip([(15, 20), (0, 0)], c):
        w, e, s, n = tile_bounds(x, y)
        lat, lon = math.degrees(math.asin(v[1])), math.degrees(math.atan2(v[0], v[2]))
        assert w < lon < e and s < lat < n


@pytest.fixture(scope="module")
def mounted(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    main = tiny_dataset(root / "data_main")
    sub = tiny_dataset(root / "data_x", seed=5)
    shutil.copytree(sub / "web", root / "web_x")
    return root, main, sub


def test_extra_dataset_mount_shares_embedder(mounted, monkeypatch):
    root, main, sub = mounted
    monkeypatch.setattr(app_mod, "ROOT", root)
    embed = KeywordEmbedder(main / "fake_topics.npz").embed
    s = app_mod.load_searcher(main, main / "web", main / "retrieval.json", embed)
    c = TestClient(app_mod.create_app(searcher=s, assets=main / "web", data=main, mounts=["x"]))
    assert c.get("/healthz").json()["ok"]
    sub_items = json.loads((root / "web_x" / "data" / "items.json").read_text())
    assert c.get("/x/healthz").json()["items"] == len(sub_items)
    r = c.post("/x/search", json={"query": "rings of saturn"})
    assert r.status_code == 200
    assert {n["id"] for n in r.json()["neighbors"]} <= {it["id"] for it in sub_items}
    assert c.get("/x/data/items.json").json() == sub_items
    redirect = c.get("/x", follow_redirects=False)
    assert redirect.status_code in (307, 302) and redirect.headers["location"] == "/x/"


def test_mount_names_rejects_paths(monkeypatch):
    monkeypatch.setenv("GALLERY_MOUNTS", "moon, ../etc")
    with pytest.raises(ValueError):
        app_mod.mount_names()


def test_bundle_with_globe_carries_tiles_and_atlas(tmp_path):
    data = tiny_dataset(tmp_path / "d")
    ids = pd.read_parquet(data / "items.parquet")["id"].tolist()
    (data / "atlas.jpg").write_bytes(b"\xff\xd8fake")
    (data / "globe.json").write_text(json.dumps(
        {"z": 5, "cell": 128, "atlas": "atlas.jpg", "tiles": {i: [k % 32, k // 32] for k, i in enumerate(ids)}}))
    meta = build(data, tmp_path / "web")
    items = json.loads((tmp_path / "web" / "data" / "items.json").read_text())
    assert meta["globe"] == {"z": 5, "cell": 128, "atlas": "data/atlas.jpg"}
    assert (tmp_path / "web" / "data" / "atlas.jpg").read_bytes() == b"\xff\xd8fake"
    assert all(it["tile"] == [ids.index(it["id"]) % 32, ids.index(it["id"]) // 32] for it in items)
