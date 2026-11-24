"""The labelling numerics (gallery.labeling) and the labelling API, on small invented vectors: three
tight clusters in 16 dimensions, far from each other."""
import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gallery import labeling
from gallery.label_app import create_app

DIM = 16


def make_vectors(sizes=(30, 20, 10), spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.eye(DIM)[: len(sizes)]
    rows = [c + rng.normal(0, spread, (m, DIM)) for c, m in zip(centres, sizes)]
    V = np.vstack(rows).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    return V, truth


@pytest.fixture(scope="module")
def world():
    V, truth = make_vectors()
    return V, truth, labeling.Neighbours(V)


def test_neighbours_are_nearest_first_and_exclude_self(world):
    V, truth, nb = world
    assert nb.idx.shape == (60, labeling.K)
    assert not (nb.idx == np.arange(60)[:, None]).any()
    assert (np.diff(nb.sim, axis=1) <= 1e-6).all()
    # cluster 0 has 30 items, so all 15 neighbours of each come from it
    assert (truth[nb.idx[:30]] == 0).all()


def test_check_selection_separates_what_a_circle_caught_by_accident(world):
    V, truth, nb = world
    seed = list(range(0, 25)) + list(range(30, 38)) + [55]     # 25 of cluster 0, 8 of cluster 1, 1 of cluster 2
    res = labeling.check_selection(nb, seed)
    assert [sorted(g) for g in res["groups"]] == [list(range(0, 25)), list(range(30, 38))]
    assert res["strays"] == [55]


def test_check_selection_inside_a_pool_ignores_items_outside_it(world):
    V, truth, nb = world
    res = labeling.check_selection(nb, [0, 1, 2, 40], pool=range(0, 30))
    assert sorted(i for g in res["groups"] for i in g) + res["strays"] == [0, 1, 2]


@pytest.mark.parametrize("seed, pool, message", [([], None, "empty"), ([999], None, "out of range"), ([40], range(0, 30), "belongs")])
def test_check_selection_rejects_bad_input(world, seed, pool, message):
    with pytest.raises(ValueError, match=message):
        labeling.check_selection(world[2], seed, pool)


def test_ward_tree_cut_recovers_the_clusters_largest_first(world):
    V, truth, nb = world
    t = labeling.ward_tree(V, range(60))
    assert len(t["merges"]) == 59 and t["heights"] == sorted(t["heights"])
    lab = labeling.cut_tree(t["merges"], 60, 3)
    assert (lab == truth).all()            # sizes 30 > 20 > 10, so group numbers match the truth
    assert len(set(labeling.cut_tree(t["merges"], 60, 1))) == 1
    assert len(set(labeling.cut_tree(t["merges"], 60, 60))) == 60


def test_ward_tree_of_a_subset_keeps_member_order(world):
    V, truth, nb = world
    t = labeling.ward_tree(V, [59, 31, 30, 58])
    assert t["members"] == [30, 31, 58, 59]
    lab = labeling.cut_tree(t["merges"], 4, 2)
    assert lab[0] == lab[1] != lab[2] == lab[3]


def test_name_groups_picks_the_words_that_set_a_group_apart(world):
    V, truth, nb = world
    items = ([{"title": f"Aureus (Coin) Portraying Emperor {i}", "summary": "A gold coin of Rome."} for i in range(30)]
             + [{"title": f"Tetradrachm (Coin) Depicting Athena {i}", "summary": "A silver coin of Athens."} for i in range(20)]
             + [{"title": f"Nebula {i}", "summary": "A cloud of gas."} for i in range(10)])
    out = labeling.name_groups(V, items, [list(range(30)), list(range(30, 50))])
    assert out[0]["words"][0] in ("aureus", "emperor") and "coin" not in out[0]["words"]   # both groups say "coin"
    assert "tetradrachm" in out[1]["words"] and "silver" in out[1]["words"]
    assert len(out[0]["examples"]) == 6 and set(out[0]["examples"]) <= set(range(30))


def test_suggest_proposes_the_rest_of_a_cluster_and_nothing_else(world):
    V, truth, nb = world
    got = labeling.suggest(nb, [{"node": "a", "members": list(range(20)), "pool": None, "rejected": [25], "group": None}])
    assert {s["item"] for s in got} == set(range(20, 30)) - {25}      # 25 was turned down before
    assert all(s["node"] == "a" and 0.2 <= s["confidence"] <= 1 and s["of"] == labeling.K for s in got)
    assert got == sorted(got, key=lambda s: -s["confidence"])


def test_suggest_siblings_compete_and_small_labels_stay_quiet(world):
    V, truth, nb = world
    labels = [{"node": "x", "members": list(range(0, 10)), "pool": None, "rejected": [], "group": "g"},
              {"node": "y", "members": list(range(10, 22)), "pool": None, "rejected": [], "group": "g"},
              {"node": "tiny", "members": [30, 31], "pool": None, "rejected": [], "group": "g"}]
    got = labeling.suggest(nb, labels)
    items = [s["item"] for s in got]
    assert len(items) == len(set(items))                 # one proposal per item among siblings
    assert {s["node"] for s in got} <= {"x", "y"}        # a 2-item label proposes nothing
    assert set(items) <= set(range(22, 30))


def test_suggest_inside_a_pool_votes_among_the_pool_only(world):
    V, truth, nb = world
    got = labeling.suggest(nb, [{"node": "sub", "members": [0, 1, 2, 3], "pool": list(range(30)), "rejected": [], "group": "p"}])
    assert got and all(s["item"] < 30 and s["of"] == labeling.pool_k(30) for s in got)


STATE = {
    "nodes": [{"id": "c", "name": "Coins", "parent": None, "color": "#b5482f", "at": "2026-11-23T10:00:00Z", "via": "circle"},
              {"id": "g", "name": "Gold", "parent": "c", "color": "#c98a2b", "at": "2026-11-23T10:05:00Z", "via": "cluster"}],
    "labels": [{"i": 0, "node": "c", "source": "person", "method": "circle", "review": "by-hand", "at": "2026-11-23T10:00:00Z"},
               {"i": 1, "node": "c", "source": "suggestion", "method": "neighbour-vote", "review": "accepted", "confidence": 0.8, "at": "2026-11-23T10:01:00Z"},
               {"i": 1, "node": "g", "source": "suggestion", "method": "cluster", "review": "accepted-in-bulk", "at": "2026-11-23T10:05:00Z"},
               {"i": 2, "node": "g", "source": "suggestion", "method": "cluster", "review": "accepted-in-bulk", "at": "2026-11-23T10:05:00Z"}],
    "pending": [{"i": 3, "node": "c", "method": "neighbour-vote", "confidence": 0.4, "at": "2026-11-23T10:02:00Z"}],
    "rejected": [{"i": 4, "node": "c", "at": "2026-11-23T10:03:00Z"}],
}
ITEMS = [{"id": f"t:{i}", "title": f"Work {i}", "summary": ""} for i in range(60)]
DATASET = {"n": 60, "ids_sha1": "abc"}


def test_export_keeps_confirmed_and_suggested_labels_apart():
    doc, flat = labeling.export(STATE, ITEMS, DATASET)
    assert [(t["path"], t["items"]) for t in doc["tree"]] == [("Coins", 3), ("Coins > Gold", 2)]   # Gold's works count as Coins
    by = {r["id"]: r for r in doc["items"]}
    assert [x["label"] for x in by["t:1"]["labels"]] == ["Coins", "Coins > Gold"]
    assert by["t:0"]["labels"][0] | {"label": "Coins"} == {"label": "Coins", "source": "person", "method": "circle",
                                                           "review": "by-hand", "confidence": None, "margin": None, "at": "2026-11-23T10:00:00Z"}
    assert by["t:3"]["labels"] == [] and by["t:3"]["suggested"][0]["review"] == "pending"
    assert doc["rejected"] == [{"id": "t:4", "label": "Coins", "at": "2026-11-23T10:03:00Z"}]
    rows = list(csv.DictReader(io.StringIO(flat)))
    assert len(rows) == 5 and {r["source"] for r in rows} == {"person", "suggestion"}
    assert [r for r in rows if r["item_id"] == "t:3"][0]["review"] == "pending"


@pytest.mark.parametrize("change, message", [
    (lambda s: s["nodes"].append(dict(s["nodes"][0])), "duplicate"),
    (lambda s: s["nodes"][0].update(parent="g"), "parent chain"),
    (lambda s: s["nodes"][1].update(name="  "), "no name"),
    (lambda s: s["labels"][0].update(node="zzz"), "unknown label"),
    (lambda s: s["labels"][0].update(i=60), "out of range"),
    (lambda s: s["labels"][0].update(source="robot"), "bad source"),
])
def test_validate_state_refuses_broken_state(change, message):
    state = json.loads(json.dumps(STATE))
    change(state)
    with pytest.raises(ValueError, match=message):
        labeling.validate_state(state, 60)


def test_save_then_load_round_trips_and_refuses_another_bundle(tmp_path):
    info = labeling.save(tmp_path, STATE, ITEMS, DATASET)
    assert info["labels"] == 4 and (tmp_path / "label_rows.csv").is_file() and not list(tmp_path.glob("*.tmp"))
    assert labeling.load(tmp_path, ITEMS, DATASET) == STATE
    with pytest.raises(ValueError, match="different bundle"):
        labeling.load(tmp_path, ITEMS, {"n": 60, "ids_sha1": "other"})
    assert labeling.load(tmp_path / "nowhere", ITEMS, DATASET)["nodes"] == []


# ---------- the API ----------

class FakeEmbedder:
    """Stands in for EmbeddingGemma: a phrase lands on the centre of the cluster it names."""
    status, error = "ready", None

    def queries(self, texts):
        return np.eye(DIM, dtype=np.float32)[[{"zero": 0, "one": 1, "two": 2}[t] for t in texts]]


@pytest.fixture()
def client(tmp_path):
    V, _ = make_vectors()
    assets, data = tmp_path / "web", tmp_path / "data"
    (assets / "data").mkdir(parents=True)
    (assets / "thumbs").mkdir()
    data.mkdir()
    items = [{"id": f"t:{i}", "source": "aic", "modality": "image", "title": f"Work {i}", "summary": "x"} for i in range(60)]
    (assets / "data" / "items.json").write_text(json.dumps(items))
    (assets / "data" / "points.bin").write_bytes(np.zeros((60, 3), "<f4").tobytes() + bytes(180))
    np.save(data / "vectors.npy", V)
    (data / "vector_ids.json").write_text(json.dumps([it["id"] for it in items]))
    (data / "retrieval.json").write_text(json.dumps({"mode": "raw", "k": 12, "weak_threshold": 0.3}))
    app = create_app(embedder=FakeEmbedder(), assets=assets, data=data, retrieval=data / "retrieval.json")
    return TestClient(app), data


def test_api_page_and_gallery_files_are_served(client):
    c, _ = client
    assert "Labelling" in c.get("/label/").text
    assert c.get("/js/cloud.js").status_code == 200 and c.get("/data/items.json").status_code == 200
    assert c.get("/label/api/health").json() == {"ok": True, "items": 60, "model": "ready", "error": None}


def test_api_check_tree_names_suggest(client):
    c, _ = client
    r = c.post("/label/api/check", json={"seed": list(range(25)) + [55]}).json()
    assert len(r["groups"]) == 1 and len(r["groups"][0]) == 25 and r["strays"] == [55]
    t = c.post("/label/api/tree", json={"members": list(range(60))}).json()
    assert len(t["merges"]) == 59
    nm = c.post("/label/api/names", json={"groups": [[0, 1, 2], [30, 31]]}).json()
    assert len(nm["groups"]) == 2 and nm["groups"][1]["examples"][0] in (30, 31)
    s = c.post("/label/api/suggest", json={"labels": [{"node": "a", "members": list(range(20))}]}).json()
    assert {x["item"] for x in s["suggestions"]} == set(range(20, 30))


def test_api_split_scores_each_member_against_each_phrase(client):
    c, _ = client
    r = c.post("/label/api/split", json={"members": [0, 35, 55], "words": ["zero", " one ", "two"]}).json()
    assert r["words"] == ["zero", "one", "two"] and r["members"] == [0, 35, 55]
    assert [int(np.argmax(row)) for row in r["sims"]] == [0, 1, 2]


@pytest.mark.parametrize("path, body", [
    ("check", {"seed": []}), ("check", {"seed": [60]}), ("tree", {"members": [1]}),
    ("names", {"groups": [[], [1]]}), ("split", {"members": [0], "words": ["zero"]}),
])
def test_api_bad_requests_are_422_with_a_reason(client, path, body):
    r = client[0].post(f"/label/api/{path}", json=body)
    assert r.status_code == 422 and r.json()["detail"]


def test_api_split_says_when_the_model_is_still_loading(client, tmp_path):
    c, data = client

    class Loading(FakeEmbedder):
        status = "loading"

        def queries(self, texts):
            from gallery.label_app import LazyEmbedder
            return LazyEmbedder.queries(self, texts)

    app = create_app(embedder=Loading(), assets=data.parent / "web", data=data, retrieval=data / "retrieval.json")
    r = TestClient(app).post("/label/api/split", json={"members": [0, 1], "words": ["zero", "one"]})
    assert r.status_code == 503 and "loading" in r.json()["detail"]


def test_api_state_saves_files_and_backs_up_the_file_it_started_from(client):
    c, data = client
    assert c.get("/label/api/state").json() == {"nodes": [], "labels": [], "pending": [], "rejected": []}
    r = c.put("/label/api/state", json=STATE)
    assert r.status_code == 200 and r.json()["labels"] == 4
    assert c.get("/label/api/state").json() == STATE
    saved = json.loads((data / "labels" / "labels.json").read_text())
    assert saved["tree"][1]["path"] == "Coins > Gold" and len(saved["items"]) == 4
    assert not list((data / "labels").glob("labels.before-*.json"))      # nothing existed before this run
    bad = c.put("/label/api/state", json={**STATE, "labels": [{"i": 99, "node": "c", "source": "person", "review": "by-hand"}]})
    assert bad.status_code == 422 and "out of range" in bad.json()["detail"]
    assert c.get("/label/api/state").json() == STATE                      # a refused save changes nothing
    # a new server run over the same folder keeps the file it found before overwriting it
    app2 = create_app(embedder=FakeEmbedder(), assets=data.parent / "web", data=data, retrieval=data / "retrieval.json")
    TestClient(app2).put("/label/api/state", json={**STATE, "pending": []})
    backups = list((data / "labels").glob("labels.before-*.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text())["state"] == STATE
