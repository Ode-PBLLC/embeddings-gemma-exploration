"""The static site's in-browser search (web/js/static_search.js) answers exactly like gallery.search.Searcher.

Both are run on one random index with the same query vectors and filters (raw and quota mode, every
filter field, the error cases); node runs the JS side. Skipped when node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from gallery.search import QueryError, RetrievalConfig, Searcher
from gallery.schema import ROOT

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

N, DIM = 300, 768
RUNNER = """
import { makeIndex, rank } from %s;
import { readFileSync } from 'node:fs';
const c = JSON.parse(readFileSync(0, 'utf8'));
const ix = makeIndex(c.items, Float32Array.from(c.xyz), Float32Array.from(c.vectors), c.config);
const out = c.cases.map(({ query, qv, filters }) => {
  try { return rank(ix, query, Float32Array.from(qv), filters); }
  catch (e) { return { error: e.constructor.name, message: e.message }; }
});
process.stdout.write(JSON.stringify(out));
"""

CASES = [
    ("light on water", None),
    ("  padded query  ", {}),
    ("images only", {"modality": ["image"]}),
    ("two sources", {"source": ["wiki", "aic"]}),
    ("video or sound from nasa", {"modality": ["audio", "video"], "source": ["nasa"]}),
    ("one artist", {"artist": ["Artist 3"]}),
    ("two artists", {"artist": ["Artist 1", "Artist 2", "Artist 1"]}),
    ("a span of years", {"year": [1800, 1950]}),
    ("bce", {"year": [-500, 0]}),
    ("everything at once", {"modality": ["image"], "source": ["aic"], "year": [1700, 2000]}),
    ("empty lists are no filter", {"modality": [], "source": []}),
    ("unknown artist", {"artist": ["Nobody"]}),
    ("nothing eligible", {"year": [3000, 3100]}),
    ("bad year order", {"year": [1950, 1800]}),
    ("bad modality", {"modality": ["smell"]}),
    ("", None),
]


def fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(N, DIM)).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    xyz = rng.normal(size=(N, 3)).astype(np.float32)
    src = rng.choice(["aic", "nasa", "wiki"], N)
    mod = np.where(src == "wiki", "text", rng.choice(["image", "video", "audio"], N))
    art = np.array([f"Artist {i % 5}" if s == "aic" else "" for i, s in enumerate(src)], dtype=object)
    years = []
    for s in src:
        if s == "wiki":
            years.append(None)
        else:
            y = int(rng.integers(-800, 2020))
            years.append([y, y + int(rng.integers(0, 40))])
    items = [{"id": f"x:{i}", "source": str(src[i]), "modality": str(mod[i]), "artist": art[i], "years": years[i]}
             for i in range(N)]
    # queries near an item so similarities spread over a useful range, plus one far from everything
    qs = [vec[i] + rng.normal(scale=s, size=DIM).astype(np.float32) for i, s in ((3, .6), (40, .9), (77, 1.3))]
    qs.append(rng.normal(size=DIM).astype(np.float32))
    return items, vec, xyz, qs


def python_side(items, vec, xyz, cfg, qv, query, filters):
    yrs = np.array([it["years"] or [np.nan, np.nan] for it in items], dtype=float)
    s = Searcher(vectors=vec, layout=xyz, sources=np.array([it["source"] for it in items]),
                 ids=[it["id"] for it in items], config=cfg, embed=lambda _: qv,
                 modalities=np.array([it["modality"] for it in items]),
                 artists=np.array([it["artist"] for it in items], dtype=object), years=yrs)
    try:
        return s.search(query, filters)
    except (QueryError, ValueError) as e:
        return {"error": type(e).__name__, "message": str(e)}


@pytest.mark.parametrize("mode,threshold", [("raw", 0.3), ("quota", 0.3), ("raw", -1.0)])
def test_js_rank_matches_python_searcher(mode, threshold, tmp_path):
    items, vec, xyz, qs = fixture()
    cfg = RetrievalConfig(mode=mode, k=12, weak_threshold=threshold)
    cases = [(q, f, qv) for qv in qs for q, f in CASES]
    runner = tmp_path / "run.mjs"
    runner.write_text(RUNNER % json.dumps((ROOT / "web" / "js" / "static_search.js").as_uri()))
    payload = {"items": items, "xyz": xyz.ravel().tolist(), "vectors": vec.ravel().tolist(),
               "config": {"mode": mode, "k": 12, "weak_threshold": threshold},
               "cases": [{"query": q, "qv": qv.tolist(), "filters": f} for q, f, qv in cases]}
    js = json.loads(subprocess.run(["node", str(runner)], input=json.dumps(payload), capture_output=True,
                                   text=True, check=True).stdout)
    assert len(js) == len(cases)
    n_errors = 0
    for (q, f, qv), got in zip(cases, js):
        want = python_side(items, vec, xyz, cfg, qv, q, f)
        if "error" in want:
            n_errors += 1
            assert got.get("error") == "QueryError", (q, f, want, got)
            continue
        assert "error" not in got, (q, f, got)
        assert [n["id"] for n in got["neighbors"]] == [n["id"] for n in want["neighbors"]], (q, f)
        assert [n["rank"] for n in got["neighbors"]] == [n["rank"] for n in want["neighbors"]]
        np.testing.assert_allclose([n["similarity"] for n in got["neighbors"]],
                                   [n["similarity"] for n in want["neighbors"]], atol=1e-5)
        np.testing.assert_allclose(got["position"], want["position"], atol=1e-5)
        assert got["weak"] == want["weak"], (q, f)
        assert got["eligible"] == want["eligible"], (q, f)
        assert got["query"] == want["query"] and got["mode"] == want["mode"]
        assert got["filters"] == {k: (list(map(float, v)) if k == "year" else v) for k, v in want["filters"].items()}
    assert n_errors == 5 * len(qs)   # unknown artist, nothing eligible, bad year order, bad modality, empty
