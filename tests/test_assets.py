"""US-013: the front-end bundle (points.bin, items.json, thumbs)."""
import json

import numpy as np
import pandas as pd
import pytest

from _interface_data import tiny_dataset
from gallery.build_assets import build, load_summaries
from gallery.paths import assets_dir, data_dir

REQUIRED = {"id", "source", "modality", "title", "creator", "date", "summary", "thumb", "source_url", "topic"}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return tiny_dataset(tmp_path_factory.mktemp("assets"))


def check_bundle(data_root, web_root):
    items = json.loads((web_root / "data" / "items.json").read_text())
    blob = (web_root / "data" / "points.bin").read_bytes()
    vids = json.loads((data_root / "vector_ids.json").read_text())
    layout = np.load(data_root / "layout.npy")
    n = len(items)
    assert len(blob) == n * 15
    assert (web_root / "data" / "items.json").stat().st_size <= 6 * 1024 * 1024
    # order: a subsequence of vector_ids.json, in the same order
    pos = {v: i for i, v in enumerate(vids)}
    rows = [pos[it["id"]] for it in items]
    assert rows == sorted(rows)
    xyz = np.frombuffer(blob, "<f4", count=n * 3).reshape(n, 3)
    np.testing.assert_array_equal(xyz, layout[rows])
    rgb = np.frombuffer(blob, np.uint8, offset=n * 12).reshape(n, 3)
    df = pd.read_parquet(data_root / "items.parquet").set_index("id")
    for it, c in zip(items, rgb):
        assert REQUIRED <= set(it)
        assert "#%02x%02x%02x" % tuple(c) == df.loc[it["id"], "color_hex"]
        if it["modality"] in ("image", "video"):
            assert it["thumb"].startswith("thumbs/") and (web_root / it["thumb"]).is_file()
        else:
            assert it["thumb"] == ""
        if it["modality"] == "video":   # clips also carry their silent loop
            assert it["video"].startswith("videos/") and (web_root / it["video"]).is_file()
    return items


def test_bundle(data):
    items = check_bundle(data, data / "web")
    summaries = load_summaries(data / "summaries.jsonl")
    assert {it["id"] for it in items} == set(summaries)  # only curated items
    assert all(it["summary"] == summaries[it["id"]] for it in items)
    assert not any(it.get("summary_pending") for it in items)


def test_include_pending(data, tmp_path):
    meta = build(data, tmp_path, include_pending=True)
    items = check_bundle(data, tmp_path)
    total = len(pd.read_parquet(data / "items.parquet"))
    assert len(items) == total
    pending = [it for it in items if it.get("summary_pending")]
    assert len(pending) == meta["n_pending"] > 0
    assert all(len(it["summary"]) <= 421 for it in pending)


def test_summaries_last_line_wins(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"id": "a", "summary": "old"}\n{"id": "a", "summary": "new"}\n{"id": "b", "summary": "x"}\n'
                 '{"id": "b", "summary": "  "}\n')
    assert load_summaries(p) == {"a": "new"}


def test_misaligned_vector_ids_fail(data, tmp_path):
    import shutil
    for f in ("items.parquet", "summaries.jsonl", "layout.npy"):
        shutil.copy(data / f, tmp_path / f)
    ids = json.loads((data / "vector_ids.json").read_text())
    (tmp_path / "vector_ids.json").write_text(json.dumps(ids[::-1]))
    with pytest.raises(ValueError, match="order"):
        build(tmp_path, tmp_path / "web")


@pytest.mark.skipif(not (data_dir() / "vector_ids.json").is_file() or not (assets_dir() / "data" / "items.json").is_file(),
                    reason="real bundle not built yet")
def test_real_bundle():
    check_bundle(data_dir(), assets_dir())
