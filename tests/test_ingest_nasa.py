"""Offline tests for the NASA ingester: parsing, dedup, dominant colour."""

import numpy as np
import pytest
from PIL import Image

from gallery.ingest import _common as C
from gallery.ingest import nasa


def _hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _stripes(parts, size=200):
    """Image made of horizontal bands: parts = [(rgb, fraction), ...]."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    y = 0
    for rgb, frac in parts:
        h = int(round(frac * size))
        arr[y:y + h] = rgb
        y += h
    arr[y:] = parts[-1][0]
    return Image.fromarray(arr)


def test_dominant_skips_black_when_colour_cluster_is_large():
    img = _stripes([((0, 0, 0), 0.7), ((220, 40, 30), 0.3)])
    r, g, b = _hex_rgb(C.dominant_color(img))
    assert r > 180 and g < 80 and b < 80


def test_dominant_skips_white_when_colour_cluster_is_large():
    img = _stripes([((255, 255, 255), 0.75), ((30, 60, 200), 0.25)])
    r, g, b = _hex_rgb(C.dominant_color(img))
    assert b > 160 and r < 80


def test_dominant_prefers_saturated_over_grey():
    img = _stripes([((128, 128, 128), 0.5), ((240, 140, 20), 0.2), ((0, 0, 0), 0.3)])
    r, g, b = _hex_rgb(C.dominant_color(img))
    assert r > 200 and b < 60


def test_dominant_keeps_black_for_all_black_image():
    img = _stripes([((5, 5, 5), 1.0)])
    assert max(_hex_rgb(C.dominant_color(img))) < 20


def test_dominant_uses_small_accent_when_image_is_almost_all_black():
    img = _stripes([((2, 2, 2), 0.95), ((200, 120, 20), 0.05)])
    r, g, b = _hex_rgb(C.dominant_color(img))
    assert r > 150 and b < 60


def test_pick_dominant_rule_10_percent():
    centers = np.array([[0, 0, 0], [250, 250, 250], [20, 160, 60]], dtype=float)
    assert list(C.pick_dominant(centers, np.array([60, 25, 15]))) == [20, 160, 60]


def test_parse_item_prefers_variants_and_rejects_non_images(fixture_json):
    items = [nasa.parse_item(h) for h in fixture_json("nasa_search.json")["collection"]["items"]]
    first, second, video, third = items
    assert video is None
    assert nasa.image_url(first).endswith("PIA14417~medium.jpg")
    assert nasa.image_url(second).endswith("A2~thumb.jpg")
    assert first["center"] == "JPL" and second["photographer"] == "NASA/Kim Shiflett"
    assert third["links"] == {}


def test_dedup_by_id_and_exact_title(fixture_json):
    items = [nasa.parse_item(h) for h in fixture_json("nasa_search.json")["collection"]["items"]]
    items = [i for i in items if i]
    pools = nasa.dedup({"nebula": items, "galaxy": [items[0], items[2]]})
    ids = [i["nasa_id"] for v in pools.values() for i in v]
    # A2 has the same title as PIA14417 up to case and spacing; PIA14417 and A3 repeat by id
    assert sorted(ids) == ["A3", "PIA14417"]
    assert len(ids) == len(set(ids))


def test_dedup_round_robin_gives_shared_item_to_higher_rank():
    a = {"nasa_id": "X", "title": "x"}
    b = {"nasa_id": "Y", "title": "y"}
    pools = nasa.dedup({"k1": [b, a], "k2": [a]})
    assert [i["nasa_id"] for i in pools["k1"]] == ["Y"]
    assert [i["nasa_id"] for i in pools["k2"]] == ["X"]


def test_to_row_fields_and_truncation(fixture_json):
    it = nasa.parse_item(fixture_json("nasa_search.json")["collection"]["items"][0])
    it["description"] = "<b>Word</b> " * 600
    row = nasa.to_row(it, "nebula", "m.jpg", "t.jpg", "#123456")
    assert row["id"] == "nasa:PIA14417"
    assert row["topic"] == "nebula" and row["creator"] == "JPL" and row["date"] == "2011-08-10"
    assert row["source_url"] == "https://images.nasa.gov/details/PIA14417"
    assert "<b>" not in row["raw_text"] and len(row["raw_text"]) <= 1201


def test_to_row_uses_title_when_description_empty(fixture_json):
    it = nasa.parse_item(fixture_json("nasa_search.json")["collection"]["items"][1])
    row = nasa.to_row(it, "nebula", "m.jpg", "t.jpg", "#123456")
    assert row["raw_text"] == it["title"]
    assert row["creator"] == "NASA/Kim Shiflett"


def test_keyword_list():
    assert len(nasa.KEYWORDS) >= 12 and len(set(nasa.KEYWORDS)) == len(nasa.KEYWORDS)


def test_described_first_is_stable():
    a = {"nasa_id": "a", "title": "A", "description": ""}
    b = {"nasa_id": "b", "title": "B", "description": "<p>Real text.</p>"}
    c = {"nasa_id": "c", "title": "C", "description": "C"}  # description that only repeats the title
    d = {"nasa_id": "d", "title": "D", "description": "More text."}
    assert [i["nasa_id"] for i in nasa.described_first([a, b, c, d])] == ["b", "d", "a", "c"]
