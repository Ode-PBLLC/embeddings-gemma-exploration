"""Offline tests for the AIC ingester's parsing and conversion helpers."""

import io

import numpy as np
import pytest
from PIL import Image

from gallery.ingest import _common as C
from gallery.ingest import aic


@pytest.mark.parametrize("h,s,l,expected", [
    (0, 100, 50, "#ff0000"),
    (120, 100, 25, "#008000"),
    (240, 100, 50, "#0000ff"),
    (0, 0, 50, "#808080"),
    (0, 0, 100, "#ffffff"),
    (0, 0, 0, "#000000"),
    (360, 100, 50, "#ff0000"),  # hue wraps
    (30, 50, 50, "#bf8040"),
])
def test_hsl_to_hex(h, s, l, expected):
    assert C.hsl_to_hex(h, s, l) == expected


def test_color_of_uses_api_hsl_and_none_when_missing(fixture_json):
    recs = fixture_json("aic_records.json")
    assert aic.color_of(recs[2]) == "#bf8040"
    assert aic.color_of(recs[1]) is None
    assert aic.color_of({"color": {"h": 10, "s": None, "l": 5}}) is None


def test_keep_requires_public_domain_and_image(fixture_json):
    recs = fixture_json("aic_records.json")
    assert [aic.keep(r) for r in recs] == [True, True, True, False, False]


def test_raw_text_prefers_description_stripped_of_html(fixture_json):
    rec = fixture_json("aic_records.json")[0]
    text = aic.compose_text(rec)
    assert text == "One of four posts sculpted for the palace of the king (ogoga) of Ikere & a masterwork."


def test_raw_text_falls_back_to_short_description(fixture_json):
    rec = fixture_json("aic_records.json")[1]
    assert aic.compose_text(rec) == "A painted silk fan."


def test_raw_text_composed_when_no_description(fixture_json):
    rec = fixture_json("aic_records.json")[2]
    text = aic.compose_text(rec)
    assert text == ("Study of a Horse by Edgar Degas (c. 1866). Graphite on paper. "
                    "Pencil sketch of a horse.")


def test_raw_text_never_empty_even_with_bare_record():
    assert aic.compose_text({"title": None}).strip() != ""


def test_to_row_maps_fields(fixture_json):
    rec = fixture_json("aic_records.json")[0]
    row = aic.to_row(rec, "data/media/aic/102611.jpg", "data/thumbs/aic/102611.jpg", "#123456")
    assert row["id"] == "aic:102611"
    assert row["creator"] == "Olowe of Ise (died 1938)"  # first line of artist_display
    assert row["topic"] == "Arts of Africa"
    assert row["source_url"] == "https://www.artic.edu/artworks/102611"
    assert row["license"] == "CC0 (image); CC BY 4.0 (text)"
    assert row["modality"] == "image"


def _jpeg(color, size=(400, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


class _Resp:
    def __init__(self, content):
        self.content = content
        self.headers = {"content-type": "image/jpeg"}


def test_fetch_one_resizes_and_falls_back_to_dominant_color(fixture_json, tmp_path, monkeypatch):
    rec = fixture_json("aic_records.json")[1]  # no API colour
    monkeypatch.setattr(C, "media_paths", lambda s, n: (tmp_path / "m" / f"{n}.jpg", tmp_path / "t" / f"{n}.jpg"))
    monkeypatch.setattr(C, "rel", lambda p: str(p))
    calls = []

    def fake_fetch(url, **kw):
        calls.append((url, kw.get("headers")))
        return _Resp(_jpeg((200, 30, 30), (1600, 900)))

    monkeypatch.setattr(C, "fetch", fake_fetch)
    row = aic.fetch_one("Textiles", rec)
    assert row is not None
    assert calls[0][0] == "https://www.artic.edu/iiif/2/abc/full/843,/0/default.jpg"
    assert "AIC-User-Agent" in calls[0][1]
    assert max(Image.open(row["media_path"]).size) == 768
    assert max(Image.open(row["thumb_path"]).size) == 320
    r, g, b = (int(row["color_hex"][i:i + 2], 16) for i in (1, 3, 5))
    assert r > 180 and g < 60 and b < 60


def test_fetch_one_logs_and_drops_on_failure(fixture_json, tmp_path, monkeypatch):
    rec = fixture_json("aic_records.json")[0]
    monkeypatch.setattr(C, "media_paths", lambda s, n: (tmp_path / "m.jpg", tmp_path / "t.jpg"))
    monkeypatch.setattr(C, "FAILURES", tmp_path / "fail.jsonl")

    def boom(url, **kw):
        raise RuntimeError("HTTP 403")

    monkeypatch.setattr(C, "fetch", boom)
    assert aic.fetch_one("Arts of Africa", rec) is None
    log = (tmp_path / "fail.jsonl").read_text()
    assert '"id": "aic:102611"' in log and "HTTP 403" in log
    assert not (tmp_path / "m.jpg").exists()


def test_allocate_balances_and_respects_capacity():
    q = C.allocate({"a": 1000, "b": 1000, "c": 10, "d": 5}, 100)
    assert sum(q.values()) == 100 and q["c"] == 10 and q["d"] == 5
    assert abs(q["a"] - q["b"]) <= 1


def test_allocate_when_capacity_short():
    assert C.allocate({"a": 3, "b": 2}, 10) == {"a": 3, "b": 2}


def test_select_and_fetch_tops_up_failures_and_caps_topics():
    pools = {"big": list(range(100)), "small": list(range(5))}
    # every third item of "big" fails
    fetch_one = lambda t, c: None if (t == "big" and c % 3 == 0) else {"topic": t, "c": c}
    rows = C.select_and_fetch(pools, target=40, cap=30, fetch_one=fetch_one, workers=2)
    topics = [r["topic"] for r in rows]
    assert len(rows) == 35  # big is capped at 30, small has only 5
    assert topics.count("big") == 30 and topics.count("small") == 5
    assert all(r["c"] % 3 for r in rows if r["topic"] == "big")


def test_iiif_url_never_upscales():
    big = {"image_id": "abc", "thumbnail": {"width": 4000}}
    small = {"image_id": "abc", "thumbnail": {"width": 750}}
    unknown = {"image_id": "abc", "thumbnail": None}
    assert aic.iiif_url(big) == "https://www.artic.edu/iiif/2/abc/full/843,/0/default.jpg"
    assert aic.iiif_url(small) == "https://www.artic.edu/iiif/2/abc/full/750,/0/default.jpg"
    assert aic.iiif_url(unknown) == "https://www.artic.edu/iiif/2/abc/full/843,/0/default.jpg"
