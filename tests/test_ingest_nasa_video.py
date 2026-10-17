"""Offline tests for the NASA video ingester's filters and parsers."""
import pytest

from gallery.ingest.nasa_video import CATALOG_TITLE, EXCLUDE, meta_duration, parse_item, vtt_text


def test_vtt_text_drops_timings_tags_and_repeats():
    raw = ("WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nThe aurora <i>glows</i>\n\n"
           "2\n00:00:02.000 --> 00:00:04.000\nThe aurora <i>glows</i>\nover Canada\n")
    assert vtt_text(raw) == "The aurora glows over Canada"


@pytest.mark.parametrize("meta, want", [
    ({"QuickTime:Duration": "0:02:12"}, 132.0),
    ({"QuickTime:Duration": "2:10:51"}, 7851.0),
    ({"QuickTime:Duration": "12.3 s"}, 12.3),
    ({"QuickTime:TrackDuration": "0:00:45"}, 45.0),
    ({}, None),
])
def test_meta_duration(meta, want):
    assert meta_duration(meta) == want


@pytest.mark.parametrize("title, dropped", [
    ("The Aurora Named STEVE", False),
    ("Jupiter's Hot Spots", False),
    ("NASA Science Live: Aurora Glow", True),
    ("How Can I See the Northern Lights? We Asked a NASA Expert", True),
    ("Mars Rover Update", True),
    ("Sun-Earth Day 2009 Promo 1", True),
    ("KSC-04-S-00193", True),
])
def test_title_filters(title, dropped):
    assert bool(EXCLUDE.search(title) or CATALOG_TITLE.match(title)) is dropped


def test_parse_item_keeps_videos_only():
    hit = {"data": [{"media_type": "video", "nasa_id": "X1", "title": "Solar Flare Erupts", "description": "A flare."}],
           "href": "https://example/collection.json", "links": [{"rel": "preview", "href": "https://example/t.jpg"}]}
    assert parse_item(hit)["nasa_id"] == "X1"
    hit["data"][0]["media_type"] = "image"
    assert parse_item(hit) is None
