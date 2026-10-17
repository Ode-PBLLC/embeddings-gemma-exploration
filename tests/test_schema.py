"""One test per gallery.schema.validate() rejection rule, plus the happy path."""

import pandas as pd
import pytest

from gallery import schema
from gallery.schema import COLUMNS, WIKI_COLOR, SchemaError, validate


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(schema, "ROOT", tmp_path)
    for p in ("data/media/aic/1.jpg", "data/thumbs/aic/1.jpg"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_bytes(b"jpg")
    return tmp_path


def good() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "aic:1", "source": "aic", "modality": "image", "title": "Water Lilies",
         "raw_text": "A pond.", "media_path": "data/media/aic/1.jpg", "thumb_path": "data/thumbs/aic/1.jpg",
         "source_url": "https://www.artic.edu/artworks/1", "license": "CC0", "color_hex": "#3a6b8c",
         "topic": "Painting and Sculpture of Europe", "creator": "Claude Monet", "date": "1906"},
        {"id": "wiki:9:0", "source": "wiki", "modality": "text", "title": "Claude Monet",
         "raw_text": "Oscar-Claude Monet was a French painter.", "media_path": "", "thumb_path": "",
         "source_url": "https://en.wikipedia.org/wiki/Claude_Monet", "license": "CC BY-SA 4.0",
         "color_hex": WIKI_COLOR, "topic": "Claude Monet", "creator": "", "date": ""},
    ])[COLUMNS]


def test_valid_frame_passes(root):
    df = good()
    assert validate(df) is df


def test_rejects_missing_column(root):
    with pytest.raises(SchemaError, match="missing columns"):
        validate(good().drop(columns=["topic"]))


def test_rejects_duplicate_id(root):
    df = good()
    df.loc[1, "id"] = "aic:1"
    with pytest.raises(SchemaError, match="duplicate ids"):
        validate(df)


def test_rejects_bad_source(root):
    df = good()
    df.loc[0, "source"] = "met"
    with pytest.raises(SchemaError, match="bad source"):
        validate(df)


def test_rejects_bad_modality(root):
    df = good()
    df.loc[0, "modality"] = "smell"
    with pytest.raises(SchemaError, match="bad modality"):
        validate(df)


@pytest.mark.parametrize("color", ["#3A6B8C", "3a6b8c", "#3a6b8", "red", None])
def test_rejects_bad_color_hex(root, color):
    df = good()
    df.loc[0, "color_hex"] = color
    with pytest.raises(SchemaError, match="bad color_hex"):
        validate(df)


@pytest.mark.parametrize("col", ["title", "source_url", "raw_text"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_rejects_empty_required_text(root, col, value):
    df = good()
    df.loc[0, col] = value
    with pytest.raises(SchemaError, match=f"empty {col}"):
        validate(df)


def test_rejects_image_without_thumb_on_disk(root):
    (root / "data/thumbs/aic/1.jpg").unlink()
    with pytest.raises(SchemaError, match="thumb_path missing on disk"):
        validate(good())


def test_rejects_image_without_media_on_disk(root):
    (root / "data/media/aic/1.jpg").unlink()
    with pytest.raises(SchemaError, match="media_path missing on disk"):
        validate(good())


def test_rejects_image_with_empty_thumb_path(root):
    df = good()
    df.loc[0, "thumb_path"] = ""
    with pytest.raises(SchemaError, match="thumb_path missing"):
        validate(df)


def test_check_files_false_skips_disk_check(root):
    (root / "data/thumbs/aic/1.jpg").unlink()
    validate(good(), check_files=False)
