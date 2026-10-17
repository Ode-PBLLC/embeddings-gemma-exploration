"""Real llama.cpp server (gallery-gemma4 container on :8081)."""
from pathlib import Path

import pytest

from gallery.llm import GemmaClient, word_count

pytestmark = pytest.mark.gpu
FX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def client():
    c = GemmaClient()
    try:
        ok = c.health()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Gemma server not reachable: {e}")
    if not ok:
        pytest.skip("Gemma server not healthy")
    yield c
    c.close()


def test_image_summary(client):
    item = {"id": "nasa:test", "source": "nasa", "modality": "image", "title": "spiral galaxy M83",
            "creator": "", "date": "", "raw_text": "Hubble image of a galaxy.",
            "media_path": str(FX / "galaxy.jpg")}
    s = client.summarize(item)
    assert 40 <= word_count(s) <= 120, s
    assert any(w in s.lower() for w in ("galaxy", "spiral", "stars")), s


def test_text_summary(client):
    raw = ("Impressionism is a 19th-century art movement characterized by relatively small, thin, yet "
           "visible brush strokes, open composition, emphasis on accurate depiction of light in its "
           "changing qualities, ordinary subject matter, unusual visual angles, and inclusion of movement "
           "as a crucial element of human perception and experience. Impressionism originated with a group "
           "of Paris-based artists whose independent exhibitions brought them to prominence during the "
           "1870s and 1880s.")
    s = client.summarize({"id": "wiki:test", "source": "wiki", "modality": "text",
                          "title": "Impressionism", "raw_text": raw})
    assert 40 <= word_count(s) <= 120, s
