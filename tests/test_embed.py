"""EmbeddingGemma 2 on the GPU: shapes, normalization, cross-modal sanity."""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
FX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def emb():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from gallery.embed import Embedder
    return Embedder()


def _check(a, n):
    assert a.dtype == np.float32
    assert a.shape == (n, 768)
    assert np.isfinite(a).all()
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-3)


def test_text_shapes(emb):
    _check(emb.embed_texts(["a spiral galaxy", "a bowl of fruit", "x"], kind="query"), 3)
    _check(emb.embed_texts(["Some paragraph about Monet."]), 1)
    _check(emb.embed_texts(["Some paragraph about Monet."], titles=["Claude Monet"]), 1)


def test_image_shapes(emb):
    paths = [FX / "galaxy.jpg", FX / "still_life.jpg"]
    _check(emb.embed_images(paths), 2)
    _check(emb.embed_images(paths, titles=["spiral galaxy M83", "The Plate of Apples"]), 2)


def test_cross_modal_sanity(emb):
    imgs = emb.embed_images([FX / "galaxy.jpg", FX / "still_life.jpg"])
    q = emb.embed_texts(["a spiral galaxy", "a bowl of fruit"], kind="query")
    s = q @ imgs.T  # rows: queries, cols: [galaxy, still_life]
    assert s[0, 0] > s[0, 1], s
    assert s[1, 1] > s[1, 0], s


def test_bad_kind(emb):
    with pytest.raises(ValueError):
        emb.embed_texts(["x"], kind="banana")
