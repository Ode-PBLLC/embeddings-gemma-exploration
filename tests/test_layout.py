"""US-012: 3D UMAP layout, deterministic, scaled to [-1, 1], with a neighbourhood report."""
import numpy as np
import pytest

from gallery.layout import neighbourhood_overlap, run, scale_unit
from gallery.paths import data_dir


def clustered(n=400, d=64, k=8, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(k, d))
    v = c[rng.integers(k, size=n)] + rng.normal(0, 0.6, size=(n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    out = []
    for i in range(2):
        d = tmp_path_factory.mktemp(f"layout{i}")
        np.save(d / "vectors.npy", clustered())
        layout, share = run(d, d / "layout.md")
        out.append((d, layout, share))
    return out


def test_shape_and_scale(two_runs):
    d, layout, _ = two_runs[0]
    saved = np.load(d / "layout.npy")
    assert saved.shape == (400, 3) and saved.dtype == np.float32
    assert np.allclose(saved.min(0), -1) and np.allclose(saved.max(0), 1)


def test_deterministic_byte_equal(two_runs):
    (a, _, _), (b, _, _) = two_runs
    assert (a / "layout.npy").read_bytes() == (b / "layout.npy").read_bytes()


def test_report_and_neighbourhood(two_runs):
    d, _, share = two_runs[0]
    text = (d / "layout.md").read_text()
    assert "PASS" in text or "FAIL" in text
    assert f"{share:.3f}" in text
    assert share >= 0.6  # well-clustered input should preserve neighbourhoods


def test_scale_unit_constant_axis():
    x = np.array([[0, 1, 5], [2, 1, 7]], float)
    s = scale_unit(x)
    assert s[:, 0].tolist() == [-1, 1] and s[:, 2].tolist() == [-1, 1]
    assert np.isfinite(s).all()


def test_overlap_identity():
    v = clustered(n=120)
    # a "layout" equal to the vectors' own first 3 PCs is not identity, but a perfect copy is:
    ov = neighbourhood_overlap(v, v)  # euclidean on unit vectors ranks like cosine
    assert (ov == 10).all()


@pytest.mark.skipif(not (data_dir() / "layout.npy").is_file(), reason="real layout not built yet")
def test_real_layout():
    layout = np.load(data_dir() / "layout.npy")
    vectors = np.load(data_dir() / "vectors.npy", mmap_mode="r")
    assert layout.shape == (len(vectors), 3)
    assert layout.min() >= -1 - 1e-6 and layout.max() <= 1 + 1e-6
