"""Small synthetic dataset for the interface tests (search API, assets)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic():
    spec = importlib.util.spec_from_file_location("make_synthetic", ROOT / "scripts" / "make_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_dataset(out: Path, n: int = 240, seed: int = 3) -> Path:
    """Synthetic data in `out` with a random (non-UMAP) layout and a built asset bundle in out/web."""
    from gallery.build_assets import build

    _make_synthetic().build(out, n=n, seed=seed, layout=False)
    rng = np.random.default_rng(seed)
    np.save(out / "layout.npy", rng.uniform(-1, 1, size=(n, 3)).astype(np.float32))
    build(out, out / "web")
    return out
