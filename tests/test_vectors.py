"""data/vectors.npy invariants (skips until embed_corpus has run)."""
import json

import numpy as np
import pandas as pd
import pytest

from gallery.schema import DATA

VEC, IDS, ITEMS = DATA / "vectors.npy", DATA / "vector_ids.json", DATA / "items.parquet"
pytestmark = pytest.mark.skipif(not (VEC.exists() and IDS.exists() and ITEMS.exists()),
                                reason="vectors not built yet")


def test_vectors():
    v = np.load(VEC)
    ids = json.loads(IDS.read_text())
    items = pd.read_parquet(ITEMS, columns=["id"])
    assert v.dtype == np.float32
    assert v.shape == (len(items), 768)
    assert ids == items["id"].tolist()
    assert not np.isnan(v).any()
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)
