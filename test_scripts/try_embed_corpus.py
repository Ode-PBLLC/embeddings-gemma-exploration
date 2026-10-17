"""Run embed_corpus on a 40-row sample (20 aic + 20 nasa + 6 fake text) into a scratch dir; then resume."""
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from gallery import embed_corpus as ec
from gallery.schema import COLUMNS

aic = pd.read_parquet("data/raw/aic.parquet").head(20)
nasa = pd.read_parquet("data/raw/nasa.parquet").head(20)
txt = pd.DataFrame([{c: "" for c in COLUMNS} | {"id": f"wiki:{i}", "source": "wiki", "modality": "text",
                     "title": "Impressionism", "raw_text": f"Paragraph {i} about light and colour in painting."} for i in range(6)])
df = pd.concat([aic[COLUMNS], txt, nasa[COLUMNS]], ignore_index=True)
tmp = Path(tempfile.mkdtemp())
df.to_parquet(tmp / "items.parquet")
ec.ITEMS, ec.VECTORS, ec.VECTOR_IDS, ec.CHUNKS = tmp / "items.parquet", tmp / "v.npy", tmp / "ids.json", tmp / "chunks"
assert ec.main(["--chunk", "16"]) == 0
v = np.load(tmp / "v.npy"); ids = json.loads((tmp / "ids.json").read_text())
assert v.shape == (46, 768) and ids == df["id"].tolist()
print("norms", np.linalg.norm(v, axis=1).min(), np.linalg.norm(v, axis=1).max())
assert ec.main(["--chunk", "16"]) == 0  # resume: all chunks cached
assert np.array_equal(np.load(tmp / "v.npy"), v)
print("OK", tmp)
