"""Ad hoc: do EmbeddingGemma 2 video embeddings (16 sampled frames) land near related works?
For each trial clip: nearest existing items (current image+title / text vectors), and the rank of the
clip itself for a text query about its topic, among all items plus the clips."""
import json
import numpy as np, pandas as pd, torch
from PIL import Image
from gallery.embed import Embedder, QUERY_PROMPT
from gallery.ingest.nasa_video import frames_dir
from gallery.schema import DATA

v = pd.read_parquet(DATA / "raw" / "nasa_video.parquet")
v = v[~v["title"].str.match(r"^[A-Z]{2,6}[-_ ]?\d")]
items = pd.read_parquet(DATA / "items.parquet")
vecs = np.load(DATA / "vectors.npy")
emb = Embedder()
torch.cuda.reset_peak_memory_stats()
frames = [[Image.open(p).convert("RGB") for p in sorted(frames_dir(i).glob("f*.jpg"))] for i in v["id"]]
pure = emb._encode([{"video": f} for f in frames], 1)
titled = emb._encode([{"text": f"title: {t} | <|video|>", "video": f} for t, f in zip(v["title"], frames)], 1)
print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
q = emb.embed_texts([f"{t}" for t in v["topic"]], kind="query")
for name, V in [("frames only", pure), ("frames + title", titled)]:
    print(f"\n=== {name} ===")
    allv = np.vstack([vecs, V])
    ranks = []
    for k, (title, topic) in enumerate(zip(v["title"], v["topic"])):
        sims = vecs @ V[k]
        top = np.argsort(-sims)[:4]
        r = int((allv @ q[k] > allv[len(vecs) + k] @ q[k]).sum()) + 1
        ranks.append(r)
        print(f"- {title[:44]:44} [{topic}] query-rank {r:4d} | near: " + "; ".join(f"{items.source[i]}:{items.title[i][:26]} {sims[i]:.2f}" for i in top))
    print(f"median rank of the clip for its own topic query: {int(np.median(ranks))} of {len(allv)}")
