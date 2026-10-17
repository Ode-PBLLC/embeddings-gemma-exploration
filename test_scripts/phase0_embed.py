"""Phase 0 measurements for EmbeddingGemma 2. Prints JSON to stdout."""
import json, time, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "src")
from gallery.embed import Embedder

ROOT = Path(__file__).resolve().parents[1]
FX = ROOT / "tests/fixtures"
res = {}
t0 = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
e = Embedder()
res["load_s"] = round(time.perf_counter() - t0, 1)
res["after_load_alloc_MiB"] = round(torch.cuda.memory_allocated() / 2**20)
imgs = e.embed_images([FX / "galaxy.jpg", FX / "still_life.jpg"])
q = e.embed_texts(["a spiral galaxy", "a bowl of fruit"], kind="query")
res["sim_query_x_image"] = (q @ imgs.T).round(4).tolist()
it = e.embed_images([FX / "galaxy.jpg", FX / "still_life.jpg"], titles=["spiral galaxy M83", "The Plate of Apples"])
res["sim_query_x_image_plus_title"] = (q @ it.T).round(4).tolist()
# latency: single text query, 50 runs after 3 warmups
for _ in range(3): e.embed_texts(["warmup query"], kind="query")
lat = []
for i in range(50):
    torch.cuda.synchronize(); t = time.perf_counter()
    e.embed_texts([f"paintings of the sea at night {i}"], kind="query")
    torch.cuda.synchronize(); lat.append((time.perf_counter() - t) * 1000)
res["query_ms_p50"] = round(float(np.percentile(lat, 50)), 1)
res["query_ms_p95"] = round(float(np.percentile(lat, 95)), 1)
# image throughput: 64 images (fixtures repeated) at batch sizes
paths = [FX / "galaxy.jpg", FX / "still_life.jpg"] * 32
for bs in (8, 16, 32):
    e.image_batch = bs
    e.embed_images(paths[:bs])
    torch.cuda.synchronize(); t = time.perf_counter()
    e.embed_images(paths)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    res[f"images_per_s_bs{bs}"] = round(len(paths) / dt, 1)
texts = ["The Art Institute of Chicago holds a large collection of Impressionist paintings. " * 6] * 128
torch.cuda.synchronize(); t = time.perf_counter()
e.embed_texts(texts, kind="document", titles=["Impressionism"] * 128)
torch.cuda.synchronize(); res["doc_texts_per_s_bs32"] = round(128 / (time.perf_counter() - t), 1)
res["peak_alloc_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20)
res["peak_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / 2**20)
res["dtype"] = e.dtype
print(json.dumps(res, indent=1))
