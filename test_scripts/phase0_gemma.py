"""Phase 0 timing for Gemma 4 E4B via llama.cpp: mean s/image item (sequential, 10) and 4-way throughput."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "src")
from gallery.llm import GemmaClient, word_count

ROOT = Path(__file__).resolve().parents[1]
items = [
    {"id": "nasa:fx", "source": "nasa", "modality": "image", "title": "spiral galaxy M83", "creator": "",
     "date": "", "raw_text": "This Hubble image shows the spiral galaxy M83, also known as the Southern Pinwheel.",
     "media_path": str(ROOT / "tests/fixtures/galaxy.jpg")},
    {"id": "aic:65811", "source": "aic", "modality": "image", "title": "The Plate of Apples", "creator": "Paul Cezanne",
     "date": "1877", "raw_text": "Oil on canvas.", "media_path": str(ROOT / "tests/fixtures/still_life.jpg")},
]
extra = [Path(p) for p in sys.argv[1:]]
c = GemmaClient(max_attempts=3)
c.summarize(items[0])  # warm
seq = []
outs = []
for i in range(10):
    it = items[i % 2]
    t = time.perf_counter(); s = c.summarize(it); seq.append(time.perf_counter() - t); outs.append((it["id"], word_count(s), s))
t = time.perf_counter()
with ThreadPoolExecutor(4) as ex:
    par = list(ex.map(lambda i: c.summarize(items[i % 2]), range(12)))
dt = time.perf_counter() - t
print(json.dumps({"seq_mean_s": round(sum(seq) / len(seq), 2), "seq_all_s": [round(x, 2) for x in seq],
                  "parallel4_12items_s": round(dt, 1), "parallel4_s_per_item": round(dt / 12, 2),
                  "examples": outs[:2] + [("aic:65811", word_count(par[1]), par[1])]}, indent=1))
