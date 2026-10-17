"""Ad hoc: (1) loading the audio encoder must not change text/image embeddings (the live server embeds
queries without it); (2) do sound embeddings land near related works?"""
import gc
import numpy as np, pandas as pd, torch
from gallery.embed import Embedder
from gallery.ingest.nasa_audio import audio_paths
from gallery.schema import DATA, ROOT

texts = ["wind on Mars", "a rocket engine test", "the color blue"]
img = ROOT / "tests" / "fixtures" / "galaxy.jpg"
a = Embedder()
t0, i0 = a.embed_texts(texts, kind="query"), a.embed_images([img], titles=["A spiral galaxy"])
del a; gc.collect(); torch.cuda.empty_cache()
b = Embedder(with_audio=True)
t1, i1 = b.embed_texts(texts, kind="query"), b.embed_images([img], titles=["A spiral galaxy"])
print("text cos (no-audio vs audio-loaded):", np.round((t0 * t1).sum(1), 6), " image:", np.round((i0 * i1).sum(1), 6))

s = pd.read_parquet(DATA / "raw" / "nasa_audio.parquet")
items = pd.read_parquet(DATA / "items.parquet")
vecs = np.load(DATA / "vectors.npy")
torch.cuda.reset_peak_memory_stats()
A = b.embed_audio([audio_paths(i)[0] for i in s["id"]], titles=s["title"].tolist())
print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
for k, (t, topic) in enumerate(zip(s["title"], s["topic"])):
    sims = vecs @ A[k]
    top = np.argsort(-sims)[:3]
    print(f"- {t[:46]:46} [{topic}] near: " + "; ".join(f"{items.modality[i][:3]}:{items.title[i][:24]} {sims[i]:.2f}" for i in top))
q = b.embed_texts(["sounds from Mars", "a rocket launch", "a marsquake", "sonic boom"], kind="query")
for qq, name in zip(q, ["sounds from Mars", "a rocket launch", "a marsquake", "sonic boom"]):
    allv = np.vstack([vecs, A]); sims = allv @ qq
    top = np.argsort(-sims)[:12]
    n_snd = int((top >= len(vecs)).sum())
    print(f"query {name!r}: {n_snd} of the top 12 are sounds; best sound rank {int(np.argmax(np.isin(np.argsort(-sims), np.arange(len(vecs), len(allv))))) + 1}")
