"""Embed the parity queries with the server's own embedder code (gallery.embed.Embedder) on the CPU.

    uv run python test_scripts/static_parity/embed_torch.py bfloat16|float32

The running server uses bf16 on the GPU, which has no room for a second copy, so this is the
closest stand-in for the server's query vector; analyze.py checks it against the server's answers.
Writes out/torch_cpu_<dtype>.f32 in queries.json order (gallery then moon).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from gallery.embed import Embedder

HERE = Path(__file__).resolve().parent

dtype = sys.argv[1]
sets = json.loads((HERE / "queries.json").read_text())
queries = [q["q"].strip() for q in sets["gallery"] + sets["moon"]]
t = time.perf_counter()
model = Embedder(device="cpu", dtype=dtype)
vecs = model.embed_texts(queries, kind="query")
(HERE / "out" / f"torch_cpu_{dtype}.f32").write_bytes(vecs.astype("<f4").tobytes())
print(f"{dtype}: {vecs.shape} in {time.perf_counter() - t:.0f} s")
