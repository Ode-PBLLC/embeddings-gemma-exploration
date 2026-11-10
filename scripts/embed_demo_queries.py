"""Precompute query vectors for the suggested searches on the static site.

    uv run python scripts/embed_demo_queries.py [--device cpu|cuda]

The static site embeds queries in the browser, which first downloads the model. These prompts are
embedded here with the server's own code (gallery.embed.Embedder, bf16), so they answer at once,
before the download finishes. Writes static/demo_queries.json:
    {"gallery": [{"q": ..., "v": <base64 float32 x 768>}, ...], "moon": [...]}
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "demo_queries.json"

# the scripted demo queries (eval/queries.yaml) and the best-mixed walls from reports/demo_prompt_sweep.md
GALLERY = [
    "fire and heat", "light on water", "loneliness", "the color blue", "machines that fly",
    "the night sky", "storms and turbulent skies", "forests", "volcanoes", "glaciers", "the ocean",
    "the sun", "mountains", "birds", "mirrors", "the sound of mars", "listening", "voyages",
]
MOON = [
    "a huge impact basin", "dark smooth lava plains", "a fresh crater with bright rays", "mountain ranges",
    "winding rilles", "the Sea of Tranquility", "the lunar south pole", "a crater with a central peak",
    "Tycho", "chains of small craters",
]


def encode(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode()


def main() -> None:
    from gallery.embed import Embedder

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model = Embedder(device=args.device, dtype="bfloat16")
    out = {}
    for name, qs in (("gallery", GALLERY), ("moon", MOON)):
        vecs = model.embed_texts(qs, kind="query")
        out[name] = [{"q": q, "v": encode(v)} for q, v in zip(qs, vecs)]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{OUT}: {len(GALLERY)} gallery + {len(MOON)} moon prompts")


if __name__ == "__main__":
    main()
