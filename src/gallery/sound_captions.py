"""Short static captions for the sound works ("Wind on Mars"), shown on wordless sound cards.

GEMMA_BASE_URL=... uv run python -m gallery.sound_captions
Writes "caption" into each sound's sidecar JSON (data/media/nasa_audio/<id>.json).
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from gallery.ingest.nasa_audio import audio_paths
from gallery.llm import SOUND_CAPTION_SYSTEM, GemmaClient, _clean, _meta_block
from gallery.schema import DATA

log = logging.getLogger("gallery.sound_captions")


def caption_for(client: GemmaClient, item: dict) -> str:
    msgs = [{"role": "system", "content": SOUND_CAPTION_SYSTEM},
            {"role": "user", "content": _meta_block(item) + "\n\nName the sound now."}]
    for attempt in range(3):
        c = _clean(client.complete(msgs, temperature=0.3 + 0.2 * attempt)).rstrip(".").strip("\"' ")
        if 2 <= len(c.split()) <= 7:
            return c
    raise RuntimeError(f"no 2-7 word caption for {item['id']} (last: {c!r})")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    items = pd.read_parquet(DATA / "items.parquet")
    client = GemmaClient()
    for r in items[items["modality"] == "audio"].to_dict("records"):
        side = audio_paths(r["id"])[1]
        meta = json.loads(side.read_text())
        if meta.get("speech", True):        # speech shows its transcript; only wordless sounds need a caption
            meta.pop("caption", None)
            side.write_text(json.dumps(meta))
            continue
        meta["caption"] = caption_for(client, r)
        side.write_text(json.dumps(meta))
        log.info("%-55s -> %s", r["title"][:55], meta["caption"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
