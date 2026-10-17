"""Ingest the Art Institute of Chicago audio-guide tracks for artworks already in the corpus.

uv run python -m gallery.ingest.aic_audio          (after gallery.ingest.aic)

One sound per artwork (its first linked track), as its own sound work that sits next to the
painting in the embedding space. Per track:
  data/media/aic/sound-<uuid>.mp3          the guide as AIC serves it (what the UI plays)   = media_path
  data/media/nasa_audio/sound-<uuid>.wav   16 kHz mono, up to 5 min (what EmbeddingGemma hears)
  data/media/nasa_audio/sound-<uuid>.json  {"duration", "peaks", "artwork_id"} for the waveform and the link back
Colour, creator, date and department come from the artwork the guide describes.

License: unlike AIC's public-domain images, the audio is AIC's own material: "noncommercial
educational and personal use ... retain all copyright and other proprietary notices ... cite the
author and source" (AIC API terms). Every row carries that notice; the cards show it.
"""

from __future__ import annotations

import argparse
import json
import logging
import re

import pandas as pd

from gallery.ingest import _common as C
from gallery.ingest.aic import HEADERS
from gallery.ingest.nasa_audio import AUDIO_DIR, EMBED_S, MAX_S, MIN_RMS_DB, peaks_and_level
from gallery.ingest.nasa_video import duration_s, ffmpeg
from gallery.schema import DATA, validate

log = logging.getLogger("gallery.ingest.aic_audio")

SOURCE = "aic"
ARTWORKS = "https://api.artic.edu/api/v1/artworks"
SOUNDS = "https://api.artic.edu/api/v1/sounds/{id}"
LICENSE = "© Art Institute of Chicago (audio guide); noncommercial educational use per AIC terms"
CACHE = DATA / "raw" / "aic_audio_cache"

api_limiter = C.RateLimiter(1.0)
asset_limiter = C.RateLimiter(0.5)


def linked_sounds(artwork_ids: list[str]) -> dict[str, str]:
    """artwork id -> its first linked sound id (AIC artworks API, 100 ids per request, cached)."""
    out = {}
    for k in range(0, len(artwork_ids), 100):
        chunk = artwork_ids[k:k + 100]
        params = {"ids": ",".join(chunk), "fields": "id,sound_ids", "limit": 100}
        data = C.cached_json(SOURCE, f"artwork_sounds_{C.cache_key(chunk)}",
                             lambda: C.fetch(ARTWORKS, params=params, headers=HEADERS, limiter=api_limiter).json())
        for a in data["data"]:
            if a.get("sound_ids"):
                out[str(a["id"])] = a["sound_ids"][0]
    return out


def sound_record(sound_id: str) -> dict:
    return C.cached_json(SOURCE, f"sound_{sound_id}", lambda: C.fetch(
        SOUNDS.format(id=sound_id), headers=HEADERS, limiter=api_limiter).json())["data"]


def clean_title(raw: str, artwork_title: str) -> str:
    """'T28 Shiva As Lord Of The Dance Nataraja 24548' -> the artwork's own title, as the guide is about it."""
    return f"{artwork_title} — audio guide"


def fetch_one(art: dict, sound_id: str) -> dict | None:
    native = f"sound-{sound_id}"
    mp3 = DATA / "media" / SOURCE / f"{native}.mp3"
    wav, side = AUDIO_DIR / f"{native}.wav", AUDIO_DIR / f"{native}.json"
    try:
        rec = sound_record(sound_id)
        url = rec.get("content")
        if not url:
            raise ValueError("sound has no content url")
        if not mp3.is_file():
            mp3.parent.mkdir(parents=True, exist_ok=True)
            r = C.fetch(url, headers=HEADERS, limiter=asset_limiter, timeout=120)
            if "audio" not in r.headers.get("content-type", ""):
                raise ValueError(f"not audio: {r.headers.get('content-type')}")
            tmp = mp3.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(mp3)
        dur = duration_s(mp3)
        if not (8 <= dur <= MAX_S):
            raise ValueError(f"duration {dur:.0f}s outside 8-{MAX_S}s")
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        if not wav.is_file():
            ffmpeg("-i", str(mp3), "-vn", "-ac", "1", "-ar", "16000", "-t", str(EMBED_S), str(wav))
        peaks, loud_db = peaks_and_level(wav)
        if loud_db < MIN_RMS_DB:
            raise ValueError(f"effectively silent (loudest {loud_db:.0f} dBFS)")
        side.write_text(json.dumps({"duration": round(dur, 2), "peaks": peaks, "artwork_id": art["id"]}))
    except Exception as e:  # noqa: BLE001 - every per-item failure is logged and the item dropped
        C.log_failure(SOURCE, f"aic:{native}", "audio", e)
        for p in (wav, side):
            p.unlink(missing_ok=True)
        return None
    return {
        "id": f"aic:{native}", "source": SOURCE, "modality": "audio", "title": clean_title(rec.get("title", ""), art["title"]),
        "raw_text": art["raw_text"], "media_path": C.rel(mp3), "thumb_path": "",
        "source_url": art["source_url"], "license": LICENSE, "color_hex": art["color_hex"],
        "topic": art["topic"], "creator": art["creator"], "date": art["date"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    C.setup_logging()
    arts = pd.read_parquet(DATA / "raw" / "aic.parquet")
    arts["native"] = arts["id"].str.split(":").str[1]
    links = linked_sounds(arts["native"].tolist())
    log.info("%d of %d artworks have an audio guide", len(links), len(arts))
    jobs, seen = [], set()
    for a in arts.to_dict("records"):   # a track shared by several artworks hangs once, with the first
        sid = links.get(a["native"])
        if sid and sid not in seen:
            seen.add(sid)
            jobs.append((a, sid))
    rows = [r for r in C.run_pool(lambda j: fetch_one(*j), jobs, workers=3) if r]
    df = pd.DataFrame(rows)
    validate(df)
    out = C.write_parquet(df, "aic_audio")
    log.info("wrote %s: %d rows", out, len(df))
    print(df["topic"].value_counts().to_string())


if __name__ == "__main__":
    main()
