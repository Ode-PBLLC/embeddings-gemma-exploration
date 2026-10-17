"""Ingest ~40 NASA sound works: the audio track of NASA videos that are about sound (Mars microphone
recordings, marsquakes, sonifications, launches and engine tests), one sound per source video.

uv run python -m gallery.ingest.nasa_audio [--target 40]

Per sound, from the full-length ~preview.mp4 (shared download cache with the video ingester):
  data/media/nasa/<id>.m4a            the sound the UI plays (mono AAC, up to 5.5 min)       = media_path
  data/media/nasa_audio/<id>.wav      16 kHz mono, up to 5 min: what EmbeddingGemma hears
  data/media/nasa_audio/<id>.json     {"duration": s, "peaks": [0..255 x 160]} for the waveform
raw_text = description + the opening of the caption track, when there is one. No thumbnail: sounds
hang as listening cards, not pictures. Colour comes from a frame of the source video.

Any video already ingested as a clip (data/raw/nasa_video.parquet) is skipped, so nothing appears twice.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess

import numpy as np
import pandas as pd
from PIL import Image

from gallery.ingest import _common as C
from gallery.ingest.nasa import API, LICENSE, creator_of, dedup, described_first, safe_name
from gallery.ingest.nasa_video import (CACHE_DIR, CATALOG_TITLE, duration_s, ffmpeg, manifest, meta_duration,
                                       vtt_text)
from gallery.schema import DATA, validate

log = logging.getLogger("gallery.ingest.nasa_audio")

SOURCE = "nasa"
# topic -> search phrases; the topic is what the sound is of
QUERIES = {
    "sounds of mars": ["sounds from mars", "perseverance microphone", "open mic on mars", "ingenuity helicopter sound"],
    "marsquakes": ["marsquake", "meteoroid striking mars", "insight seismometer sound"],
    "sonification": ["sonification", "sounds from around the milky way", "black hole sound", "listening to the universe"],
    "space plasma sounds": ["juno sounds", "voyager plasma sound", "chorus waves", "earth magnetic field sound", "sounds of the sun"],
    "launches": ["saturn v launch enhanced sound", "space shuttle launch sound", "artemis launch", "rocket launch sound"],
    "engine tests": ["rs-25 engine test", "hot fire test", "solid rocket booster test", "water deluge test"],
    "aircraft": ["sonic boom", "x-59 supersonic", "acoustic testing"],
}
TARGET = 40
PAGE_SIZE = 100
MAX_S = 330            # what the UI plays
EMBED_S = 300          # what the model hears (its context holds ~327 s of audio)
N_PEAKS = 160
DESC_CHARS = 1200
CAPTION_CHARS = 700
MIN_RMS_DB = -45.0     # quieter than this throughout = effectively silent
# Interviews, explainers and narration are welcome (their speech becomes the karaoke transcript);
# only drop what has no usable sound or is a government proceeding.
EXCLUDE = re.compile(
    r"\b(no sound|silent|hearing|testif|podcast|live coverage|news conference|briefing|"
    r"media reel|nomination|confirmation|webinar|town hall)\b", re.I)
AUDIO_DIR = DATA / "media" / "nasa_audio"

api_limiter = C.RateLimiter(1.0)
asset_limiter = C.RateLimiter(0.25)


def search(q: str) -> list[dict]:
    params = {"q": q, "media_type": "video", "page_size": PAGE_SIZE}
    data = C.cached_json(SOURCE, f"audio_search_{C.cache_key(q, PAGE_SIZE)}",
                         lambda: C.fetch(API, params=params, limiter=api_limiter).json())
    out = []
    for item in data["collection"]["items"]:
        d = (item.get("data") or [{}])[0]
        title, desc = (d.get("title") or "").strip(), d.get("description") or ""
        if not d.get("nasa_id") or not title or EXCLUDE.search(title) or EXCLUDE.search(desc[:300]) or CATALOG_TITLE.match(title):
            continue
        out.append({"nasa_id": d["nasa_id"], "title": title, "description": desc, "center": d.get("center") or "",
                    "photographer": d.get("photographer") or "", "date_created": d.get("date_created") or "",
                    "manifest": item.get("href") or ""})
    return out


def candidates() -> dict[str, list[dict]]:
    raw = {topic: [h for q in qs for h in search(q)] for topic, qs in QUERIES.items()}
    pools = {k: described_first(v) for k, v in dedup(raw).items()}
    for k in QUERIES:
        log.info("%-20s candidates %4d", k, len(pools[k]))
    return pools


def peaks_and_level(wav) -> tuple[list[int], float]:
    """Waveform envelope (N_PEAKS bins, 0-255) and the loudest bin's RMS in dBFS, from a 16 kHz mono WAV."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-f", "s16le", "-ac", "1", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if x.size == 0:
        raise ValueError("no audio samples")
    bins = np.array_split(x, N_PEAKS)
    rms = np.array([np.sqrt(np.mean(b * b)) if b.size else 0.0 for b in bins])
    loud_db = float(20 * np.log10(max(rms.max(), 1e-9)))
    peaks = (255 * rms / max(rms.max(), 1e-9)).round().astype(int).tolist()
    return peaks, loud_db


def has_audio_stream(path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return "audio" in out


def fetch_one(topic: str, it: dict) -> dict | None:
    sid = safe_name(it["nasa_id"])
    src = CACHE_DIR / f"{sid}.mp4"
    play = DATA / "media" / SOURCE / f"{sid}.m4a"
    wav, side = AUDIO_DIR / f"{sid}.wav", AUDIO_DIR / f"{sid}.json"
    frame = AUDIO_DIR / f"{sid}.jpg"
    try:
        files = manifest(it)
        mp4 = next((f for f in files if f.endswith("~preview.mp4")), None) or next(
            (f for f in files if f.endswith("~small.mp4")), None)
        if not mp4:
            raise ValueError("no ~preview/~small mp4")
        meta_url = next((f for f in files if f.endswith("metadata.json")), None)
        if meta_url and not src.is_file():
            meta = C.cached_json(SOURCE, f"video_meta_{C.cache_key(it['nasa_id'])}", lambda: C.fetch(
                meta_url.replace("http://", "https://").replace(" ", "%20"), limiter=api_limiter).json())
            listed = meta_duration(meta)
            if listed is not None and listed > MAX_S:
                raise ValueError(f"listed duration {listed:.0f}s > {MAX_S}s")
        if not src.is_file():
            src.parent.mkdir(parents=True, exist_ok=True)
            r = C.fetch(mp4.replace("http://", "https://").replace(" ", "%20"), limiter=asset_limiter, timeout=180)
            tmp = src.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(src)
        dur = duration_s(src)
        if not (8 <= dur <= MAX_S):
            raise ValueError(f"duration {dur:.0f}s outside 8-{MAX_S}s")
        if not has_audio_stream(src):
            raise ValueError("no audio stream")
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        play.parent.mkdir(parents=True, exist_ok=True)
        if not wav.is_file():
            ffmpeg("-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-t", str(EMBED_S), str(wav))
        peaks, loud_db = peaks_and_level(wav)
        if loud_db < MIN_RMS_DB:
            raise ValueError(f"effectively silent (loudest {loud_db:.0f} dBFS)")
        if not play.is_file():
            ffmpeg("-i", str(src), "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(play))
        side.write_text(json.dumps({"duration": round(dur, 2), "peaks": peaks}))
        if not frame.is_file():
            ffmpeg("-ss", f"{dur * 0.3:.2f}", "-i", str(src), "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "4", str(frame))
        color = C.dominant_color(Image.open(frame))
        captions = ""
        vtt = next((f for f in files if f.endswith(".vtt")), None)
        if vtt:
            try:
                captions = vtt_text(C.fetch(vtt.replace("http://", "https://").replace(" ", "%20"),
                                            limiter=asset_limiter).text)
            except Exception as e:  # noqa: BLE001 - captions are optional; the miss is logged
                C.log_failure(SOURCE, f"nasa:{it['nasa_id']}", "captions", e)
    except Exception as e:  # noqa: BLE001 - every per-item failure is logged and the item dropped
        C.log_failure(SOURCE, f"nasa:{it['nasa_id']}", "audio", e)
        for p in (play, wav, side):
            p.unlink(missing_ok=True)
        return None
    desc = C.truncate_words(C.strip_html(it["description"]), DESC_CHARS)
    text = desc or it["title"]
    if captions:
        text += " Narration: " + C.truncate_words(captions, CAPTION_CHARS)
    return {
        "id": f"nasa:{it['nasa_id']}", "source": SOURCE, "modality": "audio", "title": it["title"],
        "raw_text": text, "media_path": C.rel(play), "thumb_path": "",
        "source_url": f"https://images.nasa.gov/details/{it['nasa_id']}", "license": LICENSE,
        "color_hex": color, "topic": topic, "creator": creator_of(it), "date": it["date_created"][:10],
    }


def audio_paths(row_id: str):
    """(16 kHz wav for the model, sidecar json) for an audio row (shared with embedding, curation, assets)."""
    sid = safe_name(row_id.split(":", 1)[1])
    return AUDIO_DIR / f"{sid}.wav", AUDIO_DIR / f"{sid}.json"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args(argv)
    C.setup_logging()
    pools = candidates()
    taken = set(pd.read_parquet(DATA / "raw" / "nasa.parquet")["id"])
    vids = DATA / "raw" / "nasa_video.parquet"
    if vids.is_file():
        taken |= set(pd.read_parquet(vids)["id"])   # one item per source video: never a clip and a sound
    pools = {k: [it for it in v if f"nasa:{it['nasa_id']}" not in taken] for k, v in pools.items()}
    cap = max(1, int(args.target * 0.25))
    rows = C.select_and_fetch(pools, args.target, cap, fetch_one, workers=3)
    df = pd.DataFrame(rows)
    validate(df)
    out = C.write_parquet(df, "nasa_audio")
    log.info("wrote %s: %d rows", out, len(df))
    print(df.groupby("topic")["title"].apply(lambda s: list(s.str[:50])).to_string())


if __name__ == "__main__":
    main()
