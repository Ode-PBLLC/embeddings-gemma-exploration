"""Ingest ~240 short NASA videos (visualizations and footage, not talks) from the NASA Image and Video Library.

uv run python -m gallery.ingest.nasa_video [--target 240] [--limit N]

Per clip, from the full-length ~preview.mp4 (480 px):
  data/media/nasa_video/<id>/fNN.jpg   16 frames sampled evenly (what EmbeddingGemma and Gemma see)
  data/media/nasa/<id>.mp4             an 8 s silent loop from 30% in (what the UI plays)   = media_path
  data/thumbs/nasa/<id>.jpg            a still from 30% in, <=320 px                         = thumb_path
raw_text = description + the opening of the caption track (.vtt), when there is one.

Selection: keyword-balanced like the image ingester; drops talks, briefings and interviews by
title/description keywords, and anything shorter than 10 s or longer than 6 min.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess

import pandas as pd
from PIL import Image

from gallery.ingest import _common as C
from gallery.ingest.nasa import API, LICENSE, creator_of, dedup, described_first, safe_name
from gallery.schema import DATA, validate

log = logging.getLogger("gallery.ingest.nasa_video")

SOURCE = "nasa"
KEYWORDS = [
    "aurora", "solar flare", "sun", "earth from orbit", "hurricane", "nebula", "galaxy", "black hole",
    "mars", "jupiter", "saturn", "moon", "rocket launch", "space station", "comet", "supernova",
    "ocean", "sea ice", "wildfire", "volcano",
]
TARGET = 240
CAP_SHARE = 0.10
PAGE_SIZE = 100
MAX_PAGES = 3
N_FRAMES = 16
LOOP_SECONDS = 8
MIN_S, MAX_S = 10, 360
DESC_CHARS = 1200
CAPTION_CHARS = 700
# talks, briefings, interviews and other people-at-a-podium formats look dull on a gallery wall
EXCLUDE = re.compile(
    r"\b(live|briefing|interview|expert|asked|ask an?|conference|talk|webinar|q ?& ?a|podcast|news|update|"
    r"explains?|explained|this week|town hall|panel|lecture|tour|behind the scenes|event|ceremony|media|"
    r"broadcast|trailer|promo|reel|hangout|chat|discussion|meet|minute|answers?|questions?|science live|tonight|"
    r"what'?s up|how to|interview|speech|remarks|testimony|announcement|preview of)\b", re.I)

# stock-footage titles that are only a catalog number (e.g. "KSC-04-S-00193"): nothing to hang a label on
CATALOG_TITLE = re.compile(r"^\s*[A-Z]{2,6}[-_ ]?\d[\w-]*\s*$")

CACHE_DIR = DATA / "raw" / "video_cache"
FRAMES_DIR = DATA / "media" / "nasa_video"

api_limiter = C.RateLimiter(1.0)
asset_limiter = C.RateLimiter(0.25)


def search_page(keyword: str, page: int) -> dict:
    params = {"q": keyword, "media_type": "video", "page": page, "page_size": PAGE_SIZE}
    name = f"video_search_{C.cache_key(keyword, page, PAGE_SIZE)}"
    return C.cached_json(SOURCE, name, lambda: C.fetch(API, params=params, limiter=api_limiter).json())


def parse_item(item: dict) -> dict | None:
    data = (item.get("data") or [{}])[0]
    if data.get("media_type") != "video" or not data.get("nasa_id") or not (data.get("title") or "").strip():
        return None
    title, desc = data["title"].strip(), data.get("description") or data.get("description_508") or ""
    if EXCLUDE.search(title) or EXCLUDE.search(desc[:300]) or CATALOG_TITLE.match(title):
        return None
    thumb = next((ln.get("href") for ln in item.get("links") or [] if ln.get("rel") == "preview"), "")
    return {
        "nasa_id": data["nasa_id"], "title": title, "description": desc,
        "center": data.get("center") or "", "photographer": data.get("photographer") or "",
        "date_created": data.get("date_created") or "", "manifest": item.get("href") or "", "thumb": thumb,
    }


def candidates() -> dict[str, list[dict]]:
    raw: dict[str, list[dict]] = {}
    for kw in KEYWORDS:
        items: list[dict] = []
        for page in range(1, MAX_PAGES + 1):
            hits = search_page(kw, page)["collection"]["items"]
            items += [p for p in (parse_item(h) for h in hits) if p]
            if len(hits) < PAGE_SIZE:
                break
        raw[kw] = items
    pools = {k: described_first(v) for k, v in dedup(raw).items()}
    for kw in KEYWORDS:
        log.info("%-20s usable %4d  after dedup %4d", kw, len(raw[kw]), len(pools[kw]))
    return pools


def manifest(it: dict) -> list[str]:
    name = f"video_manifest_{C.cache_key(it['nasa_id'])}"
    url = it["manifest"].replace("http://", "https://").replace(" ", "%20")
    return C.cached_json(SOURCE, name, lambda: {"files": C.fetch(url, limiter=api_limiter).json()})["files"]


def meta_duration(meta: dict) -> float | None:
    """Seconds from a NASA metadata.json ("0:02:12" or "12.3 s"), or None if it is not listed."""
    v = str(meta.get("QuickTime:Duration") or meta.get("QuickTime:TrackDuration") or "").strip()
    if not v:
        return None
    if ":" in v:
        parts = [float(x) for x in v.split(":")]
        return sum(x * 60 ** i for i, x in enumerate(reversed(parts)))
    m = re.match(r"([\d.]+)\s*s", v)
    return float(m.group(1)) if m else None


def duration_s(path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def ffmpeg(*args: str) -> None:
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[:300]}")


def vtt_text(raw: str) -> str:
    """Spoken/caption text from a WebVTT file, cues joined, timings and tags dropped."""
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")) or "-->" in ln or ln.isdigit():
            continue
        ln = re.sub(r"<[^>]+>", "", ln)
        if not lines or lines[-1] != ln:
            lines.append(ln)
    return " ".join(lines)


def fetch_one(keyword: str, it: dict) -> dict | None:
    sid = safe_name(it["nasa_id"])
    src = CACHE_DIR / f"{sid}.mp4"
    frames = FRAMES_DIR / sid
    clip, thumb = DATA / "media" / SOURCE / f"{sid}.mp4", DATA / "thumbs" / SOURCE / f"{sid}.jpg"
    try:
        files = manifest(it)
        mp4 = next((f for f in files if f.endswith("~preview.mp4")), None) or next(
            (f for f in files if f.endswith("~small.mp4")), None)
        if not mp4:
            raise ValueError("no ~preview/~small mp4")
        # check the length from NASA's metadata before downloading (some entries are hours-long broadcasts)
        meta_url = next((f for f in files if f.endswith("metadata.json")), None)
        if meta_url and not src.is_file():
            meta = C.cached_json(SOURCE, f"video_meta_{C.cache_key(it['nasa_id'])}", lambda: C.fetch(
                meta_url.replace("http://", "https://").replace(" ", "%20"), limiter=api_limiter).json())
            listed = meta_duration(meta)
            if listed is not None and not (MIN_S <= listed <= MAX_S):
                raise ValueError(f"listed duration {listed:.0f}s outside {MIN_S}-{MAX_S}s")
        if not src.is_file():
            src.parent.mkdir(parents=True, exist_ok=True)
            r = C.fetch(mp4.replace("http://", "https://").replace(" ", "%20"), limiter=asset_limiter, timeout=180)
            tmp = src.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(src)
        dur = duration_s(src)
        if not (MIN_S <= dur <= MAX_S):
            raise ValueError(f"duration {dur:.0f}s outside {MIN_S}-{MAX_S}s")
        frames.mkdir(parents=True, exist_ok=True)
        for k in range(N_FRAMES):
            t = dur * (k + 0.5) / N_FRAMES
            f = frames / f"f{k:02d}.jpg"
            if not f.is_file():
                ffmpeg("-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "3", str(f))
        start = max(0.0, min(dur * 0.3, dur - LOOP_SECONDS))
        clip.parent.mkdir(parents=True, exist_ok=True)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        if not clip.is_file():
            ffmpeg("-ss", f"{start:.2f}", "-i", str(src), "-t", str(LOOP_SECONDS), "-an", "-vf", "scale=480:-2",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(clip))
        if not thumb.is_file():
            ffmpeg("-ss", f"{start + 1:.2f}", "-i", str(src), "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "3", str(thumb))
        timg = Image.open(thumb)
        color = C.dominant_color(timg)
        captions = ""
        vtt = next((f for f in files if f.endswith(".vtt")), None)
        if vtt:
            try:
                captions = vtt_text(C.fetch(vtt.replace("http://", "https://").replace(" ", "%20"),
                                            limiter=asset_limiter).text)
            except Exception as e:  # noqa: BLE001 - captions are optional; the miss is logged
                C.log_failure(SOURCE, f"nasa:{it['nasa_id']}", "captions", e)
    except Exception as e:  # noqa: BLE001 - every per-item failure is logged and the item dropped
        C.log_failure(SOURCE, f"nasa:{it['nasa_id']}", "video", e)
        for p in (clip, thumb):
            p.unlink(missing_ok=True)
        return None
    desc = C.truncate_words(C.strip_html(it["description"]), DESC_CHARS)
    text = desc or it["title"]
    if captions:
        text += " Narration: " + C.truncate_words(captions, CAPTION_CHARS)
    return {
        "id": f"nasa:{it['nasa_id']}", "source": SOURCE, "modality": "video", "title": it["title"],
        "raw_text": text, "media_path": C.rel(clip), "thumb_path": C.rel(thumb),
        "source_url": f"https://images.nasa.gov/details/{it['nasa_id']}", "license": LICENSE,
        "color_hex": color, "topic": keyword, "creator": creator_of(it), "date": it["date_created"][:10],
    }


def frames_dir(row_id: str):
    """Where a video row's sampled frames live (shared with the embedding and curation steps)."""
    return FRAMES_DIR / safe_name(row_id.split(":", 1)[1])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args(argv)
    C.setup_logging()
    pools = candidates()
    # never repeat an id the image ingest already holds
    images = pd.read_parquet(DATA / "raw" / "nasa.parquet")
    taken = set(images["id"])
    pools = {k: [it for it in v if f"nasa:{it['nasa_id']}" not in taken] for k, v in pools.items()}
    cap = max(1, int(args.target * CAP_SHARE))
    rows = C.select_and_fetch(pools, args.target, cap, fetch_one, workers=3)
    df = pd.DataFrame(rows)
    validate(df)
    out = C.write_parquet(df, "nasa_video")
    log.info("wrote %s: %d rows", out, len(df))
    print(df["topic"].value_counts().to_string())


if __name__ == "__main__":
    main()
