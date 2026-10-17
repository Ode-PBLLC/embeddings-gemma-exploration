"""Ingest ~1,700 images from the NASA Image and Video Library across a fixed keyword list.

uv run python -m gallery.ingest.nasa

Selection: keyword-balanced (water-filling, cap 15% per keyword), deduplicated by
nasa_id and by exact (case-folded) title across all keywords; within a keyword, items
with a description come first (search relevance order otherwise kept). Colour is the dominant
colour of the thumbnail from k-means (k=5).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from PIL import Image

from gallery.ingest import _common as C
from gallery.schema import validate

log = logging.getLogger("gallery.ingest.nasa")

SOURCE = "nasa"
API = "https://images-api.nasa.gov/search"
KEYWORDS = [
    "nebula",
    "galaxy",
    "earth from orbit",
    "mars surface",
    "astronaut spacewalk",
    "rocket launch",
    "jupiter",
    "aurora",
    "moon",
    "saturn",
    "comet",
    "international space station",
    "hurricane from space",
    "solar flare",
    "apollo",
    "hubble",
]
TARGET = 1700
CAP_SHARE = 0.15
PAGE_SIZE = 100
MAX_PAGES = 5
POOL_PER_KEYWORD = 220
DESC_CHARS = 1200
LICENSE = "NASA media (generally public domain)"

api_limiter = C.RateLimiter(1.0)
asset_limiter = C.RateLimiter(0.25)


def search_page(keyword: str, page: int) -> dict:
    params = {"q": keyword, "media_type": "image", "page": page, "page_size": PAGE_SIZE}
    name = f"search_{C.cache_key(keyword, page, PAGE_SIZE)}"
    return C.cached_json(SOURCE, name, lambda: C.fetch(API, params=params, limiter=api_limiter).json())


def parse_item(item: dict) -> dict | None:
    """Flatten one search hit; None if it is not a usable image."""
    data = (item.get("data") or [{}])[0]
    if data.get("media_type") != "image" or not data.get("nasa_id") or not (data.get("title") or "").strip():
        return None
    links = {}
    for ln in item.get("links") or []:
        href = ln.get("href") or ""
        for variant in ("~medium.", "~small.", "~thumb."):
            if variant in href:
                links[variant.strip("~.")] = href
        if ln.get("rel") == "preview":
            links.setdefault("preview", href)
    return {
        "nasa_id": data["nasa_id"],
        "title": data["title"].strip(),
        "description": data.get("description") or data.get("description_508") or "",
        "center": data.get("center") or "",
        "photographer": data.get("photographer") or "",
        "date_created": data.get("date_created") or "",
        "manifest": item.get("href") or "",
        "links": links,
    }


def norm_title(t: str) -> str:
    return " ".join(t.casefold().split())


def dedup(pools: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Drop repeats by nasa_id and by exact title. Keywords are visited round-robin so
    an item shared by two keywords goes to the one where it ranks higher."""
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    out: dict[str, list[dict]] = {k: [] for k in pools}
    depth = max((len(v) for v in pools.values()), default=0)
    for i in range(depth):
        for k, items in pools.items():
            if i >= len(items):
                continue
            it = items[i]
            t = norm_title(it["title"])
            if it["nasa_id"] in seen_ids or t in seen_titles:
                continue
            seen_ids.add(it["nasa_id"])
            seen_titles.add(t)
            out[k].append(it)
    return out


def has_description(it: dict) -> bool:
    d = C.strip_html(it["description"])
    return bool(d) and norm_title(d) != norm_title(it["title"])


def described_first(items: list[dict]) -> list[dict]:
    """Stable reorder: items with a real description first (they give the models text)."""
    return sorted(items, key=lambda it: not has_description(it))


def candidates() -> dict[str, list[dict]]:
    raw: dict[str, list[dict]] = {}
    for kw in KEYWORDS:
        items: list[dict] = []
        for page in range(1, MAX_PAGES + 1):
            data = search_page(kw, page)
            hits = data["collection"]["items"]
            items += [p for p in (parse_item(h) for h in hits) if p]
            if len(items) >= POOL_PER_KEYWORD * 1.6 or len(hits) < PAGE_SIZE:
                break
        raw[kw] = items
    pools = {k: described_first(v) for k, v in dedup(raw).items()}
    for kw in KEYWORDS:
        log.info("%-30s raw %4d  after dedup %4d", kw, len(raw[kw]), len(pools[kw]))
    return pools


def image_url(it: dict) -> str:
    """Prefer ~medium, then ~small, then the preview/thumb; fall back to the asset manifest."""
    for k in ("medium", "small", "preview", "thumb"):
        if it["links"].get(k):
            return it["links"][k]
    if it["manifest"]:
        files = C.fetch(it["manifest"], limiter=api_limiter).json()
        for variant in ("~medium.jpg", "~small.jpg", "~thumb.jpg"):
            for f in files:
                if f.endswith(variant):
                    return f
    raise ValueError("no image variant available")


def creator_of(it: dict) -> str:
    return it["photographer"] or it["center"]


def to_row(it: dict, keyword: str, media_rel: str, thumb_rel: str, color: str) -> dict:
    desc = C.truncate_words(C.strip_html(it["description"]), DESC_CHARS)
    return {
        "id": f"nasa:{it['nasa_id']}",
        "source": SOURCE,
        "modality": "image",
        "title": it["title"],
        "raw_text": desc or it["title"],
        "media_path": media_rel,
        "thumb_path": thumb_rel,
        "source_url": f"https://images.nasa.gov/details/{it['nasa_id']}",
        "license": LICENSE,
        "color_hex": color,
        "topic": keyword,
        "creator": creator_of(it),
        "date": it["date_created"][:10],
    }


def safe_name(nasa_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in nasa_id)


def fetch_one(keyword: str, it: dict) -> dict | None:
    media, thumb = C.media_paths(SOURCE, safe_name(it["nasa_id"]))
    try:
        if media.is_file() and thumb.is_file():
            timg = Image.open(thumb)
        else:
            url = image_url(it).replace("http://", "https://")
            r = C.fetch(url.replace(" ", "%20"), limiter=asset_limiter)
            timg = C.save_resized(r.content, media, thumb)
        color = C.dominant_color(timg)
    except Exception as e:  # noqa: BLE001 - every per-item failure is logged and the item dropped
        C.log_failure(SOURCE, f"nasa:{it['nasa_id']}", "image", e)
        for p in (media, thumb):
            p.unlink(missing_ok=True)
        return None
    return to_row(it, keyword, C.rel(media), C.rel(thumb), color)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args(argv)
    C.setup_logging()
    pools = candidates()
    cap = int(args.target * CAP_SHARE * 0.9)
    rows = C.select_and_fetch(pools, args.target, cap, fetch_one, workers=4)
    df = pd.DataFrame(rows)
    validate(df)
    out = C.write_parquet(df, SOURCE)
    log.info("wrote %s: %d rows", out, len(df))
    print(df["topic"].value_counts().to_string())


if __name__ == "__main__":
    main()
