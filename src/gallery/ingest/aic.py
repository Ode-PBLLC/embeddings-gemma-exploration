"""Ingest ~1,700 public-domain artworks from the Art Institute of Chicago API.

uv run python -m gallery.ingest.aic

Selection: department-balanced (water-filling quotas, so no department exceeds 25%);
within a department, works with a description come first, in the API's default
relevance order. Images come from AIC's IIIF server, which answers 403 (a Cloudflare
challenge) unless the request carries an `AIC-User-Agent` header, so we send one.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from gallery.ingest import _common as C
from gallery.schema import validate

log = logging.getLogger("gallery.ingest.aic")

SOURCE = "aic"
API = "https://api.artic.edu/api/v1/artworks/search"
IIIF = "https://www.artic.edu/iiif/2/{image_id}/full/{size}/0/default.jpg"
IIIF_WIDTH = 843  # AIC's recommended width; the server refuses upscaling (403)
FIELDS = [
    "id", "title", "artist_display", "artist_title", "date_display", "description",
    "short_description", "department_title", "style_title", "color", "image_id",
    "is_public_domain", "thumbnail", "medium_display",
]
HEADERS = {"AIC-User-Agent": C.USER_AGENT}
TARGET = 1700
CAP_SHARE = 0.25
PAGE = 100
MAX_PAGES_PER_TIER = 3

api_limiter = C.RateLimiter(1.0)   # <= 1 request/s on the JSON API
iiif_limiter = C.RateLimiter(0.25)  # 4 workers, <= 4 requests/s on IIIF


def _search(body: dict) -> dict:
    return C.fetch(API, json_body=body, headers=HEADERS, limiter=api_limiter).json()


def _base_filter(extra: list | None = None) -> list:
    return [{"term": {"is_public_domain": True}}, {"exists": {"field": "image_id"}}] + (extra or [])


def departments() -> dict[str, int]:
    body = {
        "query": {"bool": {"filter": _base_filter()}},
        "limit": 0,
        "aggs": {"dept": {"terms": {"field": "department_title.keyword", "size": 50}}},
    }
    data = C.cached_json(SOURCE, "departments", lambda: _search(body))
    return {b["key"]: b["doc_count"] for b in data["aggregations"]["dept"]["buckets"]}


def _tier(dept: str, with_desc: bool, need: int) -> list[dict]:
    """Up to `need` records for one department, with or without a description."""
    cond = {"exists": {"field": "description"}}
    q = {"bool": {"filter": _base_filter([{"term": {"department_title.keyword": dept}}])}}
    if with_desc:
        q["bool"]["filter"].append(cond)
    else:
        q["bool"]["must_not"] = [cond]
    out: list[dict] = []
    for page in range(1, MAX_PAGES_PER_TIER + 1):
        body = {"query": q, "fields": FIELDS, "limit": PAGE, "page": page}
        name = f"search_{C.cache_key(dept, with_desc, page)}"
        data = C.cached_json(SOURCE, name, lambda: _search(body))
        out += data["data"]
        if len(out) >= need or len(data["data"]) < PAGE:
            break
    return out


def keep(rec: dict) -> bool:
    return bool(rec.get("is_public_domain")) and bool(rec.get("image_id"))


def candidates(target: int) -> dict[str, list[dict]]:
    depts = departments()
    cap = int(target * CAP_SHARE * 0.9)
    # ask for ~1.3x the quota per department so failures can be topped up
    quota = C.allocate({d: min(n, cap) for d, n in depts.items()}, int(target * 1.03))
    pools: dict[str, list[dict]] = {}
    for dept, q in quota.items():
        if q == 0:
            continue
        want = int(q * 1.3) + 5
        recs = _tier(dept, True, want)
        if len(recs) < want:
            recs += _tier(dept, False, want - len(recs))
        seen, pool = set(), []
        for r in recs:
            if keep(r) and r["id"] not in seen:
                seen.add(r["id"])
                pool.append(r)
        pools[dept] = pool
        log.info("%-45s quota %4d  candidates %4d", dept, q, len(pool))
    return pools


def compose_text(rec: dict) -> str:
    """raw_text: description (HTML stripped) -> short_description -> composed sentence."""
    desc = C.strip_html(rec.get("description"))
    if desc:
        return desc
    short = C.strip_html(rec.get("short_description"))
    if short:
        return short
    title = (rec.get("title") or "Untitled").strip()
    artist = creator_of(rec)
    parts = [title]
    if artist:
        parts.append(f"by {artist}")
    if rec.get("date_display"):
        parts.append(f"({rec['date_display']})")
    text = " ".join(parts) + "."
    if rec.get("medium_display"):
        text += f" {rec['medium_display'].strip().rstrip('.')}."
    alt = ((rec.get("thumbnail") or {}).get("alt_text") or "").strip()
    if alt:
        text += f" {alt.rstrip('.')}."
    return text


def creator_of(rec: dict) -> str:
    return (rec.get("artist_display") or "").split("\n")[0].strip()


def color_of(rec: dict) -> str | None:
    c = rec.get("color")
    if not c or c.get("h") is None or c.get("s") is None or c.get("l") is None:
        return None
    return C.hsl_to_hex(c["h"], c["s"], c["l"])


def to_row(rec: dict, media_rel: str, thumb_rel: str, color_hex: str) -> dict:
    return {
        "id": f"aic:{rec['id']}",
        "source": SOURCE,
        "modality": "image",
        "title": (rec.get("title") or "Untitled").strip() or "Untitled",
        "raw_text": compose_text(rec),
        "media_path": media_rel,
        "thumb_path": thumb_rel,
        "source_url": f"https://www.artic.edu/artworks/{rec['id']}",
        "license": "CC0 (image); CC BY 4.0 (text)",
        "color_hex": color_hex,
        "topic": rec.get("department_title") or "",
        "creator": creator_of(rec),
        "date": rec.get("date_display") or "",
    }


def iiif_url(rec: dict) -> str:
    """843px wide, or the original width when narrower: the server refuses upscaling, and
    `full`/`max` size requests get a Cloudflare challenge, so ask for an explicit width."""
    width = (rec.get("thumbnail") or {}).get("width")
    size = f"{width}," if width and width < IIIF_WIDTH else f"{IIIF_WIDTH},"
    return IIIF.format(image_id=rec["image_id"], size=size)


def fetch_one(dept: str, rec: dict) -> dict | None:
    native = str(rec["id"])
    media, thumb = C.media_paths(SOURCE, native)
    try:
        if media.is_file() and thumb.is_file():
            from PIL import Image

            timg = Image.open(thumb)
        else:
            r = C.fetch(iiif_url(rec), headers=HEADERS, limiter=iiif_limiter)
            if not r.headers.get("content-type", "").startswith("image/"):
                raise ValueError(f"not an image: {r.headers.get('content-type')}")
            timg = C.save_resized(r.content, media, thumb)
        color = color_of(rec) or C.dominant_color(timg)
    except Exception as e:  # noqa: BLE001 - every per-item failure is logged and the item dropped
        log_id = f"aic:{native}"
        C.log_failure(SOURCE, log_id, "image", e)
        for p in (media, thumb):
            p.unlink(missing_ok=True)
        return None
    return to_row(rec, C.rel(media), C.rel(thumb), color)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args(argv)
    C.setup_logging()
    pools = candidates(args.target)
    cap = int(args.target * CAP_SHARE * 0.9)
    rows = C.select_and_fetch(pools, args.target, cap, fetch_one, workers=4)
    df = pd.DataFrame(rows)
    validate(df)
    out = C.write_parquet(df, SOURCE)
    # entity list for the wiki ingester (artist + style names of the kept works)
    kept = {r["id"] for r in rows}
    ents = [
        {"id": f"aic:{r['id']}", "artist_title": r.get("artist_title"), "style_title": r.get("style_title")}
        for pool in pools.values() for r in pool if f"aic:{r['id']}" in kept
    ]
    (C.RAW / "aic_entities.json").write_text(json.dumps(ents, ensure_ascii=False, indent=0), encoding="utf-8")
    log.info("wrote %s: %d rows", out, len(df))
    print(df["topic"].value_counts().to_string())


if __name__ == "__main__":
    main()
