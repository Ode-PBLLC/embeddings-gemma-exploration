"""Shared helpers for the ingesters: polite HTTP with retries, JSON cache, failure log,
image resize, HTML stripping, colour maths."""

from __future__ import annotations

import colorsys
import hashlib
import html
import io
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import requests
from PIL import Image

from gallery.schema import DATA, ROOT

log = logging.getLogger("gallery.ingest")

USER_AGENT = "embedding-gallery demo/0.1 (internal Ode demo; contact mason@ode.partners)"
CACHE = DATA / "raw" / "cache"
RAW = DATA / "raw"
FAILURES = ROOT / "reports" / "ingest_failures.jsonl"
MEDIA_PX = 768
THUMB_PX = 320
RETRIES = 3  # retries after the first attempt

_local = threading.local()
_fail_lock = threading.Lock()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def session(extra_headers: dict | None = None) -> requests.Session:
    """One requests.Session per thread (Session is not guaranteed thread-safe)."""
    key = json.dumps(extra_headers or {}, sort_keys=True)
    cache = getattr(_local, "sessions", None)
    if cache is None:
        cache = _local.sessions = {}
    if key not in cache:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        s.headers.update(extra_headers or {})
        cache[key] = s
    return cache[key]


class RateLimiter:
    """At most one call per `interval` seconds across threads."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


def fetch(
    url: str,
    *,
    params: dict | None = None,
    json_body: dict | None = None,
    headers: dict | None = None,
    limiter: RateLimiter | None = None,
    timeout: float = 60,
) -> requests.Response:
    """GET (or POST when json_body is given) with up to RETRIES retries and backoff.
    Raises the last error if every attempt fails. 404 is not retried."""
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        if limiter:
            limiter.wait()
        try:
            s = session(headers)
            if json_body is not None:
                r = s.post(url, params=params, json=json_body, timeout=timeout)
            else:
                r = s.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                raise requests.HTTPError(f"HTTP {r.status_code} for {r.url}", response=r)
            return r
        except requests.HTTPError as e:
            last = e
            if e.response is not None and e.response.status_code == 404:
                raise
        except requests.RequestException as e:
            last = e
        if attempt < RETRIES:
            time.sleep(2**attempt)
    assert last is not None
    raise last


def cache_key(*parts: object) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cached_json(source: str, name: str, producer: Callable[[], dict]) -> dict:
    """Return cached JSON at data/raw/cache/<source>/<name>.json, producing it on a miss."""
    path = CACHE / source / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    data = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return data


def log_failure(source: str, item_id: str, stage: str, error: object) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "id": item_id,
        "stage": stage,
        "error": str(error)[:500],
    }
    FAILURES.parent.mkdir(parents=True, exist_ok=True)
    with _fail_lock, FAILURES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.warning("failure %s %s %s: %s", source, item_id, stage, rec["error"])


# ---------------------------------------------------------------- text

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"(?i)<br\s*/?>|</p>", " ", text)
    text = _TAG.sub("", text)
    return _WS.sub(" ", html.unescape(text)).strip()


def truncate_words(text: str, max_chars: int) -> str:
    """Cut at a sentence end if one lies in the last 40% of the window, else at a word."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end >= int(max_chars * 0.6):
        return cut[: end + 1]
    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "…"


# ---------------------------------------------------------------- colour

def hsl_to_hex(h: float, s: float, l: float) -> str:
    """AIC colour: h in 0-360, s and l in 0-100."""
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, min(max(l, 0), 100) / 100.0, min(max(s, 0), 100) / 100.0)
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in (r, g, b)))


def rgb_to_hex(rgb: Iterable[float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(min(max(c, 0), 255))) for c in rgb))


def _is_extreme(rgb: np.ndarray) -> bool:
    """Near-black or near-white (low-saturation light) cluster centre."""
    r, g, b = (rgb / 255.0).tolist()
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return l < 0.12 or (l > 0.88 and s < 0.5) or max(r, g, b) < 0.15


def _saturation(rgb: np.ndarray) -> float:
    r, g, b = (rgb / 255.0).tolist()
    mx, mn = max(r, g, b), min(r, g, b)
    return 0.0 if mx == 0 else (mx - mn) / mx  # HSV saturation: vivid colours score high


def pick_dominant(centers: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Choose a display colour from k-means centres (RGB 0-255) and their pixel shares.

    Near-black / near-white clusters are skipped when another cluster holds >=10% of
    pixels. Among the remaining clusters with >=10% share, prefer saturated ones
    (score = saturation * sqrt(share)) so the cloud gets pops of colour. If no
    non-extreme cluster reaches 10%, a non-extreme accent cluster with >=3% is used."""
    shares = np.asarray(shares, dtype=float)
    shares = shares / shares.sum()
    idx = list(range(len(centers)))
    ok = [i for i in idx if not _is_extreme(centers[i]) and shares[i] >= 0.10]
    if not ok:
        # Space images are often >85% black. Rather than paint them all black, take a
        # visible accent cluster (>=3% of pixels) if there is one; else the largest.
        ok = [i for i in idx if not _is_extreme(centers[i]) and shares[i] >= 0.03]
    if not ok:
        return centers[int(np.argmax(shares))]
    return centers[max(ok, key=lambda i: (_saturation(centers[i]) + 0.05) * np.sqrt(shares[i]))]


def dominant_color(img: Image.Image, k: int = 5, seed: int = 0) -> str:
    """Dominant colour of an image via k-means (k=5) on up to 64x64 pixels."""
    from sklearn.cluster import KMeans

    small = img.convert("RGB").copy()
    small.thumbnail((64, 64))
    px = np.asarray(small, dtype=np.float64).reshape(-1, 3)
    n_unique = len(np.unique(px, axis=0))
    k = max(1, min(k, n_unique))
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(px)
    shares = np.bincount(km.labels_, minlength=k).astype(float)
    return rgb_to_hex(pick_dominant(km.cluster_centers_, shares))


# ---------------------------------------------------------------- images

def media_paths(source: str, native_id: str) -> tuple[Path, Path]:
    return DATA / "media" / source / f"{native_id}.jpg", DATA / "thumbs" / source / f"{native_id}.jpg"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def save_resized(img_bytes: bytes, media: Path, thumb: Path) -> Image.Image:
    """Decode, write <=768px media and <=320px thumb JPEGs, return the thumb image."""
    img = Image.open(io.BytesIO(img_bytes))
    img.load()
    img = img.convert("RGB")
    media.parent.mkdir(parents=True, exist_ok=True)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    m = img.copy()
    m.thumbnail((MEDIA_PX, MEDIA_PX), Image.LANCZOS)
    m.save(media, "JPEG", quality=86, optimize=True)
    t = img.copy()
    t.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
    t.save(thumb, "JPEG", quality=84, optimize=True)
    return t


def run_pool(fn: Callable, items: list, workers: int = 4) -> list:
    """Map fn over items with a small thread pool; results keep input order."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def allocate(capacity: dict[str, int], total: int) -> dict[str, int]:
    """Water-filling: split `total` as evenly as possible across keys without exceeding
    each key's capacity. Deterministic (ties broken by key order)."""
    quota = {k: 0 for k in capacity}
    left = total
    open_keys = [k for k in capacity if capacity[k] > 0]
    while left > 0 and open_keys:
        share = max(1, left // len(open_keys))
        for k in list(open_keys):
            if left == 0:
                break
            add = min(share, capacity[k] - quota[k], left)
            quota[k] += add
            left -= add
            if quota[k] >= capacity[k]:
                open_keys.remove(k)
    return quota


def select_and_fetch(
    pools: dict[str, list],
    target: int,
    cap: int,
    fetch_one: Callable[[str, object], dict | None],
    workers: int = 4,
    max_rounds: int = 6,
) -> list[dict]:
    """Balanced selection across topics with top-up rounds.

    pools: topic -> ordered candidates. fetch_one(topic, cand) returns a row or None
    (failure already logged). Each round allocates the remaining target across topics
    (respecting `cap` per topic and what is left in each pool), fetches, and repeats
    until the target is met or nothing is left."""
    accepted: dict[str, list[dict]] = {t: [] for t in pools}
    pos = {t: 0 for t in pools}
    for rnd in range(max_rounds):
        have = sum(len(v) for v in accepted.values())
        if have >= target:
            break
        capacity = {t: min(cap, len(accepted[t]) + len(pools[t]) - pos[t]) for t in pools}
        quota = allocate(capacity, target)
        jobs = []
        for t in pools:
            need = quota[t] - len(accepted[t])
            if need > 0:
                jobs += [(t, c) for c in pools[t][pos[t] : pos[t] + need]]
                pos[t] += need
        if not jobs:
            break
        log.info("round %d: fetching %d items (have %d / %d)", rnd + 1, len(jobs), have, target)
        results = run_pool(lambda j: fetch_one(*j), jobs, workers=workers)
        for (t, _), row in zip(jobs, results):
            if row is not None:
                accepted[t].append(row)
    return [r for t in pools for r in accepted[t]]


def write_parquet(df, source: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{source}.parquet"
    df.to_parquet(out, index=False)
    return out
