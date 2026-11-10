"""Structured metadata for the AIC works already ingested: numeric years and the artist's name.

uv run python -m gallery.ingest.aic_meta          (after gallery.ingest.aic)

The display strings the corpus keeps ("c. 1525-c. 1550", "19th century", "Claude Monet (French,
1840-1926)") are for people; the year filter, the timeline and the artist facet need AIC's own
structured fields. Writes data/raw/aic_meta.parquet: artwork id ("aic:<n>"), date_start, date_end
(integer years, negative = BCE, may be missing) and artist_title (AIC's canonical artist name, may
be missing, e.g. for "Egyptian" or "Roman" works). gallery.build_assets joins it in.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from gallery.ingest import _common as C
from gallery.ingest.aic import HEADERS
from gallery.schema import DATA

log = logging.getLogger("gallery.ingest.aic_meta")

SOURCE = "aic"
ARTWORKS = "https://api.artic.edu/api/v1/artworks"
FIELDS = "id,date_start,date_end,artist_title"
OUT = DATA / "raw" / "aic_meta.parquet"

api_limiter = C.RateLimiter(1.0)


def fetch_meta(native_ids: list[str]) -> pd.DataFrame:
    rows = []
    for k in range(0, len(native_ids), 100):
        chunk = native_ids[k:k + 100]
        params = {"ids": ",".join(chunk), "fields": FIELDS, "limit": 100}
        data = C.cached_json(SOURCE, f"artwork_meta_{C.cache_key(chunk)}",
                             lambda: C.fetch(ARTWORKS, params=params, headers=HEADERS, limiter=api_limiter).json())
        rows += data["data"]
    df = pd.DataFrame(rows, columns=FIELDS.split(","))
    df["id"] = "aic:" + df["id"].astype(str)
    for c in ("date_start", "date_end"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["artist_title"] = df["artist_title"].fillna("").astype(str).str.strip()
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    C.setup_logging()
    arts = pd.read_parquet(DATA / "raw" / "aic.parquet")
    native = arts["id"].str.split(":").str[1].tolist()
    df = fetch_meta(native)
    missing = sorted(set(arts["id"]) - set(df["id"]))
    if missing:   # the API silently drops ids it no longer serves: say which
        log.warning("%d artworks not returned by the AIC API, e.g. %s", len(missing), missing[:3])
    df.to_parquet(OUT, index=False)
    log.info("wrote %s: %d artworks; with a start year %d, with an artist %d", OUT, len(df),
             int(df["date_start"].notna().sum()), int((df["artist_title"] != "").sum()))


if __name__ == "__main__":
    main()
