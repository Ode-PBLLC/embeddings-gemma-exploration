"""Ingest the whole Moon as a separate dataset: every tile of the z5 grid of OpenPlanetaryMap's
"Moon Hillshaded Albedo" basemap (LOLA hillshade blended with albedo), named from the IAU gazetteer.

uv run python -m gallery.ingest.moon

Writes everything under data_moon/ (never touches data/):
  items.parquet                    one image row per tile, schema.COLUMNS, id "moon:5/<x>/<y>"
  media/moon/x<x>_y<y>.jpg         512 px, stitched from the four z6 tiles beneath (what the models see)
  thumbs/moon/x<x>_y<y>.jpg        256 px (what the UI shows)
  globe_atlas.jpg                  the whole grid at 128 px per tile, 4096 x 4096, web-mercator rows
  globe.json                       {"z", "cell", "atlas", "tiles": {id: [x, y]}} for the globe view
  raw/                             the gazetteer download

Tiles are XYZ-numbered here (row 0 at the north edge); OPM serves TMS rows (row 0 at the south),
so requests flip y. Web mercator stops at +-85.05 degrees: the poles beyond that are not covered.
Each tile is titled after the largest named feature centred in it; the text lists the named
features there (type, size, what they are named for). Colour encodes brightness: dark maria blue,
bright highlands ochre.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import zipfile

import numpy as np
import pandas as pd
import requests
from PIL import Image

from gallery.ingest import _common as C
from gallery.schema import ROOT, validate

log = logging.getLogger("gallery.ingest.moon")

SOURCE = "moon"
DATA = ROOT / "data_moon"
TILE_URL = "https://s3.amazonaws.com/opmbuilder/301_moon/tiles/w/hillshaded-albedo/{z}/{x}/{y}.png"
GAZETTEER_URL = "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/MOON_nomenclature_center_pts.zip"
BASEMAP_URL = "https://www.openplanetary.org/opm-basemaps/moon-hillshaded-albedo"
LICENSE = ("LOLA hillshade + albedo (NASA/USGS), tiles by OpenPlanetaryMap; "
           "feature names: IAU/USGS Gazetteer of Planetary Nomenclature")
Z = 5                 # the item grid: 32 x 32 tiles
MEDIA_PX = 512        # each item is the 2 x 2 block of z6 tiles under it
THUMB_PX = 256
CELL_PX = 128         # globe atlas cell
KM_PER_DEG = 2 * math.pi * 1737.4 / 360
MAX_FEATURES = 10     # named features listed per tile
TEXT_CHARS = 1500
# features broad enough that "lies within" them is worth saying for tiles inside their extent
BROAD = ("Mare", "Oceanus", "Lacus", "Sinus", "Palus", "Planitia")
BROAD_CRATER_KM = 250
# brightness ramp: dark maria -> mid -> bright highlands
RAMP = [(0.0, (45, 90, 160)), (0.5, (122, 158, 126)), (1.0, (216, 154, 58))]

tile_limiter = C.RateLimiter(0.05)


def tile_bounds(x: int, y: int, z: int = Z) -> tuple[float, float, float, float]:
    """(west, east, south, north) in degrees, longitude in [-180, 180), for XYZ tile (x, y)."""
    n = 2 ** z

    def lat(row: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / n))))

    return x / n * 360 - 180, (x + 1) / n * 360 - 180, lat(y + 1), lat(y)


def lon180(lon_east: float) -> float:
    return lon_east - 360 if lon_east >= 180 else lon_east


def fmt_lat(v: float) -> str:
    return f"{abs(v):.0f}°{'N' if v >= 0 else 'S'}"


def fmt_lon(v: float) -> str:
    return f"{abs(v):.0f}°{'E' if v >= 0 else 'W'}"


def load_gazetteer() -> pd.DataFrame:
    import shapefile

    raw = DATA / "raw" / "MOON_nomenclature_center_pts.zip"
    if not raw.is_file():
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(C.fetch(GAZETTEER_URL, timeout=120).content)
    with zipfile.ZipFile(raw) as zf:
        parts = {ext: io.BytesIO(zf.read(f"MOON_nomenclature_center_pts.{ext}")) for ext in ("shp", "shx", "dbf")}
    r = shapefile.Reader(**parts)
    df = pd.DataFrame(r.records(), columns=[f[0] for f in r.fields[1:]])
    # lettered satellite craters ("Marco Polo P") are 7,000 of the 9,000 names; they say nothing new
    df = df[df["type"] != "Satellite Feature"].copy()
    df["lon"] = df["center_lon"].astype(float).map(lon180)
    df["lat"] = df["center_lat"].astype(float)
    df["diameter"] = df["diameter"].astype(float)
    df["kind"] = df["type"].str.split(",").str[0].str.strip()
    return df.sort_values("diameter", ascending=False).reset_index(drop=True)


def _unit(lon_deg, lat_deg) -> np.ndarray:
    lon, lat = np.radians(np.asarray(lon_deg, float)), np.radians(np.asarray(lat_deg, float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def _covers(f, lon: float, lat: float) -> bool:
    lo, hi = float(f.min_lon), float(f.max_lon)   # east longitudes, may run past 360
    lon_e = lon % 360
    in_lon = any(lo <= v <= hi for v in (lon_e, lon_e + 360, lon_e - 360))
    return in_lon and float(f.min_lat) <= lat <= float(f.max_lat)


def describe(x: int, y: int, gz: pd.DataFrame, broad: pd.DataFrame) -> dict:
    """Title, text, topic and link for one tile, from the gazetteer."""
    w, e, s, n = tile_bounds(x, y)
    here = gz[(gz["lon"] >= w) & (gz["lon"] < e) & (gz["lat"] >= s) & (gz["lat"] < n)]
    clon, clat = (w + e) / 2, (s + n) / 2
    within = [f for f in broad.itertuples(index=False) if _covers(f, clon, clat) and f.name not in set(here["name"])]
    side = "near side" if -90 <= clon < 90 else "far side"
    region = "north polar region" if s >= 60 else "south polar region" if n <= -60 else side
    width_km = (e - w) * KM_PER_DEG * math.cos(math.radians(clat))
    height_km = (n - s) * KM_PER_DEG
    text = (f"A patch of the Moon's {region}, about {width_km:,.0f} by {height_km:,.0f} km, from {fmt_lat(s)} to "
            f"{fmt_lat(n)} and {fmt_lon(w)} to {fmt_lon(e)}, in a LOLA hillshade blended with surface albedo.")
    if within:
        text += " It lies within " + ", ".join(f.name for f in within[:2]) + "."
    if len(here):
        bits = []
        for f in here.head(MAX_FEATURES).itertuples(index=False):
            origin = " ".join(str(f.origin).split()).rstrip(".")
            bits.append(f"{f.name} ({f.kind.lower()}, {f.diameter:.0f} km){': ' + origin if origin else ''}")
        text += " Named features here: " + "; ".join(bits) + "."
        top = here.iloc[0]
        title, topic, url = top["name"], top["kind"].lower(), top["link"].replace("http://", "https://")
    elif within:
        title, topic, url = f"{within[0].name}, {fmt_lat(clat)} {fmt_lon(clon)}", within[0].kind.lower(), BASEMAP_URL
    else:   # nothing named here (mostly the small polar tiles): place it by the nearest named feature
        near = gz.iloc[int(np.argmax(_unit(gz["lon"], gz["lat"]) @ _unit(np.array([clon]), np.array([clat]))[0]))]
        km = math.degrees(math.acos(min(1.0, float(_unit(np.array([near["lon"]]), np.array([near["lat"]]))[0]
                                                    @ _unit(np.array([clon]), np.array([clat]))[0])))) * KM_PER_DEG
        text += f" No named feature lies in it; the nearest is {near['name']} ({near['kind'].lower()}), about {km:,.0f} km away."
        title, topic, url = f"Near {near['name']}", "terrain", BASEMAP_URL
    return {"title": title, "raw_text": C.truncate_words(text, TEXT_CHARS), "topic": topic, "source_url": url}


def fetch_tile(z: int, x: int, y: int) -> Image.Image:
    r = C.fetch(TILE_URL.format(z=z, x=x, y=2 ** z - 1 - y), limiter=tile_limiter, timeout=60)
    img = Image.open(io.BytesIO(r.content))
    img.load()
    if img.size != (256, 256):
        raise ValueError(f"tile {z}/{x}/{y}: size {img.size}")
    flat = Image.new("RGBA", img.size, (128, 128, 128, 255))   # any transparent pixels -> mid grey
    flat.alpha_composite(img.convert("RGBA"))
    return flat.convert("L")


def build_tile(xy: tuple[int, int]) -> tuple[Image.Image, float]:
    """The 512 px item image for z5 tile (x, y), saved as media + thumb; returns (image, mean brightness)."""
    x, y = xy
    native = f"x{x:02d}_y{y:02d}"
    media, thumb = DATA / "media" / SOURCE / f"{native}.jpg", DATA / "thumbs" / SOURCE / f"{native}.jpg"
    if media.is_file() and thumb.is_file():
        img = Image.open(media).convert("L")
    else:
        img = Image.new("L", (MEDIA_PX, MEDIA_PX))
        parent = None
        for i in (0, 1):
            for j in (0, 1):
                try:
                    child = fetch_tile(Z + 1, 2 * x + i, 2 * y + j)
                except requests.HTTPError as e:
                    # a few z6 tiles are absent from the pyramid (S3 answers 403 for a missing key):
                    # use that quadrant of the z5 tile, upscaled, and say so
                    if e.response is None or e.response.status_code not in (403, 404):
                        raise
                    C.log_failure(SOURCE, f"moon:{Z}/{x}/{y}", f"z6 child {i},{j} missing; z5 quadrant used", e)
                    parent = parent or fetch_tile(Z, x, y)
                    child = parent.crop((128 * i, 128 * j, 128 * i + 128, 128 * j + 128)).resize((256, 256), Image.LANCZOS)
                img.paste(child, (256 * i, 256 * j))
        media.parent.mkdir(parents=True, exist_ok=True)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(media, "JPEG", quality=88, optimize=True)
        img.resize((THUMB_PX, THUMB_PX), Image.LANCZOS).convert("RGB").save(thumb, "JPEG", quality=85, optimize=True)
    return img, float(np.asarray(img, dtype=np.float32).mean())


def ramp_hex(t: float) -> str:
    for (t0, c0), (t1, c1) in zip(RAMP, RAMP[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            return C.rgb_to_hex([a + (b - a) * u for a, b in zip(c0, c1)])
    return C.rgb_to_hex(RAMP[-1][1])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    C.setup_logging()
    gz = load_gazetteer()
    broad = gz[gz["kind"].isin(BROAD) | ((gz["kind"] == "Crater") & (gz["diameter"] >= BROAD_CRATER_KM))]
    log.info("gazetteer: %d named features (%d broad)", len(gz), len(broad))
    n = 2 ** Z
    cells = [(x, y) for y in range(n) for x in range(n)]
    built = C.run_pool(build_tile, cells, workers=args.workers)
    log.info("built %d tiles", len(built))

    atlas = Image.new("L", (n * CELL_PX, n * CELL_PX))
    for (x, y), (img, _) in zip(cells, built):
        atlas.paste(img.resize((CELL_PX, CELL_PX), Image.LANCZOS), (x * CELL_PX, y * CELL_PX))
    atlas.convert("RGB").save(DATA / "globe_atlas.jpg", "JPEG", quality=86, optimize=True)

    levels = np.array([b for _, b in built])
    rank = levels.argsort().argsort() / (len(levels) - 1)   # brightness percentile, so the ramp spans the data
    rows = []
    for (x, y), t in zip(cells, rank):
        native = f"x{x:02d}_y{y:02d}"
        rows.append({
            "id": f"moon:{Z}/{x}/{y}", "source": SOURCE, "modality": "image",
            **describe(x, y, gz, broad),
            "media_path": C.rel(DATA / "media" / SOURCE / f"{native}.jpg"),
            "thumb_path": C.rel(DATA / "thumbs" / SOURCE / f"{native}.jpg"),
            "license": LICENSE, "color_hex": ramp_hex(float(t)), "creator": "LRO LOLA / USGS", "date": "",
        })
    df = pd.DataFrame(rows)
    validate(df)
    df.to_parquet(DATA / "items.parquet", index=False)
    globe = {"z": Z, "cell": CELL_PX, "atlas": "globe_atlas.jpg",
             "tiles": {f"moon:{Z}/{x}/{y}": [x, y] for x, y in cells}}
    (DATA / "globe.json").write_text(json.dumps(globe))
    log.info("wrote %s: %d rows; %d titled after a named feature in them, %d after the nearest one",
             DATA / "items.parquet", len(df), int((df["topic"] != "terrain").sum()), int((df["topic"] == "terrain").sum()))
    print(df["topic"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
