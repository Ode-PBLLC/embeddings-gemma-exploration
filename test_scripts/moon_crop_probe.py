"""Probe: crop OPM moon-hillshaded-albedo tiles around a few IAU features, to judge what moon items look like.

uv run python test_scripts/moon_crop_probe.py <nomenclature.csv> <outdir>
"""
import io
import math
import sys
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

URL = "https://tiles.geolibre.app/opm/moon-hillshaded-albedo/{z}/{x}/{y}.png"
KM_PER_DEG = 2 * math.pi * 1737.4 / 360
OUT = 512
NAMES = ["Tycho", "Copernicus", "Mare Tranquillitatis", "Montes Apenninus", "Vallis Schröteri",
         "Clavius", "Mare Orientale", "Aristarchus", "Rupes Recta", "Petavius"]


def crop(lon, lat, diam_km, session):
    lon = lon - 360 if lon > 180 else lon
    feat_deg = diam_km / KM_PER_DEG
    z = math.log2(0.45 * OUT * 360 * math.cos(math.radians(lat)) / (feat_deg * 256))
    z = max(1, min(6, math.floor(z)))
    n = 256 * 2 ** z
    px = (lon + 180) / 360 * n
    py = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    x0, y0 = int(px - OUT / 2), int(py - OUT / 2)
    canvas = Image.new("LA", (OUT, OUT))
    for tx in range(x0 // 256, (x0 + OUT) // 256 + 1):
        for ty in range(y0 // 256, (y0 + OUT) // 256 + 1):
            if not (0 <= ty < 2 ** z):
                continue
            # OPM serves TMS rows (y counted from the south)
            r = session.get(URL.format(z=z, x=tx % 2 ** z, y=2 ** z - 1 - ty), timeout=30)
            r.raise_for_status()
            canvas.paste(Image.open(io.BytesIO(r.content)), (tx * 256 - x0, ty * 256 - y0))
    return canvas.convert("L"), z


def main(csv, outdir):
    df = pd.read_csv(csv).set_index("name")
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    tiles = []
    for name in NAMES:
        r = df.loc[name]
        img, z = crop(r.center_lon, r.center_lat, r.diameter, s)
        img.save(out / f"{name}.png")
        print(f"{name:22s} z{z} d={r.diameter:.0f}km lat={r.center_lat:.1f}")
        tiles.append(img.resize((256, 256)))
    sheet = Image.new("L", (256 * 5, 256 * 2))
    for i, t in enumerate(tiles):
        sheet.paste(t, (256 * (i % 5), 256 * (i // 5)))
    sheet.save(out / "contact_sheet.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
