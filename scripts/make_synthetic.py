"""Generate a synthetic dataset that matches the real data contracts (BUILD.md), for building
and testing the interface without the real corpus or models.

    uv run python scripts/make_synthetic.py [--out data/synthetic] [--n 5000] [--seed 7] [--no-layout]

Writes, under --out:
  items.parquet, summaries.jsonl, vectors.npy, vector_ids.json, layout.npy (UMAP via gallery.layout),
  retrieval.json, fake_topics.npz (centroids for the keyword fake embedder),
  thumbs/{aic,nasa}/<id>.jpg (placeholder art), layout_report.md.
About 4% of items get no summary so the summary_pending path is exercised.
Everything is invented. Titles and creators are fictional.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gallery.schema import COLUMNS, WIKI_COLOR, validate  # noqa: E402

DIM = 768
NOISE = 1.9  # within-topic spread; higher gives a more continuous cloud

# name, keywords, (aic, nasa, wiki) weights, hue range (deg), sat, light, title nouns
TOPICS = [
    ("Water lilies", "water lily lilies pond garden monet reflection", (5, 0, 2), (90, 200), .45, .55,
     ["Water Lilies", "The Lily Pond", "Garden Pool at Dusk", "Reflections, Morning", "Green Water"]),
    ("Impressionist fields", "landscape impressionism impressionist field fields hay haystack wheat summer", (5, 0, 2), (35, 60), .6, .6,
     ["Haystacks, End of Summer", "Wheat Field", "Poplars in Wind", "Meadow at Noon", "The Harvest"]),
    ("Portraits", "portrait face woman man self faces person", (5, 0, 2), (10, 35), .45, .5,
     ["Portrait of a Woman", "Self-Portrait", "The Reader", "Young Man in Grey", "Woman with Fan"]),
    ("Japanese prints", "japanese japan wave waves print woodblock fuji mount ukiyo", (5, 0, 2), (200, 230), .55, .45,
     ["Great Wave off a Quiet Coast", "Mount Fuji from the Ferry", "Rain at the Bridge", "Plum Garden", "Night Snow"]),
    ("Still life", "still life fruit flowers vase bouquet table", (5, 0, 2), (345, 20), .6, .5,
     ["Still Life with Lemons", "Vase of Peonies", "Apples on a Cloth", "Bouquet", "Table with Oysters"]),
    ("Storms at sea", "storm storms sea ocean ship ships wave waves hurricane weather", (3, 3, 2), (195, 215), .25, .45,
     ["Ship in a Gale", "The Storm", "Hurricane over the Atlantic", "Breakers", "Squall at Sea"]),
    ("The Moon", "moon lunar apollo crater craters moonlight moonlit", (2, 4, 2), (30, 50), .08, .7,
     ["Moonrise", "Lunar Surface, Crater Field", "Apollo Landing Site", "Earthrise over the Moon", "Moonlit Bay"]),
    ("Mars", "mars martian red planet rover dust desert", (0, 5, 2), (8, 22), .65, .45,
     ["Martian Dunes", "Rover Tracks at Sunset", "Valles Panorama", "Dust Devil", "Red Plains"]),
    ("Giant planets", "saturn jupiter rings ring planet planets giant gas", (0, 5, 2), (25, 45), .55, .6,
     ["Saturn's Rings in Shadow", "Jupiter's Great Storm", "Ring Plane Crossing", "Cloud Bands", "Moons of Saturn"]),
    ("Nebulae", "nebula nebulae star stars cosmic galaxy galaxies deep space universe", (0, 5, 2), (280, 330), .65, .5,
     ["Pillars of Gas", "Spiral Galaxy", "The Veil Nebula", "Stellar Nursery", "Cosmic Cliffs"]),
    ("Earth from orbit", "earth orbit clouds city lights home planet blue", (0, 5, 2), (180, 210), .55, .5,
     ["Earth at Night", "Blue Marble", "Cloud Streets", "River Delta from Orbit", "Aurora from the Station"]),
    ("Astronauts", "astronaut astronauts spacewalk station suit crew human", (0, 5, 2), (20, 35), .75, .55,
     ["Spacewalk", "Crew in the Cupola", "Untethered", "Suit Check", "Training Pool"]),
    ("Rockets", "rocket rockets launch shuttle fire engine liftoff", (0, 5, 2), (15, 30), .85, .5,
     ["Night Launch", "Liftoff", "Shuttle on the Pad", "Engine Test", "Booster Return"]),
    ("The Sun", "sun solar flare light star corona", (1, 4, 2), (35, 50), .9, .55,
     ["Solar Flare", "Corona", "Sunspot Group", "Prominence", "Transit"]),
    ("Abstraction", "abstract abstraction color colour geometry modern composition shapes", (5, 0, 2), (0, 360), .7, .5,
     ["Composition in Red and Blue", "Untitled (Orange)", "Squares", "Rhythm", "Field No. 4"]),
    ("Sculpture", "sculpture bronze marble statue carving figure", (5, 0, 2), (30, 45), .3, .45,
     ["Bronze Torso", "Marble Head", "Seated Figure", "Dancer", "Carved Mask"]),
    ("Textiles", "textile textiles pattern weaving fabric quilt", (5, 0, 2), (340, 360), .55, .4,
     ["Woven Panel", "Quilt, Star Pattern", "Silk Fragment", "Indigo Cloth", "Tapestry"]),
    ("Night", "night dark darkness nocturne evening stars", (3, 3, 2), (220, 250), .4, .3,
     ["Nocturne in Blue", "Night Street", "Starry Evening", "Lamplight", "Night Harbour"]),
    ("Winter", "snow winter ice cold frozen antarctica glacier", (3, 3, 2), (195, 210), .3, .75,
     ["Snow at Argenteuil", "Ice Shelf", "Winter Morning", "Frozen River", "Glacier from Orbit"]),
    ("Cities", "city cities street streets paris urban buildings", (5, 1, 2), (20, 40), .3, .55,
     ["Paris Street, Rainy Day", "Boulevard", "The Bridge", "Rooftops", "Market Square"]),
    ("Myth and devotion", "myth god goddess saint religious angel angels heaven", (5, 0, 2), (40, 50), .7, .5,
     ["Annunciation", "Saint in the Desert", "Venus Rising", "Altarpiece Panel", "Angel"]),
    ("Animals", "bird birds animal animals horse horses cat dog lion", (5, 0, 2), (25, 100), .5, .5,
     ["Horses in a Meadow", "Heron", "Sleeping Cat", "Lion Hunt", "Birds in Flight"]),
]

AIC_CREATORS = ["Elise Varnier", "Tomas Holm", "Hiroshi Aoki", "Marguerite Dulac", "Paul Rennard", "Ines Calvo",
                "Anonymous", "Workshop of R. Baldi", "Clara Weiss", "Jun Takeda", "Henri Morel", "Sofia Lind"]
NASA_CREATORS = ["NASA/JPL", "NASA Johnson", "NASA Goddard", "NASA Kennedy", "NASA/ESA", "NASA Ames"]
WIKI_ARTICLES = {t[0]: [t[0], f"History of {t[0].lower()}", f"{t[0]} in art", f"{t[0]} (overview)"] for t in TOPICS}

SENTENCES = [
    "The work turns on {kw}, rendered with a patient attention to light and surface.",
    "{title} belongs to a long conversation about {kw}, and it answers with restraint rather than spectacle.",
    "Look closely and the {kw} resolves into small decisions: an edge left soft, a colour held back.",
    "It was made at a moment when {kw} was being looked at in new ways, and that curiosity shows.",
    "The composition leads the eye from the foreground into a quieter distance.",
    "What makes it notable is how ordinary the subject is, and how carefully it is seen.",
    "Scholars often pair it with other studies of {kw} for its unusual palette.",
]


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def hsl_hex(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, min(max(l, 0), 1), min(max(s, 0), 1))
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def summary_for(rng, title: str, kws: list[str]) -> str:
    picks = rng.choice(len(SENTENCES), size=4, replace=False)
    parts = [SENTENCES[i].format(title=title, kw=" and ".join(rng.choice(kws, size=min(2, len(kws)), replace=False)))
             for i in picks]
    return " ".join(parts)


# ---------- placeholder thumbnails ----------

def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def thumb_art(rng, source: str, color: str, w: int, h: int, label: str, font) -> Image.Image:
    base = np.array(hex_rgb(color), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if source == "nasa":
        # dark field, a lit disc or glow, a scatter of stars
        bg = np.zeros((h, w, 3), np.float32) + np.array([10, 12, 22], np.float32)
        cx, cy = rng.uniform(.3, .7) * w, rng.uniform(.3, .7) * h
        r = rng.uniform(.18, .42) * min(w, h)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        if rng.random() < .5:  # planet with terminator shading
            lx, ly = rng.normal(size=2)
            shade = np.clip(1 - (((xx - cx) * lx + (yy - cy) * ly) / (r * np.hypot(lx, ly)) + 1) / 2.2, 0.08, 1)
            disc = (d < r)[..., None] * (base * (0.55 + 0.6 * shade[..., None]))
            img = np.where((d < r)[..., None], disc, bg)
        else:  # nebula glow
            glow = np.exp(-(d / (r * 1.4)) ** 2)[..., None]
            img = bg + glow * base * 1.3
        stars = rng.random((h, w)) > 0.996
        img[stars] = 235
        out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    else:
        # painted ground: two-tone gradient plus soft colour fields
        h2 = rng.uniform(-30, 30)
        r_, g_, b_ = [c / 255 for c in base]
        hh, ll, ss = colorsys.rgb_to_hls(r_, g_, b_)
        alt = np.array([c * 255 for c in colorsys.hls_to_rgb((hh + h2 / 360) % 1, min(ll * 1.25, .92), ss)], np.float32)
        t = (yy / h)[..., None]
        img = base * (1 - t) + alt * t
        for _ in range(rng.integers(1, 4)):
            bx, by = rng.uniform(0, w), rng.uniform(0, h)
            sx, sy = rng.uniform(.15, .45) * w, rng.uniform(.1, .35) * h
            f = np.exp(-(((xx - bx) / sx) ** 2 + ((yy - by) / sy) ** 2))[..., None]
            tint = np.array([c * 255 for c in colorsys.hls_to_rgb((hh + rng.uniform(-.08, .08)) % 1,
                                                                    rng.uniform(.25, .8), min(ss + .1, 1))], np.float32)
            img = img * (1 - .7 * f) + tint * .7 * f
        img += rng.normal(0, 6, img.shape)  # canvas grain
        out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    d = ImageDraw.Draw(out)
    d.text((8, h - 18), label, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    return out


# ---------- dataset ----------

def _repo_path(p: Path) -> str:
    """Repo-relative path when under the repo (schema convention), absolute otherwise (tmp test dirs)."""
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def build(out: Path, n: int, seed: int, layout: bool) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    font = _font(11)

    # topics share a parent direction per family (art / space / both), so related topics sit near each other
    family = {"art": unit(rng.normal(size=DIM)), "space": unit(rng.normal(size=DIM)), "both": unit(rng.normal(size=DIM))}
    fam_of = ["space" if w[0] == 0 else "art" if w[1] == 0 else "both" for w in (t[2] for t in TOPICS)]
    centroids = unit(np.stack([0.75 * family[f] + unit(rng.normal(size=DIM)) for f in fam_of]))
    src_dirs = {s: unit(rng.normal(size=DIM)) for s in ("aic", "nasa", "wiki")}

    # allocate counts: a third per source, spread across topics by weight
    rows, vecs, summaries = [], [], []
    per_source = {"aic": n // 3, "nasa": n // 3, "wiki": n - 2 * (n // 3)}
    for si, src in enumerate(("aic", "nasa", "wiki")):
        w = np.array([t[2][si] for t in TOPICS], float)
        topic_of = rng.choice(len(TOPICS), size=per_source[src], p=w / w.sum())
        for j, ti in enumerate(topic_of):
            name, kw, _, hue, sat, light, nouns = TOPICS[ti]
            kws = kw.split()
            # 22% bridge items between two topics keep the cloud continuous
            other = rng.integers(len(TOPICS)) if rng.random() < .22 else None
            v = centroids[ti] + 0.25 * src_dirs[src] + rng.normal(0, NOISE / np.sqrt(DIM), DIM)
            if other is not None:
                v = 0.65 * centroids[ti] + 0.65 * centroids[other] + 0.25 * src_dirs[src] + rng.normal(0, NOISE / np.sqrt(DIM), DIM)
            vecs.append(unit(v))
            native = f"{ti:02d}{j:05d}"
            iid = f"{src}:{native}"
            if src == "wiki":
                art = rng.choice(WIKI_ARTICLES[name])
                title = f"{art} — ¶{rng.integers(1, 12)}"
                raw = f"{art}. " + summary_for(rng, art, kws) + " " + summary_for(rng, art, kws)
                color, modality, thumb, creator, date = WIKI_COLOR, "text", "", "Wikipedia", ""
                url = f"https://en.wikipedia.org/wiki/{art.replace(' ', '_')}"
                lic, topic = "CC BY-SA 4.0", art
            else:
                lo, hi = hue
                hdeg = rng.uniform(lo, hi if hi > lo else hi + 360)
                color = hsl_hex(hdeg, sat + rng.uniform(-.1, .1), light + rng.uniform(-.12, .12))
                noun = rng.choice(nouns)
                title = f"{noun}" + (f", {rng.integers(1860, 1940)}" if src == "aic" and rng.random() < .3 else "")
                if src == "aic":
                    creator, date = rng.choice(AIC_CREATORS), str(rng.integers(1500, 1980))
                    url, lic, topic = f"https://example.org/synthetic/aic/{native}", "CC0", name
                else:
                    creator, date = rng.choice(NASA_CREATORS), f"{rng.integers(1965, 2026)}-{rng.integers(1, 13):02d}"
                    url, lic, topic = f"https://example.org/synthetic/nasa/{native}", "Public domain", name
                raw = f"{title}. " + summary_for(rng, title, kws)
                modality = "image"
                thumb = _repo_path(out / "thumbs" / src / f"{native}.jpg")
                aspect = rng.choice([0.75, 0.8, 1.0, 1.25, 1.33, 1.5]) if src == "aic" else rng.choice([1.0, 1.0, 1.33, 1.5])
                tw, th = (320, int(320 / aspect)) if aspect >= 1 else (int(320 * aspect), 320)
                p = ROOT / thumb
                p.parent.mkdir(parents=True, exist_ok=True)
                thumb_art(rng, src, color, tw, th, f"synthetic {iid}", font).save(p, quality=85)
            rows.append({
                "id": iid, "source": src, "modality": modality, "title": title, "raw_text": raw,
                "media_path": thumb, "thumb_path": thumb, "source_url": url, "license": lic,
                "color_hex": color, "topic": topic, "creator": creator, "date": date,
            })
            if rng.random() > 0.04:
                summaries.append({"id": iid, "summary": summary_for(rng, title, kws)})

    df = pd.DataFrame(rows, columns=COLUMNS)
    validate(df, check_files=True)
    order = rng.permutation(len(df))  # shuffle so asset order is not grouped by source
    df = df.iloc[order].reset_index(drop=True)
    V = np.asarray(vecs, np.float32)[order]
    df.to_parquet(out / "items.parquet", index=False)
    with (out / "summaries.jsonl").open("w") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")
    np.save(out / "vectors.npy", V)
    (out / "vector_ids.json").write_text(json.dumps(df["id"].tolist()))
    np.savez(out / "fake_topics.npz", centroids=centroids.astype(np.float32),
             keywords=np.array([t[1] for t in TOPICS]), names=np.array([t[0] for t in TOPICS]))

    # weak threshold from the fake embedder, using the BUILD.md midpoint rule
    from gallery.search import KeywordEmbedder
    emb = KeywordEmbedder(out / "fake_topics.npz")
    on = ["water lilies in a garden pond", "the surface of mars", "a storm at sea", "portrait of a woman",
          "rings of saturn", "astronaut on a spacewalk", "winter snow", "abstract color", "night in the city",
          "bronze sculpture", "a rocket launch", "the moon", "japanese woodblock print", "still life with fruit",
          "nebula and stars"]
    off = ["quarterly tax filing deadlines", "how to reset my router", "best pizza near me", "kubernetes pod eviction",
           "mortgage refinance rates", "python list comprehension", "fantasy football draft", "printer toner",
           "airline baggage fees", "sql join syntax"]
    top1 = lambda qs: np.array([float((emb.embed(q) @ V.T).max()) for q in qs])  # noqa: E731
    t_on, t_off = top1(on), top1(off)
    thr = float((np.percentile(t_on, 10) + np.percentile(t_off, 90)) / 2)
    cfg = {"mode": "raw", "image_input": "image", "k": 12, "weak_threshold": round(thr, 4),
           "synthetic": True, "on_topic_top1_p10": round(float(np.percentile(t_on, 10)), 4),
           "off_topic_top1_p90": round(float(np.percentile(t_off, 90)), 4)}
    (out / "retrieval.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if layout:
        from gallery.layout import run
        run(out, out / "layout_report.md")

    counts = df["source"].value_counts().to_dict()
    print(f"synthetic: {len(df)} items {counts}, {len(summaries)} summaries, weak_threshold={thr:.3f} "
          f"(on p10 {cfg['on_topic_top1_p10']}, off p90 {cfg['off_topic_top1_p90']}) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-layout", action="store_true")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    build(out, args.n, args.seed, layout=not args.no_layout)


if __name__ == "__main__":
    main()
