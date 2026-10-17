"""Ingest ~1,700 English Wikipedia paragraphs about the artists, styles and space subjects
in the AIC and NASA items.

uv run python -m gallery.ingest.wiki      (run after gallery.ingest.aic)

Article list: an art queue (AIC artist names by number of kept works, alternating with
  AIC style titles) and a space queue (subjects for each NASA keyword). The next article
  always comes from whichever side has fewer paragraphs so far. Each title is resolved through the MediaWiki API
  (redirects followed; missing and disambiguation pages skipped; repeats by pageid
  skipped). Each article yields at most 6 paragraphs of >= 40 words from its prose
  sections (references, see-also and similar sections are skipped).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from urllib.parse import quote

import pandas as pd

from gallery.ingest import _common as C
from gallery.ingest.nasa import KEYWORDS as NASA_KEYWORDS
from gallery.schema import WIKI_COLOR, validate

log = logging.getLogger("gallery.ingest.wiki")

SOURCE = "wiki"
API = "https://en.wikipedia.org/w/api.php"
TARGET = 1700
MAX_PARAS = 6
MIN_WORDS = 40
MAX_CHARS = 2000
LICENSE = "CC BY-SA 4.0"

api_limiter = C.RateLimiter(1.0)

SKIP_SECTIONS = {
    "references", "see also", "notes", "external links", "further reading", "bibliography",
    "sources", "citations", "footnotes", "notes and references", "references and notes",
    "works cited", "gallery", "selected works", "list of works", "partial list of works",
    "filmography", "discography", "awards", "awards and honors", "honours", "honors",
    "exhibitions", "selected exhibitions", "collections", "public collections",
    "museum collections", "literature", "explanatory notes", "general and cited references",
    "cited sources", "general references", "selected bibliography", "publications",
}

# Wikipedia subjects for each NASA keyword (the keyword's own article comes first).
SPACE_TOPICS: dict[str, list[str]] = {
    "nebula": ["Nebula", "Orion Nebula", "Crab Nebula", "Carina Nebula", "Eagle Nebula", "Pillars of Creation",
               "Helix Nebula", "Ring Nebula", "Planetary nebula", "Dumbbell Nebula", "Horsehead Nebula",
               "Cat's Eye Nebula", "Lagoon Nebula", "Supernova remnant", "Star formation", "H II region"],
    "galaxy": ["Galaxy", "Milky Way", "Andromeda Galaxy", "Spiral galaxy", "Elliptical galaxy",
               "Whirlpool Galaxy", "Sombrero Galaxy", "Galaxy cluster", "Galaxy merger", "Hubble Deep Field",
               "Large Magellanic Cloud", "Active galactic nucleus", "Supermassive black hole", "Dark matter"],
    "earth from orbit": ["Earth observation satellite", "The Blue Marble", "Earthrise", "Pale Blue Dot",
                         "Landsat program", "Remote sensing", "Earth", "Atmosphere of Earth", "Light pollution",
                         "Terra (satellite)"],
    "mars surface": ["Mars", "Geography of Mars", "Olympus Mons", "Valles Marineris", "Water on Mars",
                     "Curiosity (rover)", "Perseverance (rover)", "Opportunity (rover)", "Spirit (rover)",
                     "Sojourner (rover)", "Gale (crater)", "Jezero (crater)", "Viking program",
                     "Exploration of Mars", "Ingenuity (helicopter)"],
    "astronaut spacewalk": ["Extravehicular activity", "Astronaut", "Space suit",
                            "Extravehicular Mobility Unit", "Manned Maneuvering Unit", "Ed White (astronaut)",
                            "Alexei Leonov", "Bruce McCandless II", "Neil Armstrong", "Buzz Aldrin",
                            "Sally Ride", "Project Gemini", "Mercury Seven"],
    "rocket launch": ["Rocket launch", "Rocket", "Saturn V", "Space Shuttle", "Space Launch System",
                      "Kennedy Space Center", "Launch Complex 39", "Falcon 9", "Atlas V", "Delta IV Heavy",
                      "Space Shuttle Challenger disaster", "Space Shuttle Columbia disaster",
                      "Space Shuttle program", "Cape Canaveral Space Force Station", "Multistage rocket"],
    "jupiter": ["Jupiter", "Great Red Spot", "Galilean moons", "Io (moon)", "Europa (moon)", "Ganymede (moon)",
                "Callisto (moon)", "Juno (spacecraft)", "Galileo (spacecraft)", "Rings of Jupiter",
                "Atmosphere of Jupiter", "Voyager 1", "Voyager 2", "Pioneer 10"],
    "aurora": ["Aurora", "Solar wind", "Magnetosphere", "Geomagnetic storm", "Van Allen radiation belt",
               "Ionosphere", "Airglow", "Space weather", "Earth's magnetic field"],
    "moon": ["Moon", "Geology of the Moon", "Lunar mare", "Far side of the Moon", "Lunar craters",
             "Tycho (lunar crater)", "Lunar eclipse", "Moon landing", "Lunar Reconnaissance Orbiter",
             "Artemis program", "Lunar Roving Vehicle", "Lunar regolith", "Exploration of the Moon",
             "Lunar phase"],
    "saturn": ["Saturn", "Rings of Saturn", "Titan (moon)", "Enceladus", "Cassini–Huygens", "Mimas (moon)",
               "Iapetus (moon)", "Saturn's hexagon", "Moons of Saturn", "Huygens (spacecraft)"],
    "comet": ["Comet", "Halley's Comet", "Comet Hale–Bopp", "Comet Shoemaker–Levy 9", "Comet NEOWISE",
              "67P/Churyumov–Gerasimenko", "Rosetta (spacecraft)", "Deep Impact (spacecraft)", "Stardust (spacecraft)",
              "Oort cloud", "Kuiper belt", "Asteroid", "Meteor shower"],
    "international space station": ["International Space Station", "Assembly of the International Space Station",
                                     "Zvezda (ISS module)", "Destiny (ISS module)", "Kibo (ISS module)",
                                     "Columbus (ISS module)", "Cupola (ISS module)", "Canadarm2",
                                     "Soyuz (spacecraft)", "SpaceX Dragon 2", "Mir", "Skylab",
                                     "Expedition 1", "Space station"],
    "hurricane from space": ["Tropical cyclone", "Hurricane Katrina", "Hurricane Sandy", "Hurricane Harvey",
                             "Hurricane Maria", "Eye (cyclone)", "Weather satellite", "GOES-16",
                             "Typhoon Haiyan", "Saffir–Simpson scale", "Hurricane Andrew"],
    "solar flare": ["Solar flare", "Sun", "Coronal mass ejection", "Sunspot", "Solar cycle", "Stellar corona",
                    "Solar Dynamics Observatory", "Parker Solar Probe", "Solar and Heliospheric Observatory", "Solar prominence",
                    "Photosphere", "Solar eclipse", "Heliosphere"],
    "apollo": ["Apollo program", "Apollo 11", "Apollo 8", "Apollo 13", "Apollo 17", "Apollo 15", "Apollo 1",
               "Apollo Lunar Module", "Apollo command and service module", "Apollo 12", "Apollo 16",
               "Michael Collins (astronaut)", "Gene Cernan", "Mission Control Center"],
    "hubble": ["Hubble Space Telescope", "Edwin Hubble", "James Webb Space Telescope",
               "Spitzer Space Telescope", "Chandra X-ray Observatory", "Hubble Ultra-Deep Field",
               "STS-61", "Great Observatories program", "Exoplanet", "Black hole", "Supernova",
               "Big Bang", "Cosmic microwave background"],
}

_HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


def space_titles() -> list[str]:
    """Round-robin across NASA keywords so every keyword's subjects appear early."""
    lists = [SPACE_TOPICS[k] for k in NASA_KEYWORDS]
    out, seen = [], set()
    for i in range(max(len(v) for v in lists)):
        for v in lists:
            if i < len(v) and v[i] not in seen:
                seen.add(v[i])
                out.append(v[i])
    return out


_GENERIC_ARTIST = re.compile(r"(?i)\b(unknown|anonymous|ancient|culture|artist|maker|school|workshop|style of|after)\b")


def aic_titles() -> tuple[list[str], list[str]]:
    """(artists by number of kept works, styles by number of kept works) from the AIC run."""
    path = C.RAW / "aic_entities.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: run `uv run python -m gallery.ingest.aic` first")
    ents = json.loads(path.read_text(encoding="utf-8"))
    artists = Counter(e["artist_title"].strip() for e in ents if e.get("artist_title"))
    styles = Counter(e["style_title"].strip() for e in ents if e.get("style_title"))
    a = [n for n, _ in artists.most_common() if not _GENERIC_ARTIST.search(n)]
    s = [n[:1].upper() + n[1:] for n, _ in styles.most_common()]
    return a, s


def interleave(artists: list[str], styles: list[str]) -> list[tuple[str, str]]:
    """Art-side order: artist, style, artist, style, ... -> (pool, title)."""
    out = []
    for i in range(max(len(artists), len(styles))):
        if i < len(artists):
            out.append(("artist", artists[i]))
        if i < len(styles):
            out.append(("style", styles[i]))
    return out


def style_variants(style: str) -> list[str]:
    """AIC style titles are often like 'Edo (japanese period)' or 'Yoruba', which are
    missing or disambiguation pages. Try the title, then '<base> art'."""
    base = re.sub(r"\s*\(.*$", "", style).strip()
    out = [style]
    if base:
        out.append(f"{base[:1].upper()}{base[1:]} art")
    return list(dict.fromkeys(out))


def fetch_article(title: str) -> dict:
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "extracts|pageprops|info", "explaintext": 1, "exsectionformat": "wiki",
        "ppprop": "disambiguation", "titles": title,
    }
    name = f"article_{C.cache_key(title)}"
    return C.cached_json(SOURCE, name, lambda: C.fetch(API, params=params, limiter=api_limiter).json())


def chunk_article(extract: str, max_paras: int = MAX_PARAS, min_words: int = MIN_WORDS) -> list[tuple[str | None, str]]:
    """Split a plain-text extract into (section heading or None, paragraph) pairs.

    Keeps paragraphs with >= min_words words from prose sections, in article order,
    at most max_paras. Sections in SKIP_SECTIONS (and their subsections) are skipped."""
    out: list[tuple[str | None, str]] = []
    section: str | None = None
    skip_level: int | None = None  # level of the skipped ancestor heading, if any
    for line in extract.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _HEADING.match(line)
        if m:
            level, heading = len(m.group(1)), m.group(2)
            if skip_level is not None and level > skip_level:
                continue
            skip_level = level if heading.casefold() in SKIP_SECTIONS else None
            section = heading
            continue
        if skip_level is not None:
            continue
        if "\\displaystyle" in line or len(line.split()) < min_words:
            continue
        out.append((section, C.truncate_words(line, MAX_CHARS)))
        if len(out) >= max_paras:
            break
    return out


def article_url(title: str, section: str | None) -> str:
    url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,'-.:!/")
    if section:
        url += "#" + quote(section.replace(" ", "_"), safe="()_,'-.:!")
    return url


def rows_for(page: dict) -> list[dict]:
    rows = []
    for n, (section, para) in enumerate(chunk_article(page.get("extract") or "")):
        rows.append({
            "id": f"wiki:{page['pageid']}:{n}",
            "source": SOURCE,
            "modality": "text",
            "title": f"{page['title']} — {section}" if section else page["title"],
            "raw_text": para,
            "media_path": "",
            "thumb_path": "",
            "source_url": article_url(page["title"], section),
            "license": LICENSE,
            "color_hex": WIKI_COLOR,
            "topic": page["title"],
            "creator": "",
            "date": "",
        })
    return rows


def resolve(title: str) -> dict | None:
    """Fetch an article; None (logged) if it is missing, a disambiguation page or failed."""
    try:
        data = fetch_article(title)
    except Exception as e:  # noqa: BLE001 - logged, article skipped
        C.log_failure(SOURCE, title, "fetch", e)
        return None
    pages = (data.get("query") or {}).get("pages") or []
    if not pages or pages[0].get("missing") or pages[0].get("invalid"):
        log.info("skip (missing): %s", title)
        return None
    page = pages[0]
    if "disambiguation" in (page.get("pageprops") or {}):
        log.info("skip (disambiguation): %s", title)
        return None
    return page


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args(argv)
    C.setup_logging()
    artists, styles = aic_titles()
    queues = {"art": iter(interleave(artists, styles)), "space": iter(("space", t) for t in space_titles())}
    log.info("candidate articles: %d artists, %d styles, %d space", len(artists), len(styles), len(space_titles()))
    rows: list[dict] = []
    seen_pages: set[int] = set()
    per_side = Counter({"art": 0, "space": 0})
    per_pool = Counter()
    live = ["art", "space"]
    # Always draw the next article from the side (art / space) with fewer paragraphs so far.
    while len(rows) < args.target and live:
        side = min(live, key=lambda k: (per_side[k], k))
        nxt = next(queues[side], None)
        if nxt is None:
            live.remove(side)
            continue
        pool, title = nxt
        page = None
        for t in style_variants(title) if pool == "style" else [title]:
            page = resolve(t)
            if page is not None:
                break
        if page is None or page["pageid"] in seen_pages:
            continue
        seen_pages.add(page["pageid"])
        new = rows_for(page)[: args.target - len(rows)]
        rows += new
        per_side[side] += len(new)
        per_pool[pool] += len(new)
    df = pd.DataFrame(rows)
    validate(df)
    out = C.write_parquet(df, SOURCE)
    log.info("wrote %s: %d rows from %d articles; by pool %s", out, len(df), df["topic"].nunique(), dict(per_pool))


if __name__ == "__main__":
    main()
