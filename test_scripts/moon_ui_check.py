"""Ad hoc check of the Moon set at /moon/: loads on the globe, morphs to the cloud and back, runs a
query from the globe, clicks a tile; screenshots to reports/screens/moon_*.png; fails on page errors.

uv run python test_scripts/moon_ui_check.py [base_url]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000") + "/moon/"
OUT = Path(__file__).resolve().parents[1] / "reports" / "screens"
G = "window.__gallery"


def main() -> int:
    errs = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: m.type == "error" and errs.append(f"console: {m.text}"))
        pg.goto(BASE)
        pg.wait_for_function(f"{G} && {G}.ready", timeout=90000)
        pg.wait_for_timeout(2500)   # atlas texture
        g = pg.evaluate(f"{G}.globe")
        assert g and g["t"] == 0, f"should start on the globe: {g}"
        assert pg.is_visible("#globe-toggle") and not pg.is_visible("#scope-btn")
        print("holdings:", pg.inner_text("#holdings"), "| toggle:", pg.inner_text("#globe-toggle"))
        pg.screenshot(path=OUT / "moon_globe.png")

        # a tile under the centre of the globe is pickable and opens the placard
        hit = pg.evaluate(f"{G}.pick(720, 430)")
        assert hit is not None, "no tile under the globe's centre"
        pg.mouse.click(720, 430)
        pg.wait_for_timeout(600)
        assert pg.evaluate(f"{G}.detailIndex") == hit, "click did not open that tile's placard"
        pg.screenshot(path=OUT / "moon_globe_placard.png")
        pg.keyboard.press("Escape")

        pg.click("#globe-toggle")
        pg.wait_for_timeout(1300)
        pg.screenshot(path=OUT / "moon_morph_mid.png")
        pg.wait_for_function(f"{G}.globe.t === 1 && !{G}.globe.busy", timeout=15000)
        pg.wait_for_timeout(800)
        pg.screenshot(path=OUT / "moon_cloud.png")
        print("toggle now:", pg.inner_text("#globe-toggle"))

        pg.click("#globe-toggle")
        pg.wait_for_function(f"{G}.globe.t === 0 && !{G}.globe.busy", timeout=15000)

        # a query from the globe breaks it up, then hangs the wall
        pg.fill("#q", "a fresh crater with bright rays")
        pg.keyboard.press("Enter")
        pg.wait_for_function(f"{G}.phase === 'wall'", timeout=30000)
        pg.wait_for_timeout(1200)
        r = pg.evaluate(f"{G}.lastResult")
        print("query weak:", r["weak"], "top:", [n["id"] for n in r["neighbors"][:3]], "timings:", pg.evaluate(f"{G}.timings"))
        pg.screenshot(path=OUT / "moon_query_wall.png")
        b.close()
    print("errors:", errs or "none")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
