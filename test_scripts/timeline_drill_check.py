"""Ad hoc check of the timeline and the Collection -> Artist -> Years -> Query drill-down against a
running server. Screenshots to reports/screens/timeline_*.png; fails on page errors or a broken step.

uv run python test_scripts/timeline_drill_check.py [base_url]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUT = Path(__file__).resolve().parents[1] / "reports" / "screens"
G = "window.__gallery"


def main() -> int:
    errs = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: m.type == "error" and errs.append(f"console: {m.text}"))
        pg.goto(BASE + "/")
        pg.wait_for_function(f"{G} && {G}.ready", timeout=90000)
        assert pg.is_visible("#timeline-toggle") and not pg.is_visible("#trail")

        pg.click("#timeline-toggle")
        pg.wait_for_function(f"{G}.timeline.t === 1 && !{G}.timeline.busy", timeout=15000)
        pg.wait_for_timeout(400)
        n_all = pg.evaluate(f"{G}.timeline.members")
        labels = pg.eval_on_selector_all("#timeline-axis span", "els => els.map(e => e.textContent)")
        print("timeline, everything dated:", n_all, "works; ticks", labels)
        assert n_all == pg.evaluate("window.__cloud.items.filter(it => it.years).length")
        pg.screenshot(path=OUT / "timeline_all.png")

        # drill down: Collection (metadata)
        pg.click("#scope-btn")
        pg.click("[data-source=nasa]")
        pg.click("[data-source=wiki]")
        pg.wait_for_function(f"!{G}.timeline.busy", timeout=15000)
        n_aic = pg.evaluate(f"{G}.timeline.members")
        # Artist (metadata)
        pg.fill("#scope-artist", "Alfred Stieglitz")
        pg.dispatch_event("#scope-artist", "change")
        pg.wait_for_timeout(200)
        pg.wait_for_function(f"!{G}.timeline.busy", timeout=15000)
        n_st = pg.evaluate(f"{G}.timeline.members")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(500)
        print("drill:", n_all, "->", n_aic, "(Chicago) ->", n_st, "(Stieglitz); trail", pg.evaluate(f"{G}.trail"))
        assert n_aic < n_all and 0 < n_st < n_aic
        pg.screenshot(path=OUT / "timeline_stieglitz.png")

        # a work on the timeline can be hovered and clicked
        idx = pg.evaluate("[...window.__cloud.timeline.members][0]")
        pt = pg.evaluate(f"{G}.screenOf({idx})")
        pg.mouse.click(pt["x"], pt["y"])
        pg.wait_for_timeout(500)
        assert pg.evaluate(f"{G}.detailIndex") is not None, "click on a timeline dot opened nothing"
        pg.keyboard.press("Escape")

        # Query (semantic): stays on the timeline, with lines out to the hits and the wall, filters kept
        pg.fill("#q", "clouds")
        pg.keyboard.press("Enter")
        pg.wait_for_function(f"{G}.phase === 'wall'", timeout=60000)
        pg.wait_for_timeout(900)
        r = pg.evaluate(f"{G}.lastResult")
        print("query:", r["filters"], "eligible", r["eligible"], "trail", pg.evaluate(f"{G}.trail"),
              "lines", pg.evaluate(f"{G}.lineCount"), "pictures", pg.evaluate(f"{G}.billboardCount"))
        assert r["filters"].get("artist") == ["Alfred Stieglitz"] and pg.evaluate(f"{G}.timeline.t") == 1
        assert pg.evaluate(f"{G}.lineCount") == len(r["neighbors"]) and pg.evaluate(f"{G}.billboardCount") == len(r["neighbors"])
        # every line ends on its work's dot in the swarm
        off = pg.evaluate("""(() => { const c = window.__cloud; return c.lines.map((l) => {
          const i = l.index; return Math.hypot(l.to.x - c.shown[i*3], l.to.y - c.shown[i*3+1], l.to.z - c.shown[i*3+2]); }); })()""")
        assert max(off) < 1e-4, f"lines do not end on the timeline dots: {off}"
        pg.screenshot(path=OUT / "timeline_query_wall.png")

        # closing the wall stays on the timeline; a query with no filters only searches the dated works
        pg.click("#close-wall")
        pg.wait_for_timeout(900)
        assert pg.evaluate(f"{G}.timeline.t") == 1 and pg.evaluate(f"{G}.lineCount") == 0
        pg.click("#scope-btn")
        pg.click("#scope-clear")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(1500)
        pg.fill("#q", "rockets")
        pg.keyboard.press("Enter")
        pg.wait_for_function(f"{G}.phase === 'wall'", timeout=60000)
        pg.wait_for_timeout(900)
        r2 = pg.evaluate(f"{G}.lastResult")
        items = pg.evaluate("window.__cloud.items.map(it => !!it.years)")
        assert all(items[n["index"]] for n in r2["neighbors"]), "an undated work came back on the timeline"
        print("unfiltered timeline query: eligible", r2["eligible"], "wall note:", pg.inner_text("#wall-sub"))
        pg.screenshot(path=OUT / "timeline_query_rockets.png")
        pg.click("#close-wall")
        pg.wait_for_timeout(900)
        pg.click("#scope-btn")
        pg.click("[data-source=nasa]")
        pg.click("[data-source=wiki]")
        pg.fill("#scope-artist", "Alfred Stieglitz")
        pg.dispatch_event("#scope-artist", "change")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(1500)
        pg.fill("#q", "clouds")
        pg.keyboard.press("Enter")
        pg.wait_for_function(f"{G}.phase === 'wall'", timeout=60000)
        pg.wait_for_timeout(600)

        # removing a breadcrumb step clears that filter
        pg.locator("#trail .trail-step").nth(1).locator(".trail-x").click()   # the artist step
        pg.wait_for_timeout(300)
        after = pg.evaluate(f"{G}.scope")
        print("after removing the artist step:", after["filters"], "| trail", pg.evaluate(f"{G}.trail"))
        assert "artist" not in after["filters"] and after["filters"].get("source") == ["aic"]
        b.close()
    print("errors:", errs or "none")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
