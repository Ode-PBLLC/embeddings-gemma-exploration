"""Ad hoc check of the map view on the live server: flatten on zoom, push apart, thumbnails, click to open,
zoom back out to 3D, and a query from the map. Saves screenshots to reports/screens/map_*.png."""
import sys

from playwright.sync_api import sync_playwright

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
OUT = "reports/screens/"
fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
    pg.goto("http://localhost:8000")
    pg.wait_for_function("window.__gallery && window.__gallery.ready", timeout=60000)
    G = "window.__gallery"

    # 1. click a dot in the 3D cloud: find an on-screen AIC dot, click it
    pt = pg.evaluate(f"""() => {{ for (let i = 0; i < {G}.pointCount; i += 7) {{
        const s = {G}.screenOf(i); if (s.x > 500 && s.x < 1100 && s.y > 250 && s.y < 650 && {G}.pick(s.x, s.y) === i) return {{i, x: s.x, y: s.y}}; }} return null; }}""")
    check(pt is not None, "found a pickable dot in 3D")
    pg.mouse.move(pt["x"], pt["y"]); pg.wait_for_timeout(200)
    check(pg.locator("#dot-tip").evaluate("e => e.classList.contains('is-shown')"), "hover shows a name tag")
    pg.wait_for_timeout(600)  # orbit has paused under the pointer; re-read where the dot settled
    s = pg.evaluate(f"{G}.screenOf({pt['i']})")
    pg.mouse.move(s["x"], s["y"]); pg.wait_for_timeout(300)
    s2 = pg.evaluate(f"{G}.screenOf({pt['i']})")
    check(abs(s2["x"] - s["x"]) < 1 and abs(s2["y"] - s["y"]) < 1, "orbit holds still while hovering a dot")
    want = pg.evaluate(f"{G}.pick({s2['x']}, {s2['y']})")
    pg.mouse.click(s2["x"], s2["y"]); pg.wait_for_timeout(700)
    check(want is not None and pg.evaluate(f"{G}.detailIndex") == want, "click opens the placard of the dot under the pointer")
    pg.screenshot(path=OUT + "map_0_click3d.png")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
    check(pg.evaluate(f"{G}.detailIndex") is None, "Esc closes the placard")

    # 2. zoom toward a spot until the cloud flattens
    pg.mouse.move(pt["x"], pt["y"])
    for _ in range(60):
        pg.mouse.wheel(0, -240); pg.wait_for_timeout(40)
        if pg.evaluate(f"{G}.view") != "3d":
            break
    check(pg.evaluate(f"{G}.view") != "3d", f"zooming in flattens the cloud (dist={pg.evaluate(f'{G}.cameraDist'):.2f})")
    pg.wait_for_timeout(900)
    check(abs(pg.evaluate(f"{G}.morph") - 1) < 1e-3, "morph completes to flat")
    pg.screenshot(path=OUT + "map_1_flat.png")

    # 3. keep zooming: dots push apart, then thumbnails appear
    for _ in range(40):
        pg.mouse.wheel(0, -200); pg.wait_for_timeout(50)
        if pg.evaluate(f"{G}.cameraDist") < 3.4:
            break
    pg.wait_for_timeout(900)
    pg.screenshot(path=OUT + "map_2_spread.png")
    for _ in range(40):
        pg.mouse.wheel(0, -200); pg.wait_for_timeout(50)
        if pg.evaluate(f"{G}.cameraDist") < 1.6:
            break
    pg.wait_for_timeout(1500)
    n_thumbs = pg.evaluate(f"{G}.mapThumbCount")
    check(n_thumbs > 0, f"thumbnails appear up close ({n_thumbs})")
    pg.screenshot(path=OUT + "map_3_thumbs.png")
    # overlap check among on-screen dots and thumbnails (drawn positions)
    min_gap = pg.evaluate(f"""() => {{ const pts = [];
        for (let i = 0; i < {G}.pointCount; i++) {{ const s = {G}.screenOf(i);
          if (s.x > 100 && s.x < 1500 && s.y > 100 && s.y < 700 && s.z < 1 && window.__cloud.away[i] < 0.5) pts.push(s); }}
        let m = Infinity; for (let a = 0; a < pts.length; a++) for (let c = a + 1; c < pts.length; c++)
          m = Math.min(m, Math.hypot(pts[a].x - pts[c].x, pts[a].y - pts[c].y)); return [pts.length, m]; }}""")
    check(min_gap[1] > 20, f"on-screen items do not pile up: {min_gap[0]} items, closest pair {min_gap[1]:.1f}px apart")

    # 3b. zooming on a pane never rearranges it (Mason: "It still reforms on every zoom change")
    snap = "() => { const c = window.__cloud; return c.pane.members.map(i => [c.paneGoal[i*3], c.paneGoal[i*3+1], c.paneGoal[i*3+2]].map(v => +v.toFixed(5))); }"
    before_zoom = pg.evaluate(snap)
    pg.mouse.move(820, 430)
    for dy in [-150] * 5 + [150] * 3:
        pg.mouse.wheel(0, dy); pg.wait_for_timeout(120)
    pg.wait_for_timeout(600)
    after_zoom = pg.evaluate(snap)
    check(before_zoom == after_zoom, f"zooming in and out leaves every work on the pane where it hangs ({len(before_zoom)} works)")

    # 4. drag to pan in the map
    before = pg.evaluate(f"{G}.camera")
    pg.mouse.move(800, 450); pg.mouse.down(); pg.mouse.move(600, 380, steps=10); pg.mouse.up()
    pg.wait_for_timeout(500)
    check(pg.evaluate(f"{G}.camera") != before, "left-drag pans the map")

    # 5. click a thumbnail
    hit = pg.evaluate(f"""() => {{ for (let i = 0; i < {G}.pointCount; i++) {{ const s = {G}.screenOf(i);
        if (s.x > 20 && s.x < 1580 && s.y > 20 && s.y < 760 && {G}.pick(s.x, s.y) === i) return {{i, x: s.x, y: s.y}}; }} return null; }}""")
    check(hit is not None, "found a clickable item in the map")
    pg.mouse.click(hit["x"], hit["y"]); pg.wait_for_timeout(900)
    check(pg.evaluate(f"{G}.detailIndex") == hit["i"], "clicking an item in the map opens its placard")
    pg.screenshot(path=OUT + "map_4_detail.png")
    pg.click("#detail .detail-close"); pg.wait_for_timeout(300)

    # 6. a query from the map rounds back to 3D and runs the fly-in
    pg.fill("#q", "light on water"); pg.press("#q", "Enter")
    pg.wait_for_function(f"{G}.phase === 'wall'", timeout=30000)
    check(pg.evaluate(f"{G}.view") == "3d", "query from the map returns to the 3D cloud")
    check(pg.evaluate(f"{G}.lineCount") == 12, "and draws 12 lines")
    pg.screenshot(path=OUT + "map_5_query.png")
    pg.click("#close-wall"); pg.wait_for_timeout(1300)

    # 7. zoom back out of a map to 3D
    pg.mouse.move(800, 400)
    for _ in range(60):
        pg.mouse.wheel(0, -240); pg.wait_for_timeout(40)
        if pg.evaluate(f"{G}.view") != "3d":
            break
    pg.wait_for_timeout(800)
    for _ in range(60):
        pg.mouse.wheel(0, 300); pg.wait_for_timeout(40)
        if pg.evaluate(f"{G}.view") == "3d":
            break
    pg.wait_for_timeout(900)
    check(pg.evaluate(f"{G}.view") == "3d", "zooming out rounds the map back into 3D")
    check(pg.evaluate(f"{G}.mapThumbCount") == 0, "map thumbnails are cleared")
    check(not errs, f"no page errors {errs[:3]}")
    b.close()

sys.exit(1 if fails else 0)
