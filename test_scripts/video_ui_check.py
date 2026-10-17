"""Ad hoc: clips autoplay on the wall, in the 3D cloud, in the placard and on the page; Video scope works."""
import sys
from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"]
G = "window.__gallery"
fails = []
def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: fails.append(m)
PLAYING = """(sel) => [...document.querySelectorAll(sel)].map(v => ({ t0: v.currentTime, paused: v.paused, ready: v.readyState }))"""
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.fill("#q", "fire and heat"); pg.press("#q", "Enter")
    pg.wait_for_function(f"{G}.phase === 'wall' && {G}.lastResult.query === 'fire and heat' && {G}.lineCount === 12", timeout=30000)
    n_vid = pg.evaluate("document.querySelectorAll('#hang .piece--video video').length")
    pg.wait_for_timeout(2500)
    a = pg.evaluate(PLAYING, "#hang .piece--video video")
    pg.wait_for_timeout(1200)
    bb = pg.evaluate(PLAYING, "#hang .piece--video video")
    advancing = sum(1 for x, y in zip(a, bb) if not y["paused"] and y["t0"] != x["t0"])
    check(n_vid > 0 and advancing == n_vid, f"wall: {n_vid} clips hang on the wall, {advancing} are playing")
    live = pg.evaluate("window.__cloud.live.size")
    hung_vids = pg.evaluate("[...window.__cloud.billboards.keys()].filter(i => window.__cloud.items[i].modality === 'video').length")
    lv = pg.evaluate("[...window.__cloud.live.values()].map(r => [r.video.currentTime, r.video.paused])")
    pg.wait_for_timeout(1200)
    lv2 = pg.evaluate("[...window.__cloud.live.values()].map(r => [r.video.currentTime, r.video.paused])")
    moving = sum(1 for x, y in zip(lv, lv2) if not y[1] and y[0] != x[0])
    check(hung_vids > 0 and live == hung_vids and moving == live, f"cloud: {hung_vids} clips hung around the query, {live} live, {moving} advancing")
    pg.screenshot(path="reports/screens/video_wall_fire.png")
    # placard
    idx = pg.evaluate("[...window.__cloud.billboards.keys()].find(i => window.__cloud.items[i].modality === 'video')")
    s = pg.evaluate(f"window.__cloud.project(window.__cloud.billboards.get({idx}).sprite.position)")
    pg.mouse.move(s["x"], s["y"]); pg.wait_for_timeout(300); pg.mouse.click(s["x"], s["y"]); pg.wait_for_timeout(1500)
    pv = pg.evaluate(PLAYING, "#detail video")
    pg.wait_for_timeout(1000)
    pv2 = pg.evaluate(PLAYING, "#detail video")
    check(pg.evaluate(f"{G}.detailIndex") == idx and pv and not pv2[0]["paused"] and pv2[0]["t0"] != pv[0]["t0"],
          f"placard: clicked clip opens and plays ({pv2})")
    pg.screenshot(path="reports/screens/video_placard.png")
    pg.keyboard.press("Escape")
    # page view
    pg.click("#open-page"); pg.wait_for_timeout(2000)
    pp = pg.evaluate(PLAYING, "#page-hang .piece--video video")
    pg.wait_for_timeout(1000)
    pp2 = pg.evaluate(PLAYING, "#page-hang .piece--video video")
    check(pp and all(not y["paused"] and y["t0"] != x["t0"] for x, y in zip(pp, pp2)), f"page: {len(pp)} clips playing")
    pg.screenshot(path="reports/screens/video_page.png")
    pg.click("#back-to-cloud"); pg.wait_for_timeout(500)
    # scope: video only
    pg.click("#scope-btn"); pg.click("#scope [data-modality='video']")
    check(pg.evaluate(f"{G}.scope")["count"] == 240, f"Video scope: {pg.evaluate(f'{G}.scope')}")
    pg.keyboard.press("Escape")
    pg.fill("#q", "the northern lights"); pg.press("#q", "Enter")
    pg.wait_for_function(f"{G}.phase === 'wall' && {G}.lastResult.query === 'the northern lights'", timeout=30000)
    ids = [n["id"] for n in pg.evaluate(f"{G}.lastResult")["neighbors"]]
    kinds = pg.evaluate(f"{G}.lastResult.neighbors.map(n => window.__cloud.items[n.index].modality)")
    check(set(kinds) == {"video"}, f"Video scope returns only clips: {len(ids)} works, kinds {set(kinds)}")
    pg.wait_for_timeout(2000)
    pg.screenshot(path="reports/screens/video_scope_aurora.png")
    check(not errs, f"no page errors {errs[:3]}")
    b.close()
sys.exit(1 if fails else 0)
