"""Ad hoc (Mason: zooming further into the pane, the pictures "remain small and move out of view"):
zoom onto a coin, then keep zooming; report the pictures' height on screen and how many are in view.
    uv run python test_scripts/label_zoom_check.py [http://localhost:8010]      (changes no labels)"""
import json, sys
from playwright.sync_api import sync_playwright
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8010"
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
items = json.load(open("web/data/items.json"))
coins = [i for i, x in enumerate(items) if "(Coin)" in x["title"] or x["title"].startswith("Coin ")]
MEASURE = """() => { const c = window.__cloud, H = innerHeight, W = innerWidth, t = Math.tan(c.camera.fov * Math.PI / 360);
  const hs = [], inView = [];
  for (const [i, r] of c.mapSprites) { if (!r.sprite.material.map) continue; const h = r.sprite.scale.y * H / (2 * t); hs.push(h);
    const p = c.project(r.sprite.position); if (p.x > 400 && p.x < W && p.y > 0 && p.y < H) inView.push(i); }
  hs.sort((a, b) => a - b);
  return { dist: +c.camera.position.distanceTo(c.controls.target).toFixed(2), view: c.view, hung: c.mapSprites.size,
           inView: inView.length, medianHeightPx: hs.length ? Math.round(hs[hs.length >> 1]) : 0 }; }"""
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL + "/label/"); pg.wait_for_function("window.__label && window.__label.ready", timeout=120000)
    pt = pg.evaluate(f"() => {{ const s = {coins}.map((i) => window.__label.screenOf(i)); s.sort((a, b) => a.x - b.x); return s[s.length >> 1]; }}")
    pg.mouse.move(pt["x"], pt["y"])
    shot = 0
    for stop in (2.0, 1.2, 0.7, 0.4):
        for _ in range(200):
            pg.mouse.wheel(0, -120); pg.wait_for_timeout(30)
            if pg.evaluate("window.__cloud.view") == "flat" and pg.evaluate("window.__cloud.camera.position.distanceTo(window.__cloud.controls.target)") < stop: break
        pg.wait_for_timeout(1500)
        print(pg.evaluate(MEASURE))
        shot += 1; pg.screenshot(path=f"reports/screens/label/zoom_{shot}.png")
    b.close()
print("errors:", errs or "none")
