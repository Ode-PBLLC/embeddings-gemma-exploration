"""Ad hoc check: wheel-zoom toward the cursor and right-drag pan on the live page."""
from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
OUT = "reports/screens/"
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.goto("http://localhost:8000")
    pg.wait_for_function("window.__gallery && window.__gallery.ready", timeout=60000)
    cam = "() => { const c = window.__gallery.cloud || null; return null; }"
    pg.mouse.move(1000, 420)  # over the colourful art cluster, right of centre
    for _ in range(40):
        pg.mouse.wheel(0, -300); pg.wait_for_timeout(30)
    pg.wait_for_timeout(800)
    pg.screenshot(path=OUT + "zoom_deep.png")
    # right-drag pan
    pg.mouse.move(800, 450); pg.mouse.down(button="right"); pg.mouse.move(500, 300, steps=15); pg.mouse.up(button="right")
    pg.wait_for_timeout(800)
    pg.screenshot(path=OUT + "zoom_deep_panned.png")
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.wait_for_timeout(300)
    print("page errors:", errs)
    b.close()
