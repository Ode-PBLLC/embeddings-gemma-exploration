from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(300)
    pg.mouse.move(800, 420)
    for _ in range(140):
        pg.mouse.wheel(0, -160); pg.wait_for_timeout(35)
        if pg.evaluate(f"{G}.view") == "flat" and pg.evaluate(f"{G}.cameraDist") < 1.5: break
    for t in range(4):
        pg.wait_for_timeout(3000)
        print(pg.evaluate("""() => { const c = window.__cloud; let m = 0;
          for (let i = 0; i < c.flatCur.length; i++) m = Math.max(m, Math.abs(c.flatGoal[i] - c.flatCur[i]));
          const sp = [...c.mapSprites.values()]; return { relaxMs: c.relaxMs, maxDelta: m, sprites: sp.length, noMap: sp.filter(r => !r.sprite.material.map).length, lowK: sp.filter(r => r.k <= 0.999).length, fps: c.medianFps(), dist: c.camera.position.distanceTo(c.controls.target) }; }"""))
    print("errors", errs)
    b.close()
