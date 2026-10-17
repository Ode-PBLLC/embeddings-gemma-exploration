from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(300)
    pt = pg.evaluate(f"""() => {{ for (let i = 0; i < {G}.pointCount; i += 3) {{ const s = {G}.screenOf(i);
        if (s.x > 600 && s.x < 1000 && s.y > 300 && s.y < 600 && {G}.pick(s.x, s.y) === i) return s; }} return null; }}""")
    pg.mouse.move(pt["x"], pt["y"])
    for _ in range(140):
        pg.mouse.wheel(0, -160); pg.wait_for_timeout(35)
        if pg.evaluate(f"{G}.view") == "flat" and pg.evaluate(f"{G}.cameraDist") < 1.5: break
    pg.wait_for_timeout(2000)
    print(pg.evaluate("""() => { const c = window.__cloud; const p = c.pane; const k0 = p.members[0];
      let nan = 0; for (let i = 0; i < c.shown.length; i++) if (!isFinite(c.shown[i])) nan++;
      return { scale: c.chartScale, extent: p.extent, center: p.center, uv0: [p.uv[0], p.uv[1]], ruv0: p.ruv && [p.ruv[0], p.ruv[1]], nan, target: c.controls.target.toArray(), F: c.paneBasis.F.toArray(), dist: c.camera.position.distanceTo(c.controls.target), dupes: (() => { let z = 0; for (let a = 0; a < 50; a++) for (let b2 = a+1; b2 < 50; b2++) { const i = p.members[a], j = p.members[b2]; if (Math.hypot(c.positions[i*3]-c.positions[j*3], c.positions[i*3+1]-c.positions[j*3+1], c.positions[i*3+2]-c.positions[j*3+2]) < 1e-6) z++; } return z; })() }; }"""))
    print(errs)
    b.close()
