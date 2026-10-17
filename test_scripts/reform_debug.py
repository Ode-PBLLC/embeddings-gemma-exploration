from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
INFO = """() => { const c = window.__cloud, p = c.pane, {F, right, up} = c.paneBasis;
  const t = c.controls.target.clone().sub(F), cu = t.dot(right), cv = t.dot(up);
  let best = Infinity, bestChart = Infinity;
  for (let k = 0; k < p.members.length; k++) best = Math.min(best, Math.hypot(p.ruv[k*2]-cu, p.ruv[k*2+1]-cv));
  for (let i = 0; i < c.n; i++) bestChart = Math.min(bestChart, Math.hypot(c.chart[i*2]-cu, c.chart[i*2+1]-cv));
  const W = innerWidth, H = innerHeight; let vis = 0;
  for (const i of p.members) { const s = c.project(new c.camera.position.constructor(c.shown[i*3], c.shown[i*3+1], c.shown[i*3+2])); if (s.x > 0 && s.x < W && s.y > 0 && s.y < H) vis++; }
  const tanV = Math.tan(c.camera.fov * Math.PI / 360), d = c.camera.position.distanceTo(c.controls.target);
  return { reanchors: c.reanchors || 0, viewHalfW: +(d * tanV * W / H).toFixed(2), nearestMember: +best.toFixed(2), nearestAnyWork: +bestChart.toFixed(2), visibleMembers: vis, extent: p.extent }; }"""
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(300)
    pt = pg.evaluate(f"""() => {{ for (let i = 0; i < {G}.pointCount; i += 3) {{ const s = {G}.screenOf(i);
        if (s.x > 600 && s.x < 1000 && s.y > 300 && s.y < 600 && {G}.pick(s.x, s.y) === i) return s; }} return null; }}""")
    pg.mouse.move(pt["x"], pt["y"])
    for _ in range(140):
        pg.mouse.wheel(0, -160); pg.wait_for_timeout(35)
        if pg.evaluate(f"{G}.view") == "flat" and pg.evaluate(f"{G}.cameraDist") < 1.5: break
    pg.wait_for_timeout(1500)
    print("start", pg.evaluate(INFO))
    for k in range(6):
        pg.mouse.move(1300, 450); pg.mouse.down(); pg.mouse.move(300, 450, steps=15); pg.mouse.up(); pg.wait_for_timeout(1500)
        print("pan", k, pg.evaluate(INFO))
    b.close()
