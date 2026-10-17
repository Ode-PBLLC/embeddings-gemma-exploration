"""Ad hoc: every neighbour line must end on that neighbour's real dot, before and after turning the cloud."""
import sys
from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
JS = """() => { const c = window.__cloud; const out = [];
  for (const rec of c.lines) { const a = rec.line.geometry.attributes.instanceEnd.array;  // [x0,y0,z0,x1,y1,z1]
    const end = c.project(new c.queryPos.constructor(a[3], a[4], a[5]));
    const dot = c.project(c.worldPos(rec.index));
    const bb = c.billboards.get(rec.index); const pic = bb ? c.project(bb.sprite.position) : null;
    out.push({ i: rec.index, err: Math.hypot(end.x - dot.x, end.y - dot.y), pic: pic ? Math.hypot(pic.x - dot.x, pic.y - dot.y) : null }); }
  return out; }"""
fails = []
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.goto("http://localhost:8000")
    pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    for q in ["loneliness", "machines that fly"]:
        pg.fill("#q", q); pg.press("#q", "Enter")
        pg.wait_for_function(f"{G}.phase === 'wall' && {G}.lastResult.query === {q!r} && {G}.lineCount === 12", timeout=30000)
        pg.wait_for_timeout(800)
        for label in ["after fly-in", "after turning"]:
            r = pg.evaluate(JS)
            worst = max(x["err"] for x in r)
            print(f"{q} {label}: {len(r)} lines, worst line-end-to-dot {worst:.1f}px, picture-to-dot {min(x['pic'] for x in r):.0f}-{max(x['pic'] for x in r):.0f}px")
            if worst > 2 or len(r) != 12:
                fails.append((q, label, worst))
            pg.screenshot(path=f"reports/screens/lines_{q.replace(' ', '_')}_{label.replace(' ', '_')}.png")
            if label == "after fly-in":
                pg.mouse.move(300, 450); pg.mouse.down(); pg.mouse.move(520, 380, steps=20); pg.mouse.up()
                pg.wait_for_timeout(1200)
    b.close()
print("FAILS", fails)
sys.exit(1 if fails else 0)
