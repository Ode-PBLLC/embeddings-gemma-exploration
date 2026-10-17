"""Ad hoc (Mason, 2026-10-16): after thumbnails show, panning must not rearrange anything, and every
work in view should be a thumbnail (no stray dots beside them)."""
import sys
from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
SETTLED = """() => { const c = window.__cloud; let m = 0;
  for (let i = 0; i < c.paneCur.length; i++) m = Math.max(m, Math.abs(c.paneGoal[i] - c.paneCur[i]));
  return m < 1e-4 && [...c.mapSprites.values()].every(r => r.sprite.material.map && r.k > 0.999); }"""
STATE = """() => { const c = window.__cloud; const W = innerWidth, H = innerHeight; let dots = 0, items = [];
  for (let i = 0; i < c.n; i++) { const s = c.project(new c.camera.position.constructor(c.shown[i*3], c.shown[i*3+1], c.shown[i*3+2]));
    if (s.x < 30 || s.x > W - 30 || s.y < 30 || s.y > H - 30 || c.away[i] > 0.5) continue;
    items.push([i, s.x, s.y]); if (!c.mapSprites.has(i)) dots++; }
  let gap = Infinity; for (let a = 0; a < items.length; a++) for (let b = a + 1; b < items.length; b++) gap = Math.min(gap, Math.hypot(items[a][1]-items[b][1], items[a][2]-items[b][2]));
  const world = {}; for (const [i] of items) world[i] = [c.shown[i*3], c.shown[i*3+1], c.shown[i*3+2]].map(v => +v.toFixed(5));
  return { reanchors: c.reanchors || 0, members: c.pane ? c.pane.members.length : 0, inView: items.length, dots, thumbs: c.mapSprites.size, gap, spacing: c.minSpacingPx, relaxMs: c.relaxMs, world }; }"""
fails = []
DIAG = """() => { const c = window.__cloud; let m = 0, mi = -1;
  for (let i = 0; i < c.paneCur.length; i++) { const d = Math.abs(c.paneGoal[i] - c.paneCur[i]); if (d > m) { m = d; mi = i; } }
  const sp = [...c.mapSprites.values()];
  const bad = [...c.mapSprites.entries()].filter(([i, r]) => !r.sprite.material.map).slice(0, 5).map(([i]) => { const e = c.texCache.get(i); return { i, modality: c.items[i].modality, thumb: c.items[i].thumb, cached: !!e, tex: !!(e && e.tex) }; });
  return { bad, maxDelta: m, at: mi, val: [c.paneGoal[mi], c.paneCur[mi]], noMap: sp.filter(r => !r.sprite.material.map).length, lowK: sp.filter(r => r.k <= 0.999).length, view: c.view }; }"""
def settle(pg):
    try:
        pg.wait_for_function(SETTLED, timeout=30000)
    except Exception:
        print("NOT SETTLED:", pg.evaluate(DIAG)); print("console:", errs[:8]); raise
def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: fails.append(m)
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    errs = []
    for trial, (x, y) in enumerate([(800, 420), (560, 300), (1050, 560)]):
        pg = b.new_page(viewport={"width": 1600, "height": 900})
        pg.on("console", lambda m: m.type in ("error", "warning") and errs.append(m.text))
        pg.on("requestfailed", lambda r: errs.append("REQFAIL " + r.url))
        pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
        pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(300)
        pt = pg.evaluate(f"""() => {{ let best = null, bd = 1e9; for (let i = 0; i < {G}.pointCount; i += 2) {{ const s = {G}.screenOf(i);
            const d = Math.hypot(s.x - {x}, s.y - {y}); if (d < bd && {G}.pick(s.x, s.y) === i) {{ bd = d; best = s; }} }} return best; }}""")
        pg.mouse.move(pt["x"], pt["y"])
        for _ in range(140):
            pg.mouse.wheel(0, -160); pg.wait_for_timeout(35)
            if pg.evaluate(f"{G}.view") == "flat" and pg.evaluate(f"{G}.cameraDist") < 1.5: break
        pg.mouse.move(5, 890); settle(pg)
        a = pg.evaluate(STATE)
        check(a["dots"] == 0, f"[{trial}] every work in view is a thumbnail: {a['inView']} in view, {a['dots']} still dots (relax {a['relaxMs']:.0f} ms)")
        check(a["gap"] > 0.85 * a["spacing"], f"[{trial}] pictures do not overlap: closest {a['gap']:.0f}px, spacing {a['spacing']:.0f}px")
        pg.screenshot(path=f"reports/screens/pan_{trial}_before.png")
        pg.mouse.move(900, 500); pg.mouse.down(); pg.mouse.move(600, 350, steps=12); pg.mouse.up()
        pg.mouse.move(5, 890); settle(pg)
        bb = pg.evaluate(STATE)
        common = set(a["world"]) & set(bb["world"])
        moved = [i for i in common if max(abs(p - q) for p, q in zip(a["world"][i], bb["world"][i])) > 1e-3]
        check(bb["reanchors"] == a["reanchors"], f"[{trial}] a short pan stays on the same pane (re-forms: {a['reanchors']} -> {bb['reanchors']})")
        check(common and not moved, f"[{trial}] panning moved none of the {len(common)} works seen before and after ({len(moved)} moved)")
        check(bb["dots"] == 0, f"[{trial}] after the pan every work in view is still a thumbnail ({bb['inView']} in view, {bb['dots']} dots)")
        pg.screenshot(path=f"reports/screens/pan_{trial}_after.png")
        # long pan toward the edge: the pane should re-form around the new spot
        for _ in range(2):
            pg.mouse.move(1300, 450); pg.mouse.down(); pg.mouse.move(300, 450, steps=15); pg.mouse.up()
        pg.mouse.move(5, 890); settle(pg)
        cc = pg.evaluate(STATE)
        # an isolated cluster legitimately has nothing past its edge; elsewhere the pane re-forms
        print(f"INFO [{trial}] long pan: {cc['reanchors']} re-forms, {cc['members']} works on the pane, {cc['inView']} in view")
        check(cc["dots"] == 0, f"[{trial}] after the long pan every work in view is a thumbnail ({cc['inView']} in view, {cc['dots']} dots)")
        pg.screenshot(path=f"reports/screens/pan_{trial}_reformed.png")
        pg.close()
    b.close()
sys.exit(1 if fails else 0)
