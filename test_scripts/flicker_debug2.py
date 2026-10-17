import io
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(300)
    pt = pg.evaluate(f"""() => {{ for (let i = 0; i < {G}.pointCount; i += 3) {{ const s = {G}.screenOf(i);
        if (s.x > 600 && s.x < 1000 && s.y > 300 && s.y < 600 && {G}.pick(s.x, s.y) === i) return s; }} return null; }}""")
    pg.mouse.move(pt["x"], pt["y"])
    for _ in range(120):
        pg.mouse.wheel(0, -160); pg.wait_for_timeout(35)
        if pg.evaluate(f"{G}.view") == "flat" and pg.evaluate(f"{G}.cameraDist") < 1.5: break
    pg.mouse.move(5, 890)
    pg.wait_for_function("""() => { const c = window.__cloud; let m = 0;
        for (let i = 0; i < c.paneCur.length; i++) m = Math.max(m, Math.abs(c.paneGoal[i] - c.paneCur[i]));
        return m < 1e-5 && [...c.mapSprites.values()].every(r => r.sprite.material.map && r.k > 0.9999); }""", timeout=60000)
    st = pg.evaluate("""() => { const c = window.__cloud; return { view: c.view, morph: c.morph, dist: c.camera.position.distanceTo(c.controls.target), dt: c.points.material.depthTest, sprites: c.mapSprites.size, tweens: null }; }""")
    print(st)
    frames = [np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB")).astype(int) for _ in range(6)]
    Image.fromarray((np.abs(frames[1]-frames[0]).sum(-1) > 0).astype(np.uint8)*255).save("reports/screens/flicker_diff.png")
    Image.fromarray(frames[0].astype(np.uint8)).save("reports/screens/flicker_f0.png")
    for i in range(1, 6):
        d = np.abs(frames[i] - frames[0]).sum(-1)
        print(i, "pixels differing from frame0:", int((d > 0).sum()), "max", int(d.max()), "| same as frame", [j for j in range(i) if not np.abs(frames[i]-frames[j]).any()])
    print(pg.evaluate("""() => new Promise(res => { const c = window.__cloud;
      const snap = () => JSON.stringify([...c.mapSprites.entries()].map(([i, r]) => [i, r.sprite.position.toArray().map(v => v.toFixed(6)), r.sprite.scale.x.toFixed(6), r.sprite.material.opacity.toFixed(6), r.sprite.renderOrder, r.sprite.material.map && r.sprite.material.map.version]));
      const a = snap(); requestAnimationFrame(() => requestAnimationFrame(() => { const b = snap(); res({ spritesIdentical: a === b }); })); })"""))
    # what changes in JS between two rAFs?
    print(pg.evaluate("""() => new Promise(res => { const c = window.__cloud; const a = c.shown.slice(); const u = JSON.stringify(Object.fromEntries(Object.entries(c.uniforms).map(([k,v]) => [k, v.value && v.value.toArray ? v.value.toArray() : v.value])));
      requestAnimationFrame(() => requestAnimationFrame(() => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - c.shown[i]));
        const u2 = JSON.stringify(Object.fromEntries(Object.entries(c.uniforms).map(([k,v]) => [k, v.value && v.value.toArray ? v.value.toArray() : v.value])));
        res({ shownMaxDelta: m, uniformsSame: u === u2, u, u2 }); })); })"""))
    b.close()
