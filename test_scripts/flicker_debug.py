from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=120000)
    pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(300)
    pg.mouse.move(800, 450)
    for _ in range(120):
        pg.mouse.wheel(0, -160); pg.wait_for_timeout(35)
        if pg.evaluate(f"{G}.view") == "flat" and pg.evaluate(f"{G}.cameraDist") < 1.5: break
    pg.mouse.move(5, 890); pg.wait_for_timeout(2500)
    print(pg.evaluate("""() => new Promise(res => { const c = window.__cloud; const snap = () => {
        let maxGoal = 0; for (let i = 0; i < c.flatCur.length; i++) maxGoal = Math.max(maxGoal, Math.abs(c.flatGoal[i] - c.flatCur[i]));
        const sp = [...c.mapSprites.values()].map(r => [r.sprite.position.x, r.sprite.position.y, r.sprite.scale.x, r.sprite.material.opacity, r.k, r.sprite.renderOrder]);
        return { maxGoal, sp: JSON.stringify(sp), cam: c.camera.position.toArray().join(','), n: c.mapSprites.size, relax: !!c.relaxTimer }; };
      const a = snap(); setTimeout(() => { const b = snap(); res({ maxGoal: [a.maxGoal, b.maxGoal], spritesSame: a.sp === b.sp, camSame: a.cam === b.cam, n: [a.n, b.n] }); }, 400); })"""))
    b.close()
