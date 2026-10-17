from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: errs.append(m.type + ": " + m.text))
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.mouse.click(800, 120)
    pg.fill("#q", "Shiva dancing"); pg.press("#q", "Enter")
    pg.wait_for_function(f"{G}.phase === 'wall' && {G}.lineCount === 12", timeout=30000); pg.wait_for_timeout(1000)
    card = pg.locator("#hang .piece--sound:has(.sound-words)").first
    card.scroll_into_view_if_needed(); card.hover(); pg.wait_for_timeout(3000)
    print(pg.evaluate("""() => { const p = [...document.querySelectorAll('#hang .piece--sound')].find(x => x.querySelector('.sound-words'));
      const a = p.sound.audio; return { src: a.currentSrc, t: a.currentTime, paused: a.paused, ready: a.readyState, net: a.networkState, err: a.error && a.error.code,
        cls: p.sound.root.className, now: (p.querySelector('.w.is-now') || {}).textContent }; }"""))
    print(errs[-6:])
    b.close()
