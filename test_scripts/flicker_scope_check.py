"""Ad hoc: (1) no flicker once the map view settles: consecutive frames are pixel-identical;
(2) the scope control: Text only / one collection returns only matching works, deterministically."""
import io, sys
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
fails = []
def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: fails.append(m)

with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.goto("http://localhost:8000")
    pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(300)
    pt = pg.evaluate(f"""() => {{ for (let i = 0; i < {G}.pointCount; i += 3) {{ const s = {G}.screenOf(i);
        if (s.x > 600 && s.x < 1000 && s.y > 300 && s.y < 600 && {G}.pick(s.x, s.y) === i) return s; }} return null; }}""")
    pg.mouse.move(pt["x"], pt["y"])
    for _ in range(120):
        pg.mouse.wheel(0, -160); pg.wait_for_timeout(35)
        if pg.evaluate(f"{G}.view") == "flat" and pg.evaluate(f"{G}.cameraDist") < 1.5: break
    for label in ["thumbnails", "dots"]:
        if label == "dots":
            for _ in range(6): pg.mouse.wheel(0, 120); pg.wait_for_timeout(40)
        pg.mouse.move(5, 890)  # pointer off the works (no hover tag)
        # let every thumbnail finish loading and fading in before judging stability
        pg.wait_for_function("""() => { const c = window.__cloud; let m = 0;
            for (let i = 0; i < c.paneCur.length; i++) m = Math.max(m, Math.abs(c.paneGoal[i] - c.paneCur[i]));
            return m < 1e-5 && [...c.mapSprites.values()].every(r => r.sprite.material.map && r.k > 0.9999); }""", timeout=60000)
        pg.wait_for_timeout(500)
        shots = [Image.open(io.BytesIO(pg.screenshot())).convert("RGB") for _ in range(5) if not pg.wait_for_timeout(150)]
        # flicker = a visible change between settled frames; ignore sub-visible (<= 4/255) residue of fades
        worst = [max(ImageChops.difference(shots[0], s).getextrema(), key=lambda e: e[1])[1] for s in shots[1:]]
        check(max(worst) <= 4, f"map view ({label}, dist={pg.evaluate(f'{G}.cameraDist'):.2f}, thumbs={pg.evaluate(f'{G}.mapThumbCount')}): largest per-pixel change between settled frames {max(worst)}/255")
    # scope: open the control, choose Text + Wikipedia only... then NASA only
    pg.click("#scope-btn")
    check(pg.locator("#scope").is_visible(), "scope panel opens")
    pg.click("#scope [data-modality='text']")
    s = pg.evaluate(f"{G}.scope")
    check(s["filters"] == {"modality": ["text"]} and s["count"] == 1700, f"Text only -> {s}")
    pg.screenshot(path="reports/screens/scope_panel.png")
    pg.keyboard.press("Escape")
    pg.fill("#q", "the color blue"); pg.press("#q", "Enter")
    pg.wait_for_function(f"{G}.phase === 'wall' && {G}.lastResult.query === 'the color blue'", timeout=30000)
    r = pg.evaluate(f"{G}.lastResult")
    check(all(n["id"].startswith("wiki:") for n in r["neighbors"]) and len(r["neighbors"]) == 12, "Text only returns 12 text works")
    check("searched text" in pg.inner_text("#wall-sub"), f"wall states the scope: {pg.inner_text('#wall-sub')!r}")
    pg.wait_for_timeout(600)
    pg.screenshot(path="reports/screens/scope_text_wall.png")
    # NASA only, all types
    pg.click("#scope-btn")
    pg.click("#scope [data-modality='']")
    for src in ("aic", "wiki"): pg.click(f"#scope [data-source='{src}']")
    s = pg.evaluate(f"{G}.scope")
    check(s["filters"] == {"source": ["nasa"]} and s["label"] == "NASA only", f"NASA only -> {s}")
    pg.click("#scope [data-source='nasa']")  # cannot remove the last collection
    check(pg.evaluate(f"{G}.scope")["filters"] == {"source": ["nasa"]}, "the last collection cannot be switched off")
    pg.keyboard.press("Escape")
    ids = []
    for _ in range(2):
        pg.fill("#q", "loneliness"); pg.press("#q", "Enter")
        pg.wait_for_function(f"{G}.phase === 'wall' && {G}.lastResult.query === 'loneliness' && {G}.lineCount === 12", timeout=30000)
        ids.append([n["id"] for n in pg.evaluate(f"{G}.lastResult")["neighbors"]])
        pg.fill("#q", "x"); pg.press("#q", "Enter"); pg.wait_for_timeout(300)
    check(all(i.startswith("nasa:") for i in ids[0]), "NASA only returns NASA works")
    check(ids[0] == ids[1], "same query + scope gives the same works (deterministic)")
    # impossible scope: Text + NASA
    pg.click("#scope-btn"); pg.click("#scope [data-modality='text']")
    check("No works match" in pg.inner_text("#scope-count"), "impossible scope says so in the panel")
    pg.keyboard.press("Escape")
    pg.fill("#q", "loneliness"); pg.press("#q", "Enter"); pg.wait_for_timeout(1200)
    check("no works match" in pg.inner_text("#ask-error").lower(), f"and the search shows the error: {pg.inner_text('#ask-error')!r}")
    b.close()
sys.exit(1 if fails else 0)
