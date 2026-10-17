"""Ad hoc: listening cards look distinct, play on hover, highlight words in time, show captions when
wordless, play in the placard, and the Sound scope works."""
import sys
from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
fails = []
def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: fails.append(m)
def ask(pg, q):
    pg.fill("#q", q); pg.press("#q", "Enter")
    pg.wait_for_function(f"{G}.phase === 'wall' && {G}.lastResult.query === {q!r} && {G}.lineCount === 12", timeout=30000)
    pg.wait_for_timeout(1200)
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.mouse.click(800, 120)   # a user gesture, so the browser allows sound
    ask(pg, "Shiva dancing")
    cards = pg.locator("#hang .piece--sound")
    n = cards.count()
    check(n >= 1, f"wall shows {n} listening cards")
    bg = pg.evaluate("getComputedStyle(document.querySelector('#hang .piece--sound .sound-card')).backgroundColor")
    tbg = pg.evaluate("(() => { const c = document.querySelector('#hang .piece--text .card'); return c ? getComputedStyle(c).backgroundColor : null; })()")
    check(bg != tbg, f"listening card ({bg}) looks unlike a text card ({tbg})")
    speech = pg.locator("#hang .piece--sound:has(.sound-words)").first
    speech.scroll_into_view_if_needed(); speech.hover(); pg.wait_for_timeout(2500)
    st = pg.evaluate("""() => { const p = [...document.querySelectorAll('#hang .piece--sound')].find(x => x.querySelector('.sound-words'));
        const a = p.sound.audio; return { t: a.currentTime, paused: a.paused, now: p.querySelectorAll('.w.is-now').length, said: p.querySelectorAll('.w.is-said').length,
          nowWord: (p.querySelector('.w.is-now') || {}).textContent }; }""")
    pg.wait_for_timeout(2000)
    st2 = pg.evaluate("""() => { const p = [...document.querySelectorAll('#hang .piece--sound')].find(x => x.querySelector('.sound-words'));
        return { t: p.sound.audio.currentTime, said: p.querySelectorAll('.w.is-said').length }; }""")
    check(not st["paused"] and st2["t"] > st["t"], f"hovering a guide plays it ({st['t']:.1f}s -> {st2['t']:.1f}s)")
    check(st["now"] == 1 and st2["said"] > st["said"], f"karaoke: one current word ({st['nowWord']!r}), spoken words grow {st['said']} -> {st2['said']}")
    pg.screenshot(path="reports/screens/sound_wall_shiva.png")
    pg.mouse.move(400, 860); pg.wait_for_timeout(500)
    check(pg.evaluate("[...document.querySelectorAll('#hang .piece--sound')].every(p => p.sound.audio.paused)"), "moving away pauses it")
    ask(pg, "sounds from Mars")
    capn = pg.locator("#hang .piece--sound .sound-caption")
    check(capn.count() >= 1, f"wordless sounds show a static caption: {[capn.nth(i).inner_text() for i in range(min(3, capn.count()))]}")
    pg.screenshot(path="reports/screens/sound_wall_mars.png")
    # placard via the cloud billboard of a sound
    idx = pg.evaluate("[...window.__cloud.billboards.keys()].find(i => window.__cloud.items[i].modality === 'audio')")
    check(idx is not None, "sounds hang in the cloud as listening glyphs")
    s = pg.evaluate(f"window.__cloud.project(window.__cloud.billboards.get({idx}).sprite.position)")
    pg.mouse.move(s["x"], s["y"]); pg.wait_for_timeout(300); pg.mouse.click(s["x"], s["y"]); pg.wait_for_timeout(2500)
    pl = pg.evaluate("(() => { const c = document.querySelector('#detail .sound-card'); return c ? { playing: c.classList.contains('is-playing') } : null; })()")
    check(pl and pl["playing"] and pg.evaluate(f"{G}.detailIndex") == idx, f"placard opens and plays the sound ({pl})")
    pg.screenshot(path="reports/screens/sound_placard.png")
    pg.keyboard.press("Escape")
    pg.click("#scope-btn"); pg.click("#scope [data-modality='audio']")
    check(pg.evaluate(f"{G}.scope")["count"] == 218, f"Sound scope: {pg.evaluate(f'{G}.scope')}")
    pg.keyboard.press("Escape")
    ask(pg, "a buddha statue")
    kinds = pg.evaluate(f"{G}.lastResult.neighbors.map(n => window.__cloud.items[n.index].modality)")
    check(set(kinds) == {"audio"}, f"Sound scope returns only sounds ({len(kinds)})")
    pg.screenshot(path="reports/screens/sound_scope_buddha.png")
    check(not errs, f"no page errors {errs[:3]}")
    b.close()
sys.exit(1 if fails else 0)
