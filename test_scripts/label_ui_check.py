"""Ad hoc: drive the labelling page in a real browser on the real data, coins first.
    uv run python test_scripts/label_ui_check.py [http://localhost:8011] [--no-split] [--reset]
Use a server started with GALLERY_LABELS_DIR pointing at a scratch folder: this run saves labels.
It stops if the server already holds labels, unless --reset is given (which saves an empty state first).
Draw around the coins (Shift-drag, freehand) -> name them -> review suggestions -> sub-cluster with the slider ->
split by words -> reload. Prints what happened at each step; screenshots go to reports/screens/label/."""
import json
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

URL = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8011")
SPLIT = "--no-split" not in sys.argv
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
L = "window.__label"
SHOTS = "reports/screens/label"
items = json.load(open("web/data/items.json"))
coins = [i for i, x in enumerate(items) if "(Coin)" in x["title"] or x["title"].startswith("Coin ")]
is_coin = set(coins)
problems = []


def state(pg):
    return pg.evaluate(f"{L}.state")


def say(step, text):
    print(f"[{step}] {text}", flush=True)


def settle(pg, ms=400):
    pg.wait_for_function(f"!{L}.flying && {L}.view !== 'morphing'", timeout=20000)
    pg.wait_for_timeout(ms)


with sync_playwright() as p:
    if "--reset" in sys.argv:
        req = urllib.request.Request(URL + "/label/api/state", method="PUT", headers={"content-type": "application/json"},
                                     data=json.dumps({"nodes": [], "labels": [], "pending": [], "rejected": []}).encode())
        urllib.request.urlopen(req).read()
    b = p.chromium.launch(args=ARGS)
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.on("console", lambda m: problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    pg.on("response", lambda r: problems.append(f"HTTP {r.status} {r.url}") if r.status >= 400 and r.status != 503 else None)
    pg.goto(URL + "/label/")
    pg.wait_for_function(f"{L} && {L}.ready", timeout=120000)
    n0 = len(state(pg)["labels"])
    say("open", f"page ready; {n0} labels loaded; tree says: {pg.inner_text('#holdings')}")
    if n0:
        sys.exit("this server already has labels saved; point the check at a server with an empty GALLERY_LABELS_DIR")
    pg.screenshot(path=f"{SHOTS}/1_open.png")

    # --- draw a freehand loop around the coins in the 3D view: an uneven outline, wider than the coins ---
    import math
    pts = pg.evaluate(f"{json.dumps(coins)}.map((i) => {L}.screenOf(i))")
    xs, ys = sorted(q["x"] for q in pts), sorted(q["y"] for q in pts)
    cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
    d = sorted(((q["x"] - cx) ** 2 + (q["y"] - cy) ** 2) ** 0.5 for q in pts)
    r = d[int(len(d) * 0.9)] + 6
    loop = [(cx + r * (1 + 0.25 * math.sin(3 * t)) * math.cos(t), cy + r * (1 + 0.25 * math.sin(3 * t)) * 0.9 * math.sin(t))
            for t in [2 * math.pi * k / 40 for k in range(39)]]
    say("lasso", f"coins sit around ({cx:.0f}, {cy:.0f}) on screen; drawing a wobbly loop about {r:.0f}px across with Shift held")
    pg.keyboard.down("Shift")
    pg.mouse.move(*loop[0]); pg.mouse.down()
    for q in loop[1:]:
        pg.mouse.move(*q, steps=2)
    pg.screenshot(path=f"{SHOTS}/2_lasso.png")
    pg.mouse.up(); pg.keyboard.up("Shift")
    pg.wait_for_selector("#label-name", timeout=20000)
    sel = pg.evaluate("""() => { const c = [...document.querySelectorAll('#tray-body .card')];
      return { title: document.querySelector('.tray-title').textContent, text: document.querySelector('.tray-text').textContent,
               cards: c.map((e) => [e.querySelector('.card-name').textContent, e.querySelector('.card-count').textContent, e.querySelector('input').checked]),
               hint: (document.querySelector('#tray-body > .card-words') || {}).textContent, placeholder: document.querySelector('#label-name').placeholder }; }""")
    say("check", json.dumps(sel, ensure_ascii=False))
    pg.screenshot(path=f"{SHOTS}/3_selection.png")
    pg.fill("#label-name", "Coins")
    pg.click("#tray-body .btn.is-primary")
    pg.wait_for_function(f"{L}.state.labels.length > 0")
    pg.wait_for_function(f"{L}.tray === 'review'", timeout=20000)
    s = state(pg)
    lab = [x["i"] for x in s["labels"]]
    say("label", f"'Coins' holds {len(lab)} works, {sum(i in is_coin for i in lab)} of them coins by title; "
        f"sources {sorted(set((x['source'], x['method'], x['review']) for x in s['labels']))}")
    pend = s["pending"]
    say("suggest", f"{len(pend)} suggestions waiting; {sum(x['i'] in is_coin for x in pend)} are coins by title; tray: {pg.evaluate(f'{L}.tray')}")
    for x in sorted(pend, key=lambda x: -x["confidence"])[:14]:
        print(f"      {x['confidence']:.2f} {x['votes']}/{x['of']} {'COIN' if x['i'] in is_coin else '    '} {items[x['i']]['title'][:60]}")
    settle(pg)
    pg.screenshot(path=f"{SHOTS}/4_review.png")

    # --- review: one yes by hand, one no, then accept the rest at 60% or more ---
    pg.click("#tray-body .review-row >> nth=0 >> .yes")
    last = pg.locator("#tray-body .review-row").last
    last_title = last.locator(".review-title").inner_text()
    last.locator(".no").click()
    bulk = pg.locator("#tray-body .tray-actions .btn", has_text="60% or more")
    if bulk.count():
        bulk.click()
    s = state(pg)
    lab = [x["i"] for x in s["labels"]]
    say("review", f"said no to '{last_title}'; now {len(lab)} labelled ({sum(i in is_coin for i in lab)} coins of {len(coins)}), "
        f"{len(s['pending'])} waiting, {len(s['rejected'])} rejected; reviews: {sorted(set(x['review'] for x in s['labels']))}")
    pg.click("#tray-close")

    # --- sub-cluster with the granularity slider ---
    pg.click("#node-actions .btn:has-text('Sub-cluster')")
    pg.wait_for_selector("#granularity", timeout=30000)
    pg.wait_for_function("[...document.querySelectorAll('#tray-body .card-head input[type=text]')].every((e) => e.value)", timeout=20000)
    k0 = pg.input_value("#granularity")
    for k in (2, 5, 3):
        pg.evaluate(f"() => {{ const s = document.querySelector('#granularity'); s.value = {k}; s.dispatchEvent(new Event('input', {{ bubbles: true }})); }}")
        pg.wait_for_function(f"document.querySelectorAll('#tray-body .card').length === {k}")
        pg.wait_for_function("[...document.querySelectorAll('#tray-body .card-head input[type=text]')].every((e) => e.value)", timeout=20000)
        cards = pg.evaluate("() => [...document.querySelectorAll('#tray-body .card')].map((e) => [e.querySelector('input[type=text]').value, e.querySelector('.card-count').textContent, e.querySelector('.card-words').textContent])")
        say("cluster", f"k={k} (tree's own suggestion was {k0}): {cards}")
    settle(pg, 900)
    pg.screenshot(path=f"{SHOTS}/5_cluster.png")
    pg.click("#tray-body .btn.is-primary")
    pg.wait_for_function(f"{L}.tray === null")
    s = state(pg)
    say("keep", "tree: " + ", ".join(f"{nd['name']}{' < ' + [p['name'] for p in s['nodes'] if p['id'] == nd['parent']][0] if nd['parent'] else ''}" for nd in s["nodes"]))
    pg.wait_for_timeout(600)
    pg.screenshot(path=f"{SHOTS}/6_tree.png")

    # --- split by words (needs EmbeddingGemma; the server loads it in the background) ---
    if SPLIT:
        t0 = time.time()
        while json.load(urllib.request.urlopen(URL + "/label/api/health"))["model"] == "loading" and time.time() - t0 < 420:
            time.sleep(5)
        say("model", json.load(urllib.request.urlopen(URL + "/label/api/health"))["model"])
        pg.click("#node-actions .btn:has-text('Split by words')")
        pg.fill("#split-words", "gold coin, silver coin")
        pg.keyboard.press("Enter")
        pg.wait_for_selector("#split-margin", timeout=60000)
        cards = pg.evaluate("() => [...document.querySelectorAll('#tray-body .card')].map((e) => [(e.querySelector('input[type=text]') || e.querySelector('.card-name')).value || e.querySelector('.card-name').textContent, e.querySelector('.card-count').textContent])")
        say("split", f"{cards}")
        pg.wait_for_timeout(500)
        pg.screenshot(path=f"{SHOTS}/7_split.png")
        pg.click("#tray-body .btn.is-primary")
        pg.wait_for_function(f"{L}.tray === null")
        s = state(pg)
        gold = [nd for nd in s["nodes"] if nd["name"] == "Gold coin"][0]["id"]
        g = [x["i"] for x in s["labels"] if x["node"] == gold]
        metal = lambda i: [m for m in ("gold", "silver") if m in (items[i]["summary"] + items[i]["title"]).lower()]
        say("split", f"'Gold coin' has {len(g)} works; summaries that name one metal: {sum(metal(i) == ['gold'] for i in g)} gold, {sum(metal(i) == ['silver'] for i in g)} silver; "
            f"{sum(x['method'] == 'text-split' for x in s['pending'])} unsure works wait as suggestions")

    # --- the saved file, then a reload ---
    pg.wait_for_function("document.querySelector('#save-status').textContent.startsWith('Saved')", timeout=10000)
    pg.wait_for_timeout(900)
    say("save", pg.inner_text("#save-status") + " | " + pg.inner_text("#save-path"))
    before = state(pg)
    pg.reload()
    pg.wait_for_function(f"{L} && {L}.ready", timeout=120000)
    after = state(pg)
    say("reload", f"state identical after reload: {before == after}; {pg.inner_text('#holdings')}")
    # focus a label from the tree: the camera flies there and the dots take the sub-labels' colours
    pg.click("#tree .tree-row[data-node] >> nth=0")
    settle(pg, 2500)
    pg.screenshot(path=f"{SHOTS}/8_focus.png")
    say("focus", f"focus={pg.evaluate(f'{L}.focus') is not None}, view={pg.evaluate(f'{L}.view')}, pictures on the pane={pg.evaluate('window.__cloud.mapSprites.size')}")
    # --- on the pane, with Coins in focus: the Circle button, then a drag around a few pictures ---
    if pg.evaluate(f"{L}.view") == "flat":
        # an L-shaped outline: freehand selection need not be round
        pg.click("#tool-select")
        shape = [(700, 300), (1000, 300), (1000, 420), (820, 420), (820, 640), (700, 640)]
        pg.mouse.move(*shape[0]); pg.mouse.down()
        for q in shape[1:]:
            pg.mouse.move(*q, steps=12)
        pg.wait_for_timeout(200)
        pg.screenshot(path=f"{SHOTS}/9_pane_lasso.png")
        pg.mouse.up()
        pg.wait_for_selector("#label-name", timeout=20000)
        opts = pg.evaluate("() => [...document.querySelectorAll('#tray-body select option')].map((o) => o.textContent)")
        say("pane", f"L-shape on the pane -> {pg.inner_text('.tray-title')}; choices: {opts or '(only a new top-level label)'}; still on the pane: {pg.evaluate(f'{L}.view')}")
        pg.wait_for_timeout(900)
        pg.screenshot(path=f"{SHOTS}/10_pane_selection.png")
        pg.keyboard.press("Escape")
        pg.wait_for_function(f"{L}.tray === null")
    # --- everything not yet labelled, sorted into groups (the whole-corpus merge tree takes a few seconds) ---
    pg.click("#tree .tree-row.is-root")
    settle(pg, 1200)
    t0 = time.time()
    pg.click("#node-actions .btn:has-text('Sub-cluster')")
    pg.wait_for_selector("#granularity", timeout=60000)
    pg.evaluate("() => { const s = document.querySelector('#granularity'); s.value = 8; s.dispatchEvent(new Event('input', { bubbles: true })); }")
    pg.wait_for_function("document.querySelectorAll('#tray-body .card').length === 8 && [...document.querySelectorAll('#tray-body .card-head input[type=text]')].every((e) => e.value)", timeout=30000)
    cards = pg.evaluate("() => [...document.querySelectorAll('#tray-body .card')].map((e) => [e.querySelector('input[type=text]').value, e.querySelector('.card-count').textContent])")
    say("all", f"{pg.inner_text('.tray-title')} in {time.time() - t0:.1f}s: {cards}")
    settle(pg, 1500)
    pg.screenshot(path=f"{SHOTS}/11_all_works.png")
    pg.keyboard.press("Escape")
    b.close()

print("\nproblems:", problems or "none")
sys.exit(1 if problems else 0)
