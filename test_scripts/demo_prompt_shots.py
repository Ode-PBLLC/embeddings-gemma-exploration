from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    import sys
    for q in sys.argv[1:]:
        pg.fill("#q", q); pg.press("#q", "Enter")
        pg.wait_for_function(f"{G}.phase === 'wall' && {G}.lastResult.query === {q!r} && {G}.lineCount === 12", timeout=30000)
        pg.wait_for_timeout(3000)
        pg.screenshot(path=f"reports/screens/demo_pick_{q.replace(' ', '_')}.png")
    b.close()
