"""Ad hoc: zoom with the pointer held on a front-of-cloud dot away from centre. The map should form
there and that work should become a thumbnail near the pointer (Mason: "I can scroll right past some dots")."""
import sys
from playwright.sync_api import sync_playwright

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
results = []
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    for trial, region in enumerate([(350, 650, 200, 450), (950, 1250, 200, 450), (400, 700, 450, 650), (900, 1200, 450, 650)]):
        pg = b.new_page(viewport={"width": 1600, "height": 900})
        pg.goto("http://localhost:8000")
        pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
        pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up()  # stop the idle orbit
        pg.wait_for_timeout(300)
        x0, x1, y0, y1 = region
        # the dot nearest the camera inside the region: the case that used to fly past
        pt = pg.evaluate(f"""() => {{ let best = null;
            const cam = {G}.camera;
            for (let i = 0; i < {G}.pointCount; i++) {{ const s = {G}.screenOf(i);
              if (s.x > {x0} && s.x < {x1} && s.y > {y0} && s.y < {y1} && {G}.pick(s.x, s.y) === i && (!best || s.z < best.z)) best = {{i, x: s.x, y: s.y, z: s.z}}; }}
            return best; }}""")
        if not pt:
            results.append((trial, "no dot in region", False)); pg.close(); continue
        pg.mouse.move(pt["x"], pt["y"])
        for _ in range(120):
            pg.mouse.wheel(0, -160); pg.wait_for_timeout(35)
            if pg.evaluate(f"{G}.view") != "3d" and pg.evaluate(f"{G}.cameraDist") < 1.5:
                break
        pg.wait_for_timeout(1500)
        view = pg.evaluate(f"{G}.view")
        thumbs = pg.evaluate(f"{G}.mapThumbIndices")
        s = pg.evaluate(f"{G}.screenOf({pt['i']})")
        off = ((s["x"] - pt["x"]) ** 2 + (s["y"] - pt["y"]) ** 2) ** 0.5
        # a work must stay under the pointer (the aimed one, or a neighbour pinned in its place)
        under = pg.evaluate(f"{G}.pick({pt['x']}, {pt['y']})")
        ok = view == "flat" and pt["i"] in thumbs and under is not None and under in thumbs
        results.append((trial, f"view={view} thumbs={len(thumbs)} aimed-is-thumb={pt['i'] in thumbs} aimed-drift={off:.0f}px work-under-pointer={under is not None}", ok))
        pg.screenshot(path=f"reports/screens/aim_{trial}.png")
        pg.close()
    b.close()
for r in results:
    print("PASS" if r[2] else "FAIL", r[0], r[1])
sys.exit(0 if all(r[2] for r in results) else 1)
