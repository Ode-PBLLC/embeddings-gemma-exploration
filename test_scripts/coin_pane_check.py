"""Ad hoc (Mason: "The coin cloud is pulling in far too many points from elsewhere in 3d space"):
zoom onto a coin; the pane should hold the coin cluster and not much else."""
import json, sys
from playwright.sync_api import sync_playwright
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
G = "window.__gallery"
items = json.load(open("web/data/items.json"))
coins = [i for i, x in enumerate(items) if "(Coin)" in x["title"] or x["title"].startswith("Coin ")]
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS); pg = b.new_page(viewport={"width": 1600, "height": 900})
    pg.goto("http://localhost:8000"); pg.wait_for_function(f"{G} && {G}.ready", timeout=60000)
    pg.mouse.move(5, 5); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(300)
    # the coin cluster in 3D: how far apart are coins vs the nearest non-coin works?
    stats = pg.evaluate(f"""() => {{ const c = window.__cloud, coins = new Set({coins});
      const P = (i) => [c.positions[i*3], c.positions[i*3+1], c.positions[i*3+2]];
      const cen = [0,0,0]; for (const i of coins) P(i).forEach((v, a) => cen[a] += v / coins.size);
      const dist = (i) => Math.hypot(...P(i).map((v, a) => v - cen[a]));
      const cd = [...coins].map(dist).sort((a, b) => a - b);
      const od = []; for (let i = 0; i < c.n; i++) if (!coins.has(i)) od.push(dist(i)); od.sort((a, b) => a - b);
      return {{ coinR80: cd[Math.floor(cd.length * .8)], coinMax: cd[cd.length - 1], nearestOther: od[0], others_within_coinR80: od.filter(d => d < cd[Math.floor(cd.length * .8)]).length }}; }}""")
    print("cluster:", stats)
    # zoom onto an on-screen coin
    pt = pg.evaluate(f"""() => {{ for (const i of {coins}) {{ const s = {G}.screenOf(i); if (s.x > 80 && s.x < 1500 && s.y > 80 && s.y < 700 && {G}.pick(s.x, s.y) === i) return {{ i, x: s.x, y: s.y }}; }} return null; }}""")
    if not pt:
        print("no pickable coin on screen"); sys.exit(1)
    pg.mouse.move(pt["x"], pt["y"])
    for _ in range(160):
        pg.mouse.wheel(0, -140); pg.wait_for_timeout(35)
        if pg.evaluate(f"{G}.view") == "flat" and pg.evaluate(f"{G}.cameraDist") < 1.6: break
    pg.wait_for_timeout(2500)
    mem = pg.evaluate("window.__cloud.pane.members")
    n_coin = sum(1 for i in mem if i in set(coins))
    titles = [items[i]["title"] for i in mem if i not in set(coins)][:12]
    print(f"pane: {len(mem)} works, {n_coin} coins ({n_coin / len(coins):.0%} of all coins), {len(mem) - n_coin} others; e.g. {titles}")
    pg.mouse.move(5, 890); pg.wait_for_timeout(800)
    pg.screenshot(path="reports/screens/coin_pane.png")
    b.close()
