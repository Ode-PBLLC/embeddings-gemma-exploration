"""Ad hoc: score candidate demo prompts by how their wall mixes art, NASA imagery and text."""
import json, urllib.request
items = json.load(open("web/data/items.json"))
cands = ["fire and heat", "machines that fly", "the color blue", "light on water", "loneliness", "storms and turbulent skies",
         "the night sky", "spirals", "red planet dust", "ancient gods", "explorers and voyages", "gold and luxury",
         "the sun", "the moon", "rings and circles", "clouds", "ice and snow", "volcanoes", "the edge of the world",
         "a journey into the unknown", "heaven", "light and shadow", "the color red", "the color gold", "rivers seen from above",
         "stars being born", "waves", "the horizon", "mountains", "dreams", "flight", "the ocean", "Jupiter's storms",
         "Monet's water lilies", "sunlight through clouds", "orbit", "eclipse", "landscape at dusk", "northern lights", "the cosmos"]
rows = []
for q in cands:
    req = urllib.request.Request("http://localhost:8000/search", data=json.dumps({"query": q}).encode(), headers={"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    src = [items[n["index"]]["source"] for n in r["neighbors"]]
    c = {s: src.count(s) for s in ("aic", "nasa", "wiki")}
    top = [items[n["index"]]["title"][:38] for n in r["neighbors"][:4]]
    # balance: all three present, and images dominate with some text
    score = (min(c["aic"], 4) + min(c["nasa"], 4) + min(c["wiki"], 3)) + (3 if all(c.values()) else 0)
    rows.append((score, q, c, round(r["neighbors"][0]["similarity"], 3), r["weak"], top))
for row in sorted(rows, reverse=True)[:14]:
    print(row[0], repr(row[1]), row[2], "top1", row[3], "weak" if row[4] else "", row[5])
