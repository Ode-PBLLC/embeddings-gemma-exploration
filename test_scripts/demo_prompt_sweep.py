"""Sweep candidate demo prompts; score each wall on its mix of art, photographs, text and video.
Writes reports/demo_prompt_sweep.md."""
import json, urllib.request
from pathlib import Path

items = json.load(open("web/data/items.json"))
CANDS = """
the ocean|heaven|mountains|supernova|rockets|a rocket launch|fire and heat|machines that fly|the color blue|light on water|
loneliness|the color gold|the color red|storms|hurricanes|lightning|volcanoes|wildfire|ice and snow|glaciers|the arctic|
rivers|deserts|forests|gardens|flowers|clouds|rain|wind|the sun|sunrise|sunset|twilight|the moon|eclipse|the night sky|
stars|constellations|galaxies|nebulae|black holes|the birth of stars|comets|meteors|planets|rings|jupiter|saturn|mars|
earth from space|the blue marble|the edge of space|astronauts|spacewalk|weightlessness|exploration|voyages|discovery|
journeys|maps|navigation|ships|boats|flight|birds|wings|angels|gods|myths|creation|the beginning of time|time|eternity|
infinity|chaos|order|patterns|spirals|circles|light and shadow|darkness|glow|fireworks|explosions|energy|power|speed|
motion|waves|tides|water|reflections|mirrors|glass|gold|silver|jewels|crystals|metal|machines|engines|technology|
the future|the past|history|war|peace|solitude|silence|wonder|awe|beauty|the sublime|dreams|memory|home|faraway lands|
cities at night|lights from above|the northern lights|the sound of mars|music|voices|a spoken story|listening|the dance|buddha|bodhisattva|a temple|sacred objects|ancient egypt|sculpture|masks|ritual|the roar of engines|thunder|silence and noise|the human voice|stories of exploration|magnetism|the invisible|heat|cold|life|growth|plants in space|
farming|food|harvest|seasons|winter|summer|autumn|spring|color|rainbow|the color purple|the color green
"""
cands = [c.strip() for c in CANDS.replace("\n", "").split("|") if c.strip()]
def kind_of(it):
    if it["modality"] == "audio": return "sound"
    if it["modality"] == "video": return "video"
    return {"aic": "art", "wiki": "text"}.get(it["source"], "photo")
CAP = {"art": 3, "photo": 3, "text": 2, "video": 3, "sound": 2}
rows = []
for q in cands:
    req = urllib.request.Request("http://localhost:8000/search", data=json.dumps({"query": q}).encode(), headers={"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    ks = [kind_of(items[n["index"]]) for n in r["neighbors"]]
    c = {k: ks.count(k) for k in CAP}
    present = sum(v > 0 for v in c.values())
    score = sum(min(c[k], CAP[k]) for k in CAP) + 3 * present + (4 if present >= 4 else 0) + (4 if present == 5 else 0)
    top = [f"{kind_of(items[n['index']])[0].upper()}:{items[n['index']]['title'][:34]}" for n in r["neighbors"][:5]]
    rows.append((score, present, q, c, r["neighbors"][0]["similarity"], r["weak"], top))
rows.sort(key=lambda x: (-x[0], -x[4]))
lines = ["# Demo prompt sweep", "", f"{len(rows)} prompts scored on the mix of their 12-work wall "
         "(A art · P photograph · T text · V video · S sound). Score = capped counts + 3 per kind present (+4 for four kinds, +4 more for all five).", "",
         "| score | prompt | art | photo | text | video | sound | top-1 sim | first works |", "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
for s, pr, q, c, sim, weak, top in rows:
    lines.append(f"| {s} | {q}{' (weak)' if weak else ''} | {c['art']} | {c['photo']} | {c['text']} | {c['video']} | {c['sound']} | {sim:.3f} | {'; '.join(top)} |")
Path("reports/demo_prompt_sweep.md").write_text("\n".join(lines) + "\n")
print(f"{len(rows)} prompts; five kinds on {sum(r[1] == 5 for r in rows)} walls, four on {sum(r[1] == 4 for r in rows)}")
for s, pr, q, c, sim, weak, top in rows[:18]:
    print(f"{s:3d} {q!r:28} A{c['art']} P{c['photo']} T{c['text']} V{c['video']} S{c['sound']}  {top[:3]}")
