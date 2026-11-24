"""Ad hoc: the labelling numerics on the real gallery vectors, with the coins as the test case.
    uv run python test_scripts/label_numeric_check.py
Simulates circles drawn around the coin cluster from many viewing directions (orthographic), then
reports what check_selection keeps, what suggest proposes, and how a 3-way Ward cut is named."""
import collections
import json
from pathlib import Path

import numpy as np

from gallery.app import load_searcher
from gallery.labeling import Neighbours, check_selection, cut_tree, name_groups, suggest, ward_tree
from gallery.paths import assets_dir, data_dir, retrieval_config

s = load_searcher(data_dir(), assets_dir(), retrieval_config(), embed=lambda q: None)
items = json.loads((assets_dir() / "data" / "items.json").read_text())
V, xyz, n = s.vectors, s.layout, len(items)
is_coin = np.array([("(Coin)" in it["title"]) or it["title"].startswith("Coin ") for it in items])
coins = np.flatnonzero(is_coin)
nb = Neighbours(V)
print(f"{n} items, {len(coins)} coins by title")

rng = np.random.default_rng(0)
rows = []
for _ in range(40):
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    u = np.cross(d, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(d, u)
    P = np.c_[xyz @ u, xyz @ v]
    c = P[coins].mean(0)
    r = np.quantile(np.linalg.norm(P[coins] - c, axis=1), 0.9)
    seed = np.flatnonzero(np.linalg.norm(P - c, axis=1) < r)
    res = check_selection(nb, seed)
    main = np.array(res["groups"][0])
    rows.append((len(seed), int(is_coin[seed].sum()), len(main), int(is_coin[main].sum()),
                 [len(g) for g in res["groups"][1:]], len(res["strays"]), int(is_coin[res["strays"]].sum()) if res["strays"] else 0))
print("circle caught / coins in it / main group / coins in main / other groups / strays / coin strays")
for r in sorted(set(map(str, rows))): print("  ", r)
dirty = [r for r in rows if r[0] > r[1]]
print(f"{len(dirty)} of {len(rows)} views caught non-coins; in those, main group purity:",
      [f"{r[3]}/{r[2]}" for r in dirty][:12])

# label the main group of the last circle, then ask for suggestions
sug = suggest(nb, [{"node": "coins", "members": main.tolist(), "pool": None, "rejected": [], "group": None}])
print(f"\nlabelled {len(main)}; {len(sug)} suggestions; coins among them: {sum(is_coin[x['item']] for x in sug)} of {int(is_coin.sum() - is_coin[main].sum())} unlabelled coins")
for x in sug[:14]: print("  ", x["confidence"], f"{x['votes']}/{x['of']}", "COIN" if is_coin[x["item"]] else "    ", items[x["item"]]["title"][:60])

t = ward_tree(V, coins)
for k in (2, 3, 5):
    lab = cut_tree(t["merges"], len(coins), k)
    groups = [np.array(t["members"])[lab == g].tolist() for g in range(k)]
    print(f"\nward k={k}")
    for g, nm in zip(groups, name_groups(V, items, groups)):
        print("  ", len(g), nm["words"], "|", [items[i]["title"][:28] for i in nm["examples"][:2]])

# sub-suggestion inside the coins: hand-label 6 aurei, vote inside the pool
aurei = [i for i in coins if "Aureus" in items[i]["title"]]
sub = suggest(nb, [{"node": "gold", "members": aurei[:6], "pool": coins.tolist(), "rejected": [], "group": "coins"}])
print(f"\n{len(aurei)} aurei; seeded 6; proposed {len(sub)}:", collections.Counter("Aureus" in items[x["item"]]["title"] for x in sub))
# whole corpus tree
import time; t0 = time.time(); ward_tree(V, range(n)); print(f"ward on all {n}: {time.time() - t0:.1f}s")
t = ward_tree(V, range(n)); lab = cut_tree(t["merges"], n, 8)
for g, nm in zip([np.flatnonzero(lab == g).tolist() for g in range(8)], name_groups(V, items, [np.flatnonzero(lab == g).tolist() for g in range(8)])):
    print("  ", len(g), nm["words"])
