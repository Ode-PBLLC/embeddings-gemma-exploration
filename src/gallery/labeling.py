"""Labelling in embedding space: the numeric steps behind the labelling page (gallery.label_app).

Everything here works on the full 768-d vectors, never on the 3D layout: the layout is only where the
person points. Vectors are L2-normalised, so a dot product is a cosine similarity.

    Neighbours        each item's nearest items, in the whole corpus or inside a pool (a label's members)
    check_selection   works selected on the screen -> the groups of them that really sit together
    ward_tree         the merge tree of a set; the page cuts it at any number of groups
    name_groups       words that set each group apart, from titles and Gemma's notes
    suggest           labels proposed for unlabelled items by the vote of their neighbours
    export            the saved file: label tree, one row per item, and the page's own state
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

K = 15            # neighbours that vote on a suggestion, and that link items in a selection check
MIN_GROUP = 3     # a part of a selection smaller than this is a stray, shown for a yes or no
MIN_VOTE = 0.2    # propose a label when at least this share of an item's neighbours carry it
# words that describe the catalogue entry, not the work
STOP = ENGLISH_STOP_WORDS | {
    "depicting", "portraying", "depicts", "depicted", "image", "images", "photograph", "work", "works",
    "piece", "created", "made", "shows", "showing", "features", "featuring", "notable", "century",
    "view", "scene", "captured", "taken", "this", "known", "likely", "style", "period", "example",
}
WORD = re.compile(r"[a-z][a-z'-]{2,}")


def pool_k(m: int) -> int:
    """Neighbours to use inside a pool of m items: fewer in a small set, so a group of a dozen still splits."""
    return int(max(3, min(K, round(m ** 0.5))))


class Neighbours:
    """Nearest neighbours by cosine. The whole-corpus table is computed once; pools are computed on demand."""

    def __init__(self, vectors: np.ndarray):
        self.V = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n = len(self.V)
        self.idx, self.sim = self._table(np.arange(self.n), K)

    def _table(self, pool: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        S = self.V[pool] @ self.V[pool].T
        np.fill_diagonal(S, -np.inf)
        k = min(k, len(pool) - 1)
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, top, axis=1)
        order = np.argsort(-sims, axis=1)
        return pool[np.take_along_axis(top, order, axis=1)], np.take_along_axis(sims, order, axis=1)

    def table(self, pool: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(items, their neighbours, similarities). pool=None is the whole corpus."""
        if pool is None:
            return np.arange(self.n), self.idx, self.sim
        if len(pool) < 2:
            raise ValueError("a pool needs at least 2 items")
        idx, sim = self._table(pool, pool_k(len(pool)))
        return pool, idx, sim


def _indices(values, n: int, what: str) -> np.ndarray:
    a = np.unique(np.asarray(list(values), dtype=np.int64))
    if len(a) and (a[0] < 0 or a[-1] >= n):
        raise ValueError(f"{what}: index out of range 0..{n - 1}")
    return a


def check_selection(nb: Neighbours, seed, pool=None) -> dict:
    """Split what a drawn selection caught into the groups that sit together in the full vector space.

    Two caught items are linked when each is among the other's nearest neighbours (in the whole
    corpus, or in `pool` when the person is working inside a label). One-way links do not count: an
    outlier's neighbours are always somewhere, but they do not name it back. Connected groups come
    back largest first; groups smaller than MIN_GROUP are pooled as strays.
    """
    seed = _indices(seed, nb.n, "seed")
    if len(seed) == 0:
        raise ValueError("the selection is empty")
    if pool is not None:
        pool = _indices(pool, nb.n, "pool")
        seed = np.intersect1d(seed, pool)
        if len(seed) == 0:
            raise ValueError("nothing in the selection belongs to this label")
    if len(seed) == 1 or (pool is not None and len(pool) < 2):
        return {"groups": [seed.tolist()], "strays": [], "k": 0}
    members, idx, _ = nb.table(pool)
    pos = np.full(nb.n, -1)
    pos[seed] = np.arange(len(seed))
    rows_of = np.full(nb.n, -1)
    rows_of[members] = np.arange(len(members))
    nbrs = idx[rows_of[seed]]                       # (len(seed), k) neighbours of each caught item
    a = np.repeat(np.arange(len(seed)), nbrs.shape[1])
    b = pos[nbrs.ravel()]
    keep = b >= 0
    graph = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(len(seed), len(seed))).tocsr()
    _, comp = connected_components(graph.multiply(graph.T), directed=False)
    sizes = np.bincount(comp)
    groups = [seed[comp == c].tolist() for c in np.argsort(-sizes, kind="stable") if sizes[c] >= MIN_GROUP]
    strays = seed[sizes[comp] < MIN_GROUP].tolist()
    if not groups:                                  # nothing coherent: hand everything back as one group
        groups, strays = [seed.tolist()], []
    return {"groups": groups, "strays": strays, "k": int(nbrs.shape[1])}


def ward_tree(V: np.ndarray, members) -> dict:
    """Ward merge tree of the members' full vectors. merges[t] = [a, b] joins clusters a and b into
    cluster m + t (leaves are 0..m-1, in `members` order), as scipy.cluster.hierarchy.linkage numbers them."""
    members = _indices(members, len(V), "members")
    if len(members) < 2:
        raise ValueError("need at least 2 items to cluster")
    Z = linkage(V[members], method="ward")
    return {"members": members.tolist(), "merges": Z[:, :2].astype(int).tolist(),
            "heights": [round(float(h), 5) for h in Z[:, 2]]}


def cut_tree(merges: list[list[int]], m: int, k: int) -> np.ndarray:
    """Group number (0..k-1, largest group first) of each leaf when the tree is cut into k groups.
    The page does the same cut in JavaScript; this one is for tests and scripts."""
    k = max(1, min(k, m))
    parent = list(range(2 * m - 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t, (a, b) in enumerate(merges[: m - k]):
        parent[find(a)] = parent[find(b)] = m + t
    roots = np.array([find(i) for i in range(m)])
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    rank = np.empty(len(uniq), dtype=int)
    rank[np.argsort(-counts, kind="stable")] = np.arange(len(uniq))
    return rank[inv]


def _words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


def name_groups(V: np.ndarray, items: list[dict], groups: list[list[int]], n_words: int = 4, n_examples: int = 6) -> list[dict]:
    """For each group: the words that set it apart from the other groups, and its most central works.

    A word's score is the share of the group's items that use it minus the share of the other groups'
    items that do (titles count double, Gemma's notes once). No model is called.
    """
    for g in groups:
        if not g:
            raise ValueError("empty group")
    # per group: in how many items each word occurs, in titles and in notes
    counts = []
    for g in groups:
        t, s = Counter(), Counter()
        for i in g:
            t.update(_words(items[i]["title"]))
            s.update(_words(items[i].get("summary") or ""))
        counts.append((t, s))
    total_t = sum((c[0] for c in counts), Counter())
    total_s = sum((c[1] for c in counts), Counter())
    n_all = sum(len(g) for g in groups)
    out = []
    for g, (t, s) in zip(groups, counts):
        n_rest = n_all - len(g)

        def lift(w: str, mine: Counter, total: Counter) -> float:
            rest = (total[w] - mine[w]) / n_rest if n_rest else 0.0
            return mine[w] / len(g) - rest

        scored = sorted(((2 * lift(w, t, total_t) + lift(w, s, total_s), w) for w in t.keys() | s.keys()),
                        key=lambda x: (-x[0], x[1]))
        words = [w for score, w in scored[:n_words] if score > 0.05]
        idx = np.asarray(g)
        centre = V[idx].mean(0)
        order = np.argsort(-(V[idx] @ centre), kind="stable")
        out.append({"words": words, "examples": idx[order[:n_examples]].tolist()})
    return out


def suggest(nb: Neighbours, labels: list[dict], limit: int = 60) -> list[dict]:
    """Propose labels for unlabelled items from the vote of their nearest neighbours.

    labels: [{"node", "members": [...], "pool": [...]|None, "rejected": [...], "group"}]. Labels that
    share a `group` (siblings under one parent) compete: an item is proposed for the one with the
    larger vote, and an item already in one of them is not proposed for another. `pool` limits both the candidates and the neighbours to a parent label's members.
    confidence = the similarity-weighted share of the item's neighbours that carry the label.
    """
    best: dict[tuple, dict] = {}
    tables: dict[bytes | None, tuple] = {}
    taken: dict = {}                                 # group -> items already in one of its labels
    for lab in labels:
        if lab.get("group") is not None:
            mask = taken.setdefault(lab["group"], np.zeros(nb.n, dtype=bool))
            mask[_indices(lab["members"], nb.n, "members")] = True
    for lab in labels:
        members = _indices(lab["members"], nb.n, "members")
        if len(members) < MIN_GROUP:
            continue
        pool = None if lab.get("pool") is None else _indices(lab["pool"], nb.n, "pool")
        if pool is not None and len(pool) < 2:
            continue
        key = None if pool is None else pool.tobytes()
        if key not in tables:
            tables[key] = nb.table(pool)
        cand, idx, sim = tables[key]
        is_member = np.zeros(nb.n, dtype=bool)
        is_member[members] = True
        vote = (sim * is_member[idx]).sum(1) / sim.sum(1)
        count = is_member[idx].sum(1)
        skip = is_member | taken.get(lab.get("group"), False)
        skip[_indices(lab.get("rejected", []), nb.n, "rejected")] = True
        for r in np.flatnonzero((vote >= MIN_VOTE) & ~skip[cand]):
            i = int(cand[r])
            s = {"item": i, "node": lab["node"], "confidence": round(float(vote[r]), 3),
                 "votes": int(count[r]), "of": int(idx.shape[1])}
            k = (i, lab.get("group"))
            if k not in best or s["confidence"] > best[k]["confidence"]:
                best[k] = s
    out = sorted(best.values(), key=lambda s: (-s["confidence"], s["item"]))
    per_node: dict = {}
    kept = []
    for s in out:
        per_node[s["node"]] = per_node.get(s["node"], 0) + 1
        if per_node[s["node"]] <= limit:
            kept.append(s)
    return kept


# ---------- the saved file ----------

SOURCES = ("person", "suggestion")
REVIEWS = ("by-hand", "accepted", "accepted-in-bulk")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_state(state: dict, n: int) -> dict:
    """Check the page's state before it is written: nodes form a tree, every row points at a real node and item."""
    nodes = state.get("nodes", [])
    ids = [nd["id"] for nd in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate label ids")
    known = set(ids)
    for nd in nodes:
        if not str(nd.get("name", "")).strip():
            raise ValueError(f"label {nd['id']} has no name")
        seen, p = {nd["id"]}, nd.get("parent")
        while p is not None:
            if p not in known or p in seen:
                raise ValueError(f"label {nd['id']}: bad parent chain")
            seen.add(p)
            p = next(x for x in nodes if x["id"] == p).get("parent")
    for kind in ("labels", "pending", "rejected"):
        for row in state.get(kind, []):
            if row["node"] not in known:
                raise ValueError(f"{kind}: unknown label {row['node']!r}")
            if not (isinstance(row["i"], int) and 0 <= row["i"] < n):
                raise ValueError(f"{kind}: item index {row['i']!r} out of range")
    for row in state.get("labels", []):
        if row.get("source") not in SOURCES or row.get("review") not in REVIEWS:
            raise ValueError(f"labels: bad source/review on item {row['i']}: {row.get('source')!r}, {row.get('review')!r}")
    return {"nodes": nodes, "labels": state.get("labels", []), "pending": state.get("pending", []),
            "rejected": state.get("rejected", [])}


def export(state: dict, items: list[dict], dataset: dict) -> tuple[dict, str]:
    """(the JSON document, the flat CSV). The JSON has the tree, one row per labelled item, and the
    page's own state so a session can be picked up again. Labels a person confirmed and suggestions
    still waiting are kept in separate lists on each item."""
    state = validate_state(state, len(items))
    by_id = {nd["id"]: nd for nd in state["nodes"]}

    def path(node: str) -> str:
        parts = []
        while node is not None:
            parts.append(by_id[node]["name"])
            node = by_id[node].get("parent")
        return " > ".join(reversed(parts))

    counts = {nd["id"]: 0 for nd in state["nodes"]}
    seen = set()
    for row in state["labels"]:
        node = row["node"]
        while node is not None:                      # a work labelled "Gold" is also one of the "Coins"
            if (row["i"], node) not in seen:
                seen.add((row["i"], node))
                counts[node] += 1
            node = by_id[node].get("parent")
    tree = [{"id": nd["id"], "name": nd["name"], "parent": nd.get("parent"), "path": path(nd["id"]),
             "items": counts[nd["id"]], "created_at": nd.get("at"), "created_via": nd.get("via")}
            for nd in state["nodes"]]
    rows: dict[int, dict] = {}

    def row_for(i: int) -> dict:
        if i not in rows:
            rows[i] = {"id": items[i]["id"], "title": items[i]["title"], "labels": [], "suggested": []}
        return rows[i]

    flat = io.StringIO()
    w = csv.writer(flat)
    # confidence: share of nearest neighbours carrying the label (neighbour-vote). margin: how much closer
    # the work is to its phrase than to the next one, in cosine similarity (text-split).
    w.writerow(["item_id", "title", "label", "source", "method", "review", "confidence", "margin", "at"])
    for r in sorted(state["labels"], key=lambda r: (r["i"], path(r["node"]))):
        entry = {"label": path(r["node"]), "source": r["source"], "method": r.get("method"),
                 "review": r["review"], "confidence": r.get("confidence"), "margin": r.get("margin"), "at": r.get("at")}
        row_for(r["i"])["labels"].append(entry)
        w.writerow([items[r["i"]]["id"], items[r["i"]]["title"], *entry.values()])
    for r in sorted(state["pending"], key=lambda r: (r["i"], path(r["node"]))):
        entry = {"label": path(r["node"]), "source": "suggestion", "method": r.get("method"),
                 "review": "pending", "confidence": r.get("confidence"), "margin": r.get("margin"), "at": r.get("at")}
        row_for(r["i"])["suggested"].append(entry)
        w.writerow([items[r["i"]]["id"], items[r["i"]]["title"], *entry.values()])
    doc = {"version": 1, "saved_at": now(), "dataset": dataset, "tree": tree,
           "items": [rows[i] for i in sorted(rows)],
           "rejected": [{"id": items[r["i"]]["id"], "label": path(r["node"]), "at": r.get("at")} for r in state["rejected"]],
           "state": state}
    return doc, flat.getvalue()


def save(directory: Path, state: dict, items: list[dict], dataset: dict) -> dict:
    """Write labels.json and label_rows.csv (each through a temp file, then renamed into place)."""
    doc, flat = export(state, items, dataset)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in (("labels.json", json.dumps(doc, ensure_ascii=False, indent=1)), ("label_rows.csv", flat)):
        tmp = directory / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, directory / name)
    return {"saved_at": doc["saved_at"], "labels": len(state["labels"]), "pending": len(state["pending"]),
            "path": str(directory / "labels.json")}


def load(directory: Path, items: list[dict], dataset: dict) -> dict:
    """The page state from a saved file. Rows hold item indices, so a file saved for a different
    bundle (other items, or another order) is refused rather than applied to the wrong works."""
    f = directory / "labels.json"
    if not f.is_file():
        return {"nodes": [], "labels": [], "pending": [], "rejected": []}
    doc = json.loads(f.read_text(encoding="utf-8"))
    saved = doc.get("dataset", {})
    if saved.get("ids_sha1") != dataset["ids_sha1"]:
        raise ValueError(f"{f} was saved for a different bundle ({saved.get('n')} items); this one has {len(items)}")
    return validate_state(doc["state"], len(items))
