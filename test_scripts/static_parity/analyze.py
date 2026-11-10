"""Compare browser-side (ONNX) query embeddings against the running server's answers.

    uv run python test_scripts/static_parity/analyze.py [variant ...]

Inputs (from the other scripts in this folder): queries.json, out/server.json, out/<variant>.f32.
Search is replayed exactly as gallery.search.Searcher does it (raw mode, k=12, no filters) over the
same vectors the server loaded (via gallery.app.load_searcher). Prints a markdown table per set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from gallery.app import load_searcher
from gallery.schema import ROOT

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DEFAULT = ["torch_cpu_bfloat16", "torch_cpu_float32", "onnx_fp32", "onnx_fp16", "onnx_q8", "onnx_q4", "onnx_q4f16"]
SETS = {"gallery": (ROOT / "data", ROOT / "web", ROOT / "config" / "retrieval.json"),
        "moon": (ROOT / "data_moon", ROOT / "web_moon", ROOT / "data_moon" / "retrieval.json")}


def no_embed(_: str) -> np.ndarray:
    raise AssertionError("analysis never embeds")


def metrics(searcher, server: list[dict], kinds: list[str], qv: np.ndarray) -> dict:
    k, thr = searcher.config.k, searcher.config.weak_threshold
    sims = qv @ searcher.vectors.T                                    # (nq, n_items)
    ov, top1, tau, dsim, weak_ok, top1_sims = [], [], [], [], [], []
    for i, ref in enumerate(server):
        ref_idx = [n["index"] for n in ref["neighbors"]]
        ref_sim = np.array([n["similarity"] for n in ref["neighbors"]])
        mine = np.argsort(-sims[i], kind="stable")[:k]
        ov.append(len(set(mine) & set(ref_idx)) / k)
        top1.append(mine[0] == ref_idx[0])
        tau.append(kendalltau(ref_sim, sims[i, ref_idx]).statistic)   # rank agreement on the server's 12
        dsim.append(np.abs(sims[i, ref_idx] - ref_sim).max())
        weak_ok.append((sims[i].max() < thr) == ref["weak"])
        top1_sims.append(sims[i].max())
    kinds_a, t1 = np.array(kinds), np.array(top1_sims)
    on, off = t1[kinds_a != "nothing"], t1[kinds_a == "nothing"]
    ref_t1 = np.array([r["neighbors"][0]["similarity"] for r in server])
    return {
        "overlap@12 mean": np.mean(ov), "overlap@12 min": np.min(ov),
        "queries with 12/12": np.mean(np.array(ov) == 1.0),
        "top-1 same": np.mean(top1), "kendall tau mean": np.nanmean(tau),
        "max |Δsim| median": np.median(dsim), "max |Δsim| max": np.max(dsim),
        "weak flag same": np.mean(weak_ok),
        "weak: on-topic flagged": int((on < thr).sum()), "weak: nothing-close flagged": int((off < thr).sum()),
        "server: on-topic flagged": int((ref_t1[kinds_a != "nothing"] < thr).sum()),
        "server: nothing-close flagged": int((ref_t1[kinds_a == "nothing"] < thr).sum()),
        "_n_on": len(on), "_n_off": len(off),
    }


def main() -> None:
    variants = sys.argv[1:] or [v for v in DEFAULT if (OUT / f"{v}.f32").exists()]
    sets = json.loads((HERE / "queries.json").read_text())
    server = json.loads((OUT / "server.json").read_text())
    n_gal = len(sets["gallery"])
    vecs = {v: np.fromfile(OUT / f"{v}.f32", dtype="<f4").reshape(-1, 768) for v in variants}
    for v, a in vecs.items():
        assert len(a) == n_gal + len(sets["moon"]), f"{v}: {len(a)} rows"
        assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-4), f"{v}: not unit length"
    report = {}
    for name, (data, web, retr) in SETS.items():
        s = load_searcher(data, web, retr, no_embed)
        rows = slice(0, n_gal) if name == "gallery" else slice(n_gal, None)
        kinds = [q["kind"] for q in sets[name]]
        assert [r["query"] for r in server[name]] == [q["q"].strip() for q in sets[name]], "server.json order"
        res = {v: metrics(s, server[name], kinds, a[rows]) for v, a in vecs.items()}
        if "torch_cpu_bfloat16" in vecs:   # cosine to the server stand-in
            ref = vecs["torch_cpu_bfloat16"][rows]
            for v, a in vecs.items():
                c = (a[rows] * ref).sum(1)
                res[v]["cos vs torch bf16 mean"], res[v]["cos vs torch bf16 min"] = c.mean(), c.min()
        report[name] = res
        n_on, n_off = res[variants[0]]["_n_on"], res[variants[0]]["_n_off"]
        print(f"\n### {name}: {len(kinds)} queries ({n_on} on-topic, {n_off} nothing-close), "
              f"{len(s.ids)} items, weak_threshold {s.config.weak_threshold}\n")
        keys = [key for key in res[variants[0]] if not key.startswith("_")]
        print("| metric | " + " | ".join(variants) + " |")
        print("|---|" + "---:|" * len(variants))
        for key in keys:
            cells = [res[v].get(key) for v in variants]
            print(f"| {key} | " + " | ".join("" if c is None else (f"{c}" if isinstance(c, int) else f"{c:.4f}")
                                            for c in cells) + " |")
    (OUT / "parity.json").write_text(json.dumps(report, default=float, indent=1))


if __name__ == "__main__":
    main()
