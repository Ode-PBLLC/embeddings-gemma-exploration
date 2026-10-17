"""Measure the modality gap, choose the retrieval policy (SPEC §6) and calibrate the weak threshold.

Writes config/retrieval.json and reports/modality_gap.md.
Policy order: raw -> re-embed images as image+title -> per-source quota (4/4/4).
Stops at the first step where >= 12 of 15 on-topic queries return a mixed wall (>= 2 sources).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from gallery.schema import DATA, ROOT

K = 12
MIXED_NEEDED = 12
CONFIG = ROOT / "config" / "retrieval.json"
REPORT = ROOT / "reports" / "modality_gap.md"


def load_queries():
    spec = yaml.safe_load((ROOT / "eval" / "queries.yaml").read_text())
    on = [e["q"] for e in spec["on_topic"]]
    off = list(spec["off_topic"])
    return on, off


def top_k(sims: np.ndarray, sources: np.ndarray, mode: str) -> np.ndarray:
    """Indices of the wall for one query's similarity row."""
    if mode == "raw":
        return np.argsort(-sims)[:K]
    per = K // 3
    picks = []
    for s in ("aic", "nasa", "wiki"):
        idx = np.flatnonzero(sources == s)
        picks.extend(idx[np.argsort(-sims[idx])[:per]])
    picks = np.array(picks)
    return picks[np.argsort(-sims[picks])]


def measure(q_vecs: np.ndarray, vectors: np.ndarray, sources: np.ndarray, queries, mode: str,
            kinds: np.ndarray | None = None):
    sims = q_vecs @ vectors.T
    rows = []
    for q, row in zip(queries, sims):
        wall = top_k(row, sources, mode)
        counts = pd.Series(sources[wall]).value_counts().to_dict()
        r = {"query": q, "top1": float(row.max()), **{s: counts.get(s, 0) for s in ("aic", "nasa", "wiki")},
             "mixed": sum(1 for v in counts.values() if v > 0) >= 2}
        if kinds is not None:
            r["video"] = int((kinds[wall] == "video").sum())   # how many of the 12 are clips
        rows.append(r)
    return pd.DataFrame(rows)


def write_config(cfg: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from gallery.embed import Embedder

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-policy", action="store_true",
                    help="keep the mode/image_input in config/retrieval.json; only re-measure the mix and "
                         "recalibrate the weak threshold (e.g. after adding items)")
    args = ap.parse_args(argv)

    on, off = load_queries()
    items = pd.read_parquet(DATA / "items.parquet")
    ids = json.loads((DATA / "vector_ids.json").read_text())
    if ids != items["id"].tolist():
        raise SystemExit("vector_ids.json does not match items.parquet order; rerun embed_corpus")
    sources = items["source"].to_numpy()
    kinds = items["modality"].to_numpy()

    emb = Embedder()
    q_on = emb.embed_texts(on, kind="query")
    q_off = emb.embed_texts(off, kind="query")
    del emb

    sections = []
    chosen = None
    image_input = json.loads(CONFIG.read_text()).get("image_input", "image") if CONFIG.exists() else "image"

    steps = [("raw", "image"), ("raw", "image+title"), ("quota", None)]
    if args.keep_policy:
        cur = json.loads(CONFIG.read_text())
        steps = [(cur["mode"], None)]
    for mode, step_input in steps:
        if step_input is not None and step_input != image_input:
            # Re-embed images with the new input form; keep the previous vectors for the record.
            shutil.copy(DATA / "vectors.npy", DATA / f"vectors_{image_input}.npy")
            write_config({"mode": "raw", "image_input": step_input, "k": K, "weak_threshold": None})
            subprocess.run([sys.executable, "-m", "gallery.embed_corpus"], check=True, cwd=ROOT)
            image_input = step_input
        vectors = np.load(DATA / "vectors.npy")
        df = measure(q_on, vectors, sources, on, mode, kinds if (kinds == "video").any() else None)
        n_mixed = int(df["mixed"].sum())
        sections.append((f"mode={mode}, image_input={image_input}", df, n_mixed))
        if n_mixed >= MIXED_NEEDED:
            chosen = (mode, image_input, vectors)
            break
    if chosen is None and args.keep_policy:
        chosen = (steps[0][0], image_input, vectors)   # kept as configured; the report shows how it measures
    if chosen is None:
        chosen = ("quota", image_input, vectors)  # quota is the last resort; record that it still failed

    mode, image_input, vectors = chosen
    on_top1 = (q_on @ vectors.T).max(axis=1)
    off_top1 = (q_off @ vectors.T).max(axis=1)
    lo, hi = float(np.percentile(on_top1, 10)), float(np.percentile(off_top1, 90))
    threshold = round((lo + hi) / 2, 4)
    weak_off = int((off_top1 < threshold).sum())
    weak_on = int((on_top1 < threshold).sum())
    threshold_ok = weak_off >= 8 and weak_on <= 2

    write_config({"mode": mode, "image_input": image_input, "k": K, "weak_threshold": threshold})

    lines = ["# Modality gap and retrieval policy", ""]
    for title, df, n_mixed in sections:
        lines += [f"## {title}: {n_mixed}/{len(df)} mixed walls (need {MIXED_NEEDED})", "",
                  df.to_markdown(index=False, floatfmt=".3f"), ""]
    lines += [
        f"**Chosen:** mode=`{mode}`, image_input=`{image_input}`.", "",
        "## Weak threshold",
        f"- on-topic top-1 similarity: min {on_top1.min():.3f}, p10 {lo:.3f}, median {np.median(on_top1):.3f}",
        f"- off-topic top-1 similarity: p90 {hi:.3f}, median {np.median(off_top1):.3f}, max {off_top1.max():.3f}",
        f"- threshold = midpoint(p10 on, p90 off) = **{threshold}**",
        f"- off-topic flagged weak: {weak_off}/{len(off)} (need >= 8); on-topic flagged weak: {weak_on}/{len(on)} (need <= 2)",
        f"- result: {'PASS' if threshold_ok else 'FAIL'}", "",
        "Off-topic top-1: " + ", ".join(f"{q!r}={s:.3f}" for q, s in zip(off, off_top1)),
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT.read_text())
    mixed_ok = sections[-1][2] >= MIXED_NEEDED
    return 0 if (mixed_ok and threshold_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
