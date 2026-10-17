"""Calibrate the weak-result threshold for a single-collection dataset (the Moon set).

    uv run python -m gallery.calibrate_weak --data-dir data_moon --queries eval/moon_queries.yaml

The same rule as gallery.eval_gap: threshold = midpoint of the 10th-percentile best similarity of
the on-topic queries and the 90th-percentile best similarity of the off-topic ones. Writes
weak_threshold into <data>/retrieval.json (keeping its other keys) and a report next to it in
reports/. Exits 1 if the split fails (fewer than 8 of the off-topic queries come back weak, or more
than 2 on-topic ones do), with the threshold still written so the report can be inspected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from gallery.schema import ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    data = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    spec = yaml.safe_load((ROOT / args.queries).read_text())
    on, off = [e["q"] for e in spec["on_topic"]], list(spec["off_topic"])

    from gallery.embed import Embedder

    vectors = np.load(data / "vectors.npy")
    emb = Embedder()
    q = np.asarray(emb.embed_texts(on + off, kind="query"), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    top1 = (q @ vectors.T).max(1)
    on_top1, off_top1 = top1[:len(on)], top1[len(on):]
    lo, hi = float(np.percentile(on_top1, 10)), float(np.percentile(off_top1, 90))
    threshold = round((lo + hi) / 2, 4)
    weak_off, weak_on = int((off_top1 < threshold).sum()), int((on_top1 < threshold).sum())
    ok = weak_off >= 8 and weak_on <= 2

    cfg_path = data / "retrieval.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["weak_threshold"] = threshold
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    report = args.report or ROOT / "reports" / f"weak_threshold_{data.name}.md"
    lines = [f"# Weak threshold: {data.name}", "",
             f"{len(vectors)} items; {len(on)} on-topic and {len(off)} off-topic queries ({args.queries}).", "",
             f"- on-topic best similarity: p10 {lo:.4f}, median {np.median(on_top1):.4f}",
             f"- off-topic best similarity: p90 {hi:.4f}, median {np.median(off_top1):.4f}",
             f"- threshold = midpoint = **{threshold}**",
             f"- off-topic flagged weak: {weak_off}/{len(off)}; on-topic flagged weak: {weak_on}/{len(on)}",
             f"- result: {'PASS' if ok else 'FAIL'}", "", "| query | best sim | weak |", "|---|---:|---|",
             *[f"| {t} | {s:.4f} | {'yes' if s < threshold else ''} |" for t, s in zip(on + off, top1)], ""]
    report.write_text("\n".join(lines))
    print("\n".join(lines[:9]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
