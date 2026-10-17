"""Merge the per-source parquets into data/items.parquet and write reports/corpus_mix.md.

uv run python -m gallery.merge

Row order (aic, then nasa, then wiki, then nasa videos, each in ingestion order) is the canonical
order for every later stage. Videos come last so adding them leaves every earlier row in place.
"""

from __future__ import annotations

import sys

import pandas as pd

from gallery.check_corpus import SOURCES, check_all
from gallery.schema import COLUMNS, DATA, ROOT, validate

REPORT = ROOT / "reports" / "corpus_mix.md"


def merge() -> pd.DataFrame:
    frames = []
    for s in SOURCES:
        path = DATA / "raw" / f"{s}.parquet"
        if not path.is_file():
            raise SystemExit(f"{path} not found: run `uv run python -m gallery.ingest.{s}` first")
        frames.append(pd.read_parquet(path)[COLUMNS])
    for extra in ("nasa_video", "nasa_audio", "aic_audio"):   # optional: clips, NASA sounds, AIC audio guides
        path = DATA / "raw" / f"{extra}.parquet"
        if path.is_file():
            frames.append(pd.read_parquet(path)[COLUMNS])
    return validate(pd.concat(frames, ignore_index=True))


def mix_report(df: pd.DataFrame) -> str:
    n = len(df)
    lines = ["# Corpus mix", "", f"Total items: **{n}** (from `data/items.parquet`)", "",
             "## By source", "", "| source | items | share |", "|---|---:|---:|"]
    for s, c in df["source"].value_counts().reindex(list(SOURCES)).items():
        lines.append(f"| {s} | {c} | {c / n:.1%} |")
    for s in SOURCES:
        sub = df[df["source"] == s]
        vc = sub["topic"].value_counts()
        lines += ["", f"## {s}: by topic ({len(vc)} topics)", ""]
        if s == "wiki":
            lines.append(f"Paragraphs per article: min {vc.min()}, median {int(vc.median())}, max {vc.max()}. "
                         "Top 25 articles by paragraph count:")
            lines.append("")
            vc = vc.head(25)
        lines += ["| topic | items | share of source |", "|---|---:|---:|"]
        for t, c in vc.items():
            lines.append(f"| {t} | {c} | {c / len(sub):.1%} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    df = merge()
    errs = check_all(df)
    out = DATA / "items.parquet"
    df.to_parquet(out, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(mix_report(df), encoding="utf-8")
    print(f"wrote {out} ({len(df)} rows) and {REPORT}")
    for e in errs:
        print(f"FAIL: {e}", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
