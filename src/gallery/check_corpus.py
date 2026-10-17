"""Corpus checks.

uv run python -m gallery.check_corpus --source aic|nasa|wiki   # one raw parquet
uv run python -m gallery.check_corpus --all                    # data/items.parquet

Exit status is non-zero, with a message per failed check, when anything is off.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from gallery.schema import DATA, SchemaError, validate

SOURCE_RANGE = (1600, 1800)
TOPIC_CAP = {"aic": 0.25, "nasa": 0.15}
WIKI_MAX_PER_ARTICLE = 6
WIKI_MIN_WORDS = 40
TOTAL_RANGE = (4800, 5400)
SHARE_RANGE = (0.30, 0.38)   # upper bound raised from 0.36 when NASA videos joined the nasa source
SOURCES = ("aic", "nasa", "wiki")


def check_source(df: pd.DataFrame, source: str) -> list[str]:
    errs: list[str] = []
    try:
        validate(df)
    except SchemaError as e:
        return [f"{source}: schema: {e}"]
    if set(df["source"].unique()) != {source}:
        errs.append(f"{source}: unexpected source values {sorted(df['source'].unique())}")
    lo, hi = SOURCE_RANGE
    if not lo <= len(df) <= hi:
        errs.append(f"{source}: {len(df)} rows, expected {lo}-{hi}")
    if source in TOPIC_CAP:
        if (df["topic"].astype(str).str.strip() == "").any():
            errs.append(f"{source}: {int((df['topic'].astype(str).str.strip() == '').sum())} rows with empty topic")
        share = df["topic"].value_counts(normalize=True)
        if share.iloc[0] > TOPIC_CAP[source]:
            errs.append(f"{source}: topic '{share.index[0]}' holds {share.iloc[0]:.1%} > {TOPIC_CAP[source]:.0%}")
    if source == "wiki":
        per = df["topic"].value_counts()
        if per.iloc[0] > WIKI_MAX_PER_ARTICLE:
            errs.append(f"wiki: article '{per.index[0]}' has {per.iloc[0]} paragraphs > {WIKI_MAX_PER_ARTICLE}")
        short = df["raw_text"].str.split().str.len() < WIKI_MIN_WORDS
        if short.any():
            errs.append(f"wiki: {int(short.sum())} paragraphs under {WIKI_MIN_WORDS} words")
    return errs


def check_all(df: pd.DataFrame) -> list[str]:
    try:
        validate(df)
    except SchemaError as e:
        return [f"items: schema: {e}"]
    errs: list[str] = []
    lo, hi = TOTAL_RANGE
    if not lo <= len(df) <= hi:
        errs.append(f"items: {len(df)} rows, expected {lo}-{hi}")
    share = df["source"].value_counts(normalize=True)
    for s in SOURCES:
        v = float(share.get(s, 0.0))
        if not SHARE_RANGE[0] <= v <= SHARE_RANGE[1]:
            errs.append(f"items: source {s} holds {v:.1%}, expected {SHARE_RANGE[0]:.0%}-{SHARE_RANGE[1]:.0%}")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", choices=SOURCES)
    g.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    path = DATA / "items.parquet" if args.all else DATA / "raw" / f"{args.source}.parquet"
    if not path.is_file():
        print(f"FAIL: {path} does not exist", file=sys.stderr)
        return 1
    df = pd.read_parquet(path)
    errs = check_all(df) if args.all else check_source(df, args.source)
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    if args.all:
        mix = ", ".join(f"{k}={v}" for k, v in df["source"].value_counts().items())
        print(f"OK: {path.name} {len(df)} rows ({mix})")
    else:
        top = df["topic"].value_counts(normalize=True)
        print(f"OK: {path.name} {len(df)} rows, {df['topic'].nunique()} topics, largest '{top.index[0]}' {top.iloc[0]:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
