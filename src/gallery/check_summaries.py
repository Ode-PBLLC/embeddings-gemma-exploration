"""Exit non-zero unless >= 98% of items have a 40-120 word summary.

    uv run python -m gallery.check_summaries
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from gallery.curate import FAILURES, ITEMS, SUMMARIES, load_done
from gallery.llm import MAX_WORDS, MIN_WORDS, word_count

THRESHOLD = 0.98


def check(items: Path = ITEMS, summaries: Path = SUMMARIES, threshold: float = THRESHOLD) -> tuple[bool, str]:
    ids = pd.read_parquet(items, columns=["id"])["id"].tolist()
    done = load_done(summaries)
    good = sum(1 for i in ids if i in done and MIN_WORDS <= word_count(done[i]) <= MAX_WORDS)
    missing = sum(1 for i in ids if i not in done)
    off_len = sum(1 for i in ids if i in done) - good
    frac = good / len(ids) if ids else 0.0
    n_fail = 0
    if FAILURES.exists():
        n_fail = sum(1 for line in FAILURES.read_text(encoding="utf-8").splitlines() if line.strip())
    msg = (f"items={len(ids)} good={good} ({frac:.2%}) missing={missing} off_length={off_len} "
           f"failure_records={n_fail} threshold={threshold:.0%}")
    return frac >= threshold, msg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    ok, msg = check()
    print(("PASS " if ok else "FAIL ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
