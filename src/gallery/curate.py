"""Offline Gemma 4 curation: one summary per item, resumable.

    uv run python -m gallery.curate [--limit N] [--workers W]

- reads data/items.parquet (canonical order)
- appends {"id", "summary"} to data/summaries.jsonl after each item (flushed + fsynced)
- on restart, skips ids already in summaries.jsonl
- each item is tried at most 3 times; final failures go to reports/curate_failures.jsonl
- workers > 1 sends concurrent requests (match llama-server -np slots)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

import pandas as pd

from gallery.paths import data_dir
from gallery.schema import ROOT

DATA = data_dir()   # GALLERY_DATA_DIR, default data/
ITEMS = DATA / "items.parquet"
SUMMARIES = DATA / "summaries.jsonl"
FAILURES = ROOT / "reports" / "curate_failures.jsonl"
MAX_ATTEMPTS = 3

log = logging.getLogger("gallery.curate")


class Summarizer(Protocol):
    def summarize(self, item: dict) -> str: ...


def load_done(path: Path) -> dict[str, str]:
    """id -> summary for every parseable line."""
    done: dict[str, str] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # a torn write from a crash; that id is simply redone
                log.warning("ignoring unparseable line %d in %s", lineno, path)
                continue
            done[rec["id"]] = rec["summary"]
    return done


class _Appender:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # if the file ends without a newline (torn write), start a fresh line
        needs_nl = False
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_nl = f.read(1) != b"\n"
        self.f = path.open("a", encoding="utf-8")
        if needs_nl:
            self.f.write("\n")
        self.lock = threading.Lock()

    def write(self, rec: dict) -> None:
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()


def _attempt(client: Summarizer, item: dict) -> tuple[str | None, list[str]]:
    errors = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.summarize(item), errors
        except Exception as e:  # noqa: BLE001 - every failure is recorded per item, not swallowed
            errors.append(f"attempt {attempt}: {type(e).__name__}: {e}")
            log.warning("%s attempt %d failed: %s", item["id"], attempt, e)
    return None, errors


def run(items: pd.DataFrame, client: Summarizer, summaries: Path = SUMMARIES,
        failures: Path = FAILURES, workers: int = 1, limit: int | None = None) -> dict:
    done = load_done(summaries)
    todo = [r for r in items.to_dict("records") if r["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    log.info("items=%d already_done=%d todo=%d workers=%d", len(items), len(done), len(todo), workers)
    out, fail = _Appender(summaries), _Appender(failures)
    stats = {"ok": 0, "failed": 0, "skipped": len(done)}
    lock = threading.Lock()
    t0 = time.perf_counter()

    def handle(item: dict) -> None:
        summary, errors = _attempt(client, item)
        if summary is not None:
            out.write({"id": item["id"], "summary": summary})
        else:
            fail.write({"id": item["id"], "errors": errors, "ts": time.time()})
        with lock:
            stats["ok" if summary is not None else "failed"] += 1
            n = stats["ok"] + stats["failed"]
        if n % 10 == 0 or n == len(todo):
            dt = time.perf_counter() - t0
            rate = n / dt if dt else 0.0
            eta_h = (len(todo) - n) / rate / 3600 if rate else float("nan")
            log.info("progress %d/%d ok=%d failed=%d %.2f items/min eta=%.2fh",
                     n, len(todo), stats["ok"], stats["failed"], rate * 60, eta_h)

    try:
        if workers <= 1:
            for item in todo:
                handle(item)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(handle, it) for it in todo]
                for fut in as_completed(futs):
                    fut.result()  # surface bugs in handle() itself
    finally:
        out.close()
        fail.close()
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from gallery.llm import GemmaClient

    if not ITEMS.exists():
        log.error("%s not found", ITEMS)
        return 2
    items = pd.read_parquet(ITEMS)
    # curate owns the 3-attempt budget, so the client makes one call per attempt
    client = GemmaClient(max_attempts=1)
    if not client.health():
        log.error("Gemma server not healthy at %s", client.base_url)
        return 2
    stats = run(items, client, workers=args.workers, limit=args.limit)
    log.info("done: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
