"""Curation resume/retry logic with a fake client (offline)."""
import json
from collections import Counter

import pandas as pd
import pytest

from gallery import curate

GOOD = " ".join(["word"] * 60)


def _items(n=8):
    return pd.DataFrame({"id": [f"wiki:{i}" for i in range(n)], "modality": "text",
                         "title": "t", "raw_text": "r"})


class Interrupt(BaseException):
    """Simulates Ctrl-C / kill: not an Exception, so curate does not treat it as an item failure."""


class FakeClient:
    def __init__(self, interrupt_after=None, always_fail=()):
        self.calls = Counter()
        self.interrupt_after = interrupt_after
        self.always_fail = set(always_fail)
        self.ok = 0

    def summarize(self, item):
        if self.interrupt_after is not None and self.ok >= self.interrupt_after:
            raise Interrupt()  # process killed before this item reached the model
        self.calls[item["id"]] += 1
        if item["id"] in self.always_fail:
            raise RuntimeError("boom")
        self.ok += 1
        return GOOD


@pytest.mark.parametrize("workers", [1])
def test_resume_calls_each_item_once(tmp_path, workers):
    items = _items()
    s, f = tmp_path / "s.jsonl", tmp_path / "f.jsonl"
    c1 = FakeClient(interrupt_after=3)
    with pytest.raises(Interrupt):
        curate.run(items, c1, summaries=s, failures=f, workers=workers)
    assert len(curate.load_done(s)) == 3
    c2 = FakeClient()
    stats = curate.run(items, c2, summaries=s, failures=f, workers=workers)
    assert stats["skipped"] == 3 and stats["ok"] == 5
    total = c1.calls + c2.calls
    assert set(total) == set(items["id"]) and all(v == 1 for v in total.values()), total
    done = curate.load_done(s)
    assert list(done) == items["id"].tolist()
    # third run does nothing
    c3 = FakeClient()
    curate.run(items, c3, summaries=s, failures=f)
    assert sum(c3.calls.values()) == 0


def test_persistent_failure_goes_to_failures(tmp_path):
    items = _items(5)
    s, f = tmp_path / "s.jsonl", tmp_path / "f.jsonl"
    c = FakeClient(always_fail={"wiki:2"})
    stats = curate.run(items, c, summaries=s, failures=f, workers=2)
    assert stats == {"ok": 4, "failed": 1, "skipped": 0, "seconds": stats["seconds"]}
    assert c.calls["wiki:2"] == curate.MAX_ATTEMPTS == 3
    assert "wiki:2" not in curate.load_done(s)
    fails = [json.loads(l) for l in f.read_text().splitlines()]
    assert [r["id"] for r in fails] == ["wiki:2"]
    assert len(fails[0]["errors"]) == 3


def test_torn_line_is_redone(tmp_path):
    items = _items(3)
    s, f = tmp_path / "s.jsonl", tmp_path / "f.jsonl"
    s.write_text(json.dumps({"id": "wiki:0", "summary": GOOD}) + "\n" + '{"id": "wiki:1", "summ')
    c = FakeClient()
    curate.run(items, c, summaries=s, failures=f)
    assert dict(c.calls) == {"wiki:1": 1, "wiki:2": 1}
    assert list(curate.load_done(s)) == ["wiki:0", "wiki:1", "wiki:2"]


def test_llm_word_window():
    from gallery.llm import _clean, word_count
    assert word_count(_clean('"' + GOOD + '"')) == 60
    assert _clean("**Label** text") == "Label text"


def test_check_summaries_threshold(tmp_path):
    from gallery.check_summaries import check
    items = _items(100)
    p = tmp_path / "items.parquet"
    items.to_parquet(p)
    s = tmp_path / "s.jsonl"
    lines = [json.dumps({"id": f"wiki:{i}", "summary": GOOD}) for i in range(98)]
    s.write_text("\n".join(lines) + "\n")
    assert check(p, s)[0]
    s.write_text("\n".join(lines[:97] + [json.dumps({"id": "wiki:97", "summary": "too short"})]) + "\n")
    assert not check(p, s)[0]
