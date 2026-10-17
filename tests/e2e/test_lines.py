"""US-018: neighbour lines, thumbnails, dimming."""
import pytest

from e2e_helpers import ROOT, ask


def test_lines_and_billboards(page, k):
    ask(page, "astronaut on a spacewalk")
    assert page.evaluate("window.__gallery.lineCount") == k
    page.wait_for_function(f"window.__gallery.billboardCount === {k}", timeout=3000)
    assert page.evaluate("window.__gallery.dim") <= 0.25


def test_line_opacity_mapping(page):
    ask(page, "nebula and stars")
    ops = page.evaluate("window.__gallery.lineOpacities")
    sims = [n["similarity"] for n in page.evaluate("window.__gallery.lastResult.neighbors")]
    lo, hi = min(sims), max(sims)
    for o, s in zip(ops, sims):
        assert o == pytest.approx(0.35 + 0.65 * (s - lo) / (hi - lo), abs=1e-6)
    assert max(ops) == pytest.approx(1.0) and min(ops) == pytest.approx(0.35)


def test_lines_phase_duration(page):
    ask(page, "japanese woodblock print")
    t = page.evaluate("window.__gallery.timings")
    assert t["lines"] <= 1400, t


def test_lines_screenshot(page):
    out = ROOT / "reports" / "screens"
    ask(page, "abstract color", until="lines")
    page.wait_for_timeout(500)
    page.screenshot(path=str(out / "synthetic_lines_abstract.png"))


@pytest.mark.parametrize("q", ["water lilies in a garden pond", "the moon", "a storm at sea"])
def test_billboards_spread_apart(page, k, q):
    """Hung pieces in the cloud should not pile on top of each other."""
    ask(page, q)
    page.wait_for_function(f"window.__gallery.billboardCount === {k}", timeout=3000)
    page.wait_for_timeout(500)
    rects = page.evaluate("window.__gallery.billboardRects()")
    worst = 0.0
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            # visible frame is ~80% of the sprite (the rest is the drop-shadow margin)
            ix = max(0.0, min(a["x"] + .4 * a["w"], b["x"] + .4 * b["w"]) - max(a["x"] - .4 * a["w"], b["x"] - .4 * b["w"]))
            iy = max(0.0, min(a["y"] + .4 * a["h"], b["y"] + .4 * b["h"]) - max(a["y"] - .4 * a["h"], b["y"] - .4 * b["h"]))
            small = min(.64 * a["w"] * a["h"], .64 * b["w"] * b["h"])
            worst = max(worst, ix * iy / small)
    assert worst <= 0.10, f"billboards overlap by {worst:.0%} of the smaller piece"
