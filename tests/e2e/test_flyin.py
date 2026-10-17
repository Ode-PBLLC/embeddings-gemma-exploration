"""US-017: query marker, fly-in framing and phase order."""
from pathlib import Path

from e2e_helpers import ROOT, ask, record_phases


def test_phases_in_order(page):
    record_phases(page)
    ask(page, "rings of saturn")
    phases = page.evaluate("window.__phases")
    assert phases == ["idle", "flying", "lines", "wall"], phases


def test_framing_contains_query_and_neighbors(page, k):
    ask(page, "a rocket launch")
    s = page.evaluate("window.__gallery.screen()")
    vw, vh, wall = s["viewport"]["w"], s["viewport"]["h"], s["viewport"]["wall"]
    pts = [s["query"], *s["neighbors"]]
    assert len(s["neighbors"]) == k
    for p in pts:
        assert 0 <= p["x"] <= vw - wall, (p, vw, wall)  # left of the wall panel
        assert 0 <= p["y"] <= vh, p
        assert -1 <= p["z"] <= 1, p  # in front of the camera, inside the frustum


def test_flyin_timing(page):
    ask(page, "the surface of mars")
    t = page.evaluate("window.__gallery.timings")
    assert 850 <= t["fly"] <= 1500, t


def test_flyin_screenshot(page):
    out = ROOT / "reports" / "screens"
    out.mkdir(parents=True, exist_ok=True)
    ask(page, "water lilies in a garden pond", until="lines")
    page.screenshot(path=str(out / "synthetic_flyin_lilies.png"))
    assert Path(out / "synthetic_flyin_lilies.png").stat().st_size > 10_000
