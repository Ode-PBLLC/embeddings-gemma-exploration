"""US-022: consecutive queries clear the previous state; a submit mid-animation cancels cleanly."""
from e2e_helpers import ask


def _state(page):
    return page.evaluate("""() => ({ lines: window.__gallery.lineCount, boards: window.__gallery.billboardCount,
      q: window.__gallery.lastResult.query, pieces: document.querySelectorAll('#hang .piece').length,
      idx: [...document.querySelectorAll('#hang .piece')].map(e => +e.dataset.index).sort((a, b) => a - b),
      res: window.__gallery.lastResult.neighbors.map(n => n.index).sort((a, b) => a - b) })""")


def test_second_query_replaces_first(page, k):
    ask(page, "rings of saturn")
    ask(page, "water lilies in a garden pond")
    page.wait_for_function(f"window.__gallery.billboardCount === {k}", timeout=3000)
    s = _state(page)
    assert s["q"] == "water lilies in a garden pond"
    assert s["lines"] == k and s["boards"] == k and s["pieces"] == k
    assert s["idx"] == s["res"]


def test_submit_during_animation(page, k):
    ask(page, "rings of saturn", until="lines")
    page.wait_for_timeout(250)
    ask(page, "portrait of a woman")
    page.wait_for_function(f"window.__gallery.billboardCount === {k}", timeout=3000)
    page.wait_for_timeout(500)
    s = _state(page)
    assert s["q"] == "portrait of a woman"
    assert s["lines"] == k and s["boards"] == k and s["pieces"] == k
    assert s["idx"] == s["res"]


def test_submit_during_flight(page, k):
    ask(page, "the moon", until="flying")
    ask(page, "a rocket launch")
    page.wait_for_function(f"window.__gallery.billboardCount === {k}", timeout=3000)
    page.wait_for_timeout(500)
    s = _state(page)
    assert s["q"] == "a rocket launch" and s["lines"] == k and s["boards"] == k


def test_close_returns_to_idle(page):
    ask(page, "the moon")
    page.click("#close-wall")
    page.wait_for_function("window.__gallery.phase === 'idle'")
    page.wait_for_timeout(1000)
    assert page.evaluate("window.__gallery.lineCount") == 0
    assert page.evaluate("window.__gallery.billboardCount") == 0
    assert page.evaluate("window.__gallery.dim") == 1
