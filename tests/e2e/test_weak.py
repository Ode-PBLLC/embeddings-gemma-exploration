"""US-021: weak queries are labelled honestly; quota mode shows its fine print."""
import json

from e2e_helpers import ask


def _stub(page, **override):
    def handle(route):
        resp = route.fetch()
        body = resp.json() | override
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))
    page.route("**/search", handle)


def test_weak_stubbed(page):
    _stub(page, weak=True)
    ask(page, "a storm at sea")
    assert page.locator("#weak-badge").is_visible()
    assert page.inner_text("#weak-badge").strip().lower() == "nothing close"
    assert page.inner_text("#wall-kicker").strip().lower() == "weak matches"


def test_not_weak_has_no_badge(page):
    ask(page, "a storm at sea")
    assert not page.locator("#weak-badge").is_visible()
    assert page.inner_text("#wall-kicker").strip().lower() == "nearest works"


def test_weak_with_fake_embedder(page):
    ask(page, "quarterly tax filing deadlines")
    assert page.evaluate("window.__gallery.lastResult.weak") is True
    assert page.locator("#weak-badge").is_visible()


def test_quota_fine_print(page):
    _stub(page, mode="quota")
    ask(page, "the moon")
    assert page.locator("#wall-fine").is_visible()
    assert "balanced across sources" in page.inner_text("#wall-fine").lower()
