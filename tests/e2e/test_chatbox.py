"""US-016: the chat box."""
import json
import time


def test_focus_on_load(page):
    assert page.evaluate("document.activeElement.id") == "q"
    assert page.get_attribute("#q", "placeholder") == "Ask the collection…"


def test_empty_input_sends_nothing(page):
    sent = []
    page.on("request", lambda r: sent.append(r.url) if r.url.endswith("/search") else None)
    page.fill("#q", "   ")
    page.press("#q", "Enter")
    time.sleep(0.4)
    assert sent == []
    assert page.evaluate("window.__gallery.phase") == "idle"


def test_server_error_shown_inline(page):
    page.route("**/search", lambda r: r.fulfill(status=500, content_type="application/json",
                                                body=json.dumps({"detail": "search failed: RuntimeError: CUDA out of memory"})))
    page.fill("#q", "a storm at sea")
    page.press("#q", "Enter")
    page.wait_for_selector("#ask-error:not([hidden])")
    assert "CUDA out of memory" in page.inner_text("#ask-error")
    assert page.evaluate("window.__gallery.phase") == "idle"
    assert page.evaluate("window.__gallery.lastResult") is None
    assert page.is_enabled("#q")


def test_422_detail_shown(page):
    page.route("**/search", lambda r: r.fulfill(status=422, content_type="application/json",
                                                body=json.dumps({"detail": "query is empty"})))
    page.fill("#q", "x")
    page.press("#q", "Enter")
    page.wait_for_selector("#ask-error:not([hidden])")
    assert page.inner_text("#ask-error") == "query is empty"


def test_disabled_while_in_flight(page):
    # record inside the page: the Python route handler below blocks Playwright's own polling
    page.evaluate("""() => { window.__sawDisabled = false;
      new MutationObserver(() => { if (document.getElementById('q').disabled) window.__sawDisabled = true; })
        .observe(document.getElementById('q'), { attributes: true }); }""")

    def slow(route):
        time.sleep(0.8)
        route.continue_()
    page.route("**/search", slow)
    page.fill("#q", "the moon")
    page.press("#q", "Enter")
    page.wait_for_function("window.__gallery.phase === 'wall'", timeout=20000)
    assert page.evaluate("window.__sawDisabled") is True
    assert page.is_enabled("#q")
