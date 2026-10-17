"""US-020: open the wall as a full page and come back."""
from e2e_helpers import ask


def test_open_back_and_escape(page, k):
    ask(page, "winter snow")
    res = page.evaluate("window.__gallery.lastResult")
    page.click("#open-page")
    page.wait_for_function("window.__gallery.phase === 'page'")
    assert page.locator("#page").is_visible()
    assert page.locator("#page-hang .piece").count() == k
    links = page.eval_on_selector_all("#page-sources li a", "els => els.map(e => e.href)")
    assert len(links) == k
    urls = page.evaluate("(r) => r.neighbors.map(n => n.id)", res)
    assert len(urls) == k
    page.click("#back-to-cloud")
    page.wait_for_function("window.__gallery.phase === 'wall'")
    page.wait_for_function("getComputedStyle(document.getElementById('page')).visibility === 'hidden'")
    assert page.evaluate("window.__gallery.lineCount") == k  # cloud state intact
    page.click("#open-page")
    page.wait_for_function("window.__gallery.phase === 'page'")
    page.keyboard.press("Escape")
    page.wait_for_function("window.__gallery.phase === 'wall'")
    assert page.evaluate("window.__gallery.billboardCount") == k


def test_footer_links_match_items(page, items):
    ask(page, "bronze sculpture")
    page.click("#open-page")
    res = page.evaluate("window.__gallery.lastResult")
    hrefs = page.eval_on_selector_all("#page-sources li a", "els => els.map(e => e.getAttribute('href'))")
    assert hrefs == [items[n["index"]]["source_url"] for n in res["neighbors"]]
