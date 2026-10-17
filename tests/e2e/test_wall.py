"""US-019: the gallery wall panel."""


from e2e_helpers import ask


def test_pieces_in_rank_order(page, k, items):
    ask(page, "portrait of a woman")
    ranks = page.eval_on_selector_all("#hang .piece", "els => els.map(e => +e.dataset.rank)")
    assert sorted(ranks) == list(range(1, k + 1))
    # DOM order follows rank within the hang (lead first, then the columns)
    assert ranks[0] == 1
    res = page.evaluate("window.__gallery.lastResult")
    by_rank = {n["rank"]: n["index"] for n in res["neighbors"]}
    for el in page.query_selector_all("#hang .piece"):
        idx = int(el.get_attribute("data-index"))
        assert by_rank[int(el.get_attribute("data-rank"))] == idx
        kind = "image" if items[idx]["modality"] == "image" else "text"
        assert f"piece--{kind}" in el.get_attribute("class")
    assert page.locator("#wall").evaluate("e => e.classList.contains('is-open')")


def test_image_hover_shows_summary_fast(page, items):
    ask(page, "still life with fruit")
    img = page.locator("#hang .piece--image").first
    assert img.count() == 1, "query should return at least one image item"
    img.scroll_into_view_if_needed()
    page.wait_for_timeout(700)  # let the hang-in animation finish
    img.hover()
    page.wait_for_timeout(150)
    label = page.locator("#wall-label")
    assert float(label.evaluate("e => getComputedStyle(e).opacity")) >= 0.95
    idx = int(img.get_attribute("data-index"))
    assert label.locator(".wall-label-text").inner_text() == items[idx]["summary"]
    # the label hangs beside the frame, fully on screen, not over the picture
    lb, fb = label.bounding_box(), img.locator(".frame").bounding_box()
    vw, vh = page.viewport_size["width"], page.viewport_size["height"]
    assert lb["x"] >= 0 and lb["y"] >= 0 and lb["x"] + lb["width"] <= vw and lb["y"] + lb["height"] <= vh
    assert lb["x"] + lb["width"] <= fb["x"] or lb["x"] >= fb["x"] + fb["width"]
    page.mouse.move(100, 100)
    page.wait_for_timeout(250)
    assert float(label.evaluate("e => getComputedStyle(e).opacity")) <= 0.05


def test_hover_pulses_point(page):
    ask(page, "the moon")
    for el in page.query_selector_all("#hang .piece")[:4]:
        el.scroll_into_view_if_needed()
        el.hover()
        assert page.evaluate("window.__gallery.pulsedIndex") == int(el.get_attribute("data-index"))
    # move to a spot of the cloud clear of every hung picture (the callout layout varies by query)
    rects = page.evaluate("window.__gallery.billboardRects()")
    spot = next((x, y) for y in range(120, 700, 40) for x in range(40, 700, 40)
                if all(abs(x - r["x"]) > r["w"] or abs(y - r["y"]) > r["h"] for r in rects))
    page.mouse.move(*spot)
    page.wait_for_function("window.__gallery.pulsedIndex === null", timeout=1000)
