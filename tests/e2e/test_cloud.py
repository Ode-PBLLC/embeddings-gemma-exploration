"""US-015: the cloud renders every point, orbits when idle, resumes 5 s after interaction."""
import time


def test_point_count(page, items):
    assert page.evaluate("window.__gallery.pointCount") == len(items)
    assert page.locator("#stage canvas").count() == 1
    assert page.evaluate("window.__gallery.phase") == "idle"


def test_background_token(page):
    bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == "rgb(246, 244, 239)"  # --wall #f6f4ef


def test_idle_orbit_and_resume(page):
    c0 = page.evaluate("window.__gallery.camera")
    time.sleep(0.6)
    c1 = page.evaluate("window.__gallery.camera")
    assert c0 != c1, "camera should auto-orbit when idle"
    box = page.locator("#stage canvas").bounding_box()
    page.mouse.move(box["width"] * 0.3, box["height"] * 0.4)
    page.mouse.down()
    page.mouse.move(box["width"] * 0.4, box["height"] * 0.45, steps=5)
    page.mouse.up()
    assert page.evaluate("window.__gallery.autoRotate") is False
    time.sleep(3.0)
    assert page.evaluate("window.__gallery.autoRotate") is False
    page.wait_for_function("window.__gallery.autoRotate === true", timeout=4000)


def test_bench_overlay(browser, server):
    pg = browser.new_page(viewport={"width": 1280, "height": 800})
    pg.goto(server + "/?bench=1")
    pg.wait_for_function("window.__gallery && window.__gallery.fps > 0", timeout=20000)
    assert pg.locator("#bench").is_visible()
    pg.wait_for_function("/[1-9][0-9]* fps/.test(document.getElementById('bench').textContent)", timeout=3000)
    pg.close()
