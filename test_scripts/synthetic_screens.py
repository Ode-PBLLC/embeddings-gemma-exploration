"""Capture synthetic-build screenshots: idle cloud, mid-lines, wall, hover, page.

    uv run python test_scripts/synthetic_screens.py [--url http://localhost:8011] [--query "a storm at sea"] [--tag synthetic]
"""
import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8011")
ap.add_argument("--query", default="a storm at sea")
ap.add_argument("--tag", default="synthetic")
a = ap.parse_args()
out = ROOT / "reports" / "screens"
out.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    pg = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    pg.goto(a.url)
    pg.wait_for_function("window.__gallery && window.__gallery.ready", timeout=30000)
    time.sleep(2.5)
    pg.screenshot(path=out / f"{a.tag}_idle.png")
    pg.fill("#q", a.query)
    pg.press("#q", "Enter")
    pg.wait_for_function("window.__gallery.phase === 'lines'", timeout=15000)
    time.sleep(0.55)
    pg.screenshot(path=out / f"{a.tag}_lines.png")
    pg.wait_for_function("window.__gallery.phase === 'wall'", timeout=15000)
    time.sleep(1.4)
    pg.screenshot(path=out / f"{a.tag}_wall.png")
    first = pg.locator("#hang .piece--image").first
    first.hover()
    time.sleep(0.6)
    pg.screenshot(path=out / f"{a.tag}_hover.png")
    pg.click("#open-page")
    time.sleep(1.2)
    pg.screenshot(path=out / f"{a.tag}_page.png")
    print("timings", pg.evaluate("window.__gallery.timings"), "renderer", pg.evaluate("window.__gallery.renderer"))
    print("\n".join(logs) or "no console output")
    b.close()
