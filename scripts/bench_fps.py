"""Idle-orbit fps check (US-015, SPEC N2).

    uv run python scripts/bench_fps.py [--url http://localhost:8000] [--seconds 10] [--headless]

Opens the page in headed Chromium with ?bench=1 and reads window.__gallery.fps (rolling 2 s median)
once a second for --seconds. Exit codes: 0 median >= 55 fps, 1 below, 2 "UNVERIFIED: no GPU" when the
WebGL renderer is a software rasteriser (SwiftShader / llvmpipe), 3 the page did not load.
The server must already be running (scripts/serve.sh).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

SOFTWARE = ("swiftshader", "llvmpipe", "softpipe", "software")
TARGET = 55


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--headless", action="store_true", help="headless (usually software GL, so UNVERIFIED)")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    a = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=a.headless, args=["--ignore-gpu-blocklist", "--enable-gpu-rasterization"])
        page = browser.new_page(viewport={"width": a.width, "height": a.height})
        try:
            page.goto(a.url.rstrip("/") + "/?bench=1", timeout=30000)
            page.wait_for_function("window.__gallery && window.__gallery.ready", timeout=30000)
        except PlaywrightError as e:
            print(f"page did not load: {e}", file=sys.stderr)
            return 3
        renderer = page.evaluate("window.__gallery.renderer")
        points = page.evaluate("window.__gallery.pointCount")
        print(f"renderer: {renderer}\npoints: {points}")
        if any(s in renderer.lower() for s in SOFTWARE):
            print("UNVERIFIED: no GPU")
            browser.close()
            return 2
        time.sleep(2.5)  # fill the 2 s rolling window
        samples = []
        for _ in range(a.seconds):
            samples.append(page.evaluate("window.__gallery.fps"))
            time.sleep(1)
        browser.close()
    med = statistics.median(samples)
    print(f"fps samples: {samples}\nmedian: {med} (target >= {TARGET})")
    return 0 if med >= TARGET else 1


if __name__ == "__main__":
    sys.exit(main())
