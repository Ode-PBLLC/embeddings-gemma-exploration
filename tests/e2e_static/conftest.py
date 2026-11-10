"""E2E fixtures for the static site (scripts/build_static.py): serves dist/site with a plain file server,
or tests a deployed copy when STATIC_SITE_URL is set. Headless Chromium with SwiftShader WebGL, as in tests/e2e.

    uv run pytest tests/e2e_static                                  # local build, model-free checks
    RUN_MODEL_E2E=1 uv run pytest tests/e2e_static                  # also download the model and search
    STATIC_SITE_URL=https://.../ RUN_MODEL_E2E=1 uv run pytest tests/e2e_static
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "dist" / "site"
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


@pytest.fixture(scope="session")
def site_url():
    url = os.environ.get("STATIC_SITE_URL")
    if url:
        yield url.rstrip("/") + "/"
        return
    if not (SITE / "index.html").is_file():
        pytest.skip("no static build: run scripts/build_static.py")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "-b", "127.0.0.1", "-d", str(SITE)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + 20
    while True:
        try:
            urllib.request.urlopen(url, timeout=1).close()
            break
        except OSError:
            if proc.poll() is not None or time.time() > deadline:
                proc.kill()
                raise RuntimeError("static file server did not start")
            time.sleep(0.2)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(args=CHROMIUM_ARGS)
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    pg.errors = []
    pg.requests = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("request", lambda r: pg.requests.append(r.url))
    yield pg
    ctx.close()
    assert not pg.errors, f"page errors: {pg.errors}"
