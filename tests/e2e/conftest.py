"""E2E fixtures: a uvicorn server on the synthetic dataset with the keyword fake embedder, and
headless Chromium with SwiftShader WebGL (functional checks only; fps is measured by scripts/bench_fps.py)."""
from __future__ import annotations

import json
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
SYN = ROOT / "data" / "synthetic"
ASSETS = SYN / "web"
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_synthetic() -> None:
    if not (SYN / "layout.npy").is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py")], check=True, cwd=ROOT)
    if not (ASSETS / "data" / "items.json").is_file():
        subprocess.run([sys.executable, "-m", "gallery.build_assets", "--data-dir", str(SYN), "--out", str(ASSETS)],
                       check=True, cwd=ROOT)


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    ensure_synthetic()
    log_path = tmp_path_factory.mktemp("server") / "uvicorn.log"
    log = log_path.open("wb")  # a file, not a pipe: an unread pipe fills up and blocks the server
    port = _free_port()
    env = os.environ | {
        "GALLERY_DATA_DIR": str(SYN), "GALLERY_ASSETS_DIR": str(ASSETS),
        "GALLERY_RETRIEVAL": str(SYN / "retrieval.json"), "GALLERY_EMBEDDER": "fake",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "gallery.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while True:
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=1) as r:
                if r.status == 200:
                    break
        except OSError:
            pass
        if proc.poll() is not None or time.time() > deadline:
            proc.kill()
            out = log_path.read_text(errors="replace")
            raise RuntimeError(f"server did not start:\n{out}")
        time.sleep(0.3)
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


@pytest.fixture(scope="session")
def items():
    return json.loads((ASSETS / "data" / "items.json").read_text())


@pytest.fixture(scope="session")
def k():
    return json.loads((SYN / "retrieval.json").read_text())["k"]


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(args=CHROMIUM_ARGS)
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server)
    pg.wait_for_function("window.__gallery && window.__gallery.ready", timeout=30000)
    yield pg
    ctx.close()
    assert not errors, f"page errors: {errors}"
