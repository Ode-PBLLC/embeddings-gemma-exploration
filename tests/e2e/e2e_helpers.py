"""Shared helpers for the e2e tests (kept out of conftest.py so they can be imported by name)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ask(page, query: str, until: str = "wall", timeout: int = 20000) -> None:
    page.fill("#q", query)
    page.press("#q", "Enter")
    if until:
        page.wait_for_function(f"window.__gallery.phase === '{until}'", timeout=timeout)


def record_phases(page) -> None:
    page.evaluate("""() => { window.__phases = [window.__gallery.phase];
      window.__phaseTimer = setInterval(() => { const p = window.__gallery.phase;
        if (window.__phases[window.__phases.length - 1] !== p) window.__phases.push(p); }, 5); }""")
