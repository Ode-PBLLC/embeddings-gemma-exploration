"""US-023: run the five demo queries in the real UI and check the acceptance numbers.

Needs a warm server (scripts/serve.sh) on BASE (default http://localhost:8000).
Measures Enter -> wall panel open and visible, and the source mix of each wall.
Writes reports/demo_check.md and screenshots; exits non-zero if a threshold fails.
Headless Chromium renders with SwiftShader, so timings here are pessimistic versus a GPU browser.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import yaml
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("BASE", "http://localhost:8000")
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
MAX_SECONDS = 3.0


def main() -> int:
    spec = yaml.safe_load((ROOT / "eval" / "queries.yaml").read_text())
    queries = [e["q"] for e in spec["on_topic"] if e.get("demo")]
    assert len(queries) == 5, queries
    shots = ROOT / "reports" / "screens"
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=ARGS)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.goto(BASE)
        page.wait_for_function("window.__gallery && window.__gallery.ready", timeout=60000)
        page.fill("#q", "warm up")  # first query pays one-off shader/texture costs
        page.press("#q", "Enter")
        page.wait_for_function("window.__gallery.phase === 'wall'", timeout=30000)
        for q in queries:
            page.fill("#q", q)
            t0 = time.perf_counter()
            page.press("#q", "Enter")
            page.wait_for_function(
                "window.__gallery.phase === 'wall' && document.querySelector('#wall').classList.contains('is-open')"
                f" && window.__gallery.lastResult && window.__gallery.lastResult.query === {q!r}",
                timeout=30000,
            )
            page.locator("#wall [data-rank='1']").wait_for(state="visible", timeout=5000)
            elapsed = time.perf_counter() - t0
            res = page.evaluate("window.__gallery.lastResult")
            srcs = [n["id"].split(":")[0] for n in res["neighbors"]]
            counts = {s: srcs.count(s) for s in ("aic", "nasa", "wiki")}
            slug = q.replace(" ", "_")
            page.wait_for_timeout(400)
            page.screenshot(path=str(shots / f"demo_{slug}.png"))
            rows.append({"query": q, "seconds": elapsed, **counts, "weak": res["weak"],
                         "mixed": sum(v > 0 for v in counts.values()) >= 2})
        browser.close()

    n_mixed = sum(r["mixed"] for r in rows)
    slow = [r for r in rows if r["seconds"] > MAX_SECONDS]
    lines = ["# Demo check", "", f"Server: {BASE}. Browser: headless Chromium (SwiftShader).", "",
             "| query | Enter→wall (s) | aic | nasa | wiki | mixed | weak |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['query']} | {r['seconds']:.2f} | {r['aic']} | {r['nasa']} | {r['wiki']} | {r['mixed']} | {r['weak']} |"
              for r in rows]
    lines += ["", f"- Mixed walls: {n_mixed}/5 (need >= 4)",
              f"- Over {MAX_SECONDS:.1f} s: {len(slow)} (need 0)"]
    ok = n_mixed >= 4 and not slow
    lines.append(f"- Result: {'PASS' if ok else 'FAIL'}")
    out = ROOT / "reports" / "demo_check.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
