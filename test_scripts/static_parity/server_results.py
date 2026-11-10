"""Record the running server's answers (the reference) for every parity query.

    uv run python test_scripts/static_parity/server_results.py [--base http://localhost:8000]

Writes out/server.json: {"gallery": [search result, ...], "moon": [...]}, in queries.json order.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def search(base: str, query: str) -> dict:
    req = urllib.request.Request(f"{base}/search", data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    args = ap.parse_args()
    sets = json.loads((HERE / "queries.json").read_text())
    out = {name: [search(args.base + prefix, q["q"]) for q in sets[name]]
           for name, prefix in (("gallery", ""), ("moon", "/moon"))}
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "server.json").write_text(json.dumps(out))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
