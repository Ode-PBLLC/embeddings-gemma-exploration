# Embedding the parity queries in a real browser

`index.html` embeds every query in `../queries.json` with Transformers.js and leaves the vectors on `window.RESULT`.
`?dtype=q8&device=webgpu&all=1` picks the build and the runtime.

WSL's Chromium has no WebGPU adapter, so drive Edge on Windows:

1. Copy `../queries.json` next to `index.html`, then serve this folder from WSL:
   `python -m http.server <port> -b 127.0.0.1`.
2. In a Windows folder: `npm i playwright-core`, copy `edge_probe.mjs` there, and run
   `node edge_probe.mjs http://localhost:<port> q8@webgpu fp16@webgpu q4@webgpu fp16@wasm`.
   It writes `browser_<dtype>_<device>.json` per run.
3. `uv run python test_scripts/static_parity/browser/to_f32.py <that folder>`, then `analyze.py`.
