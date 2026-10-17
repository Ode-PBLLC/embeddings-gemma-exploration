# Build contracts (read before touching code)

SPEC.md is the intent. PRD.md holds the stories. This file holds the shared contracts and the
decisions taken after review round 1 (Codex). If this file and the PRD disagree, this file wins.

## Layout
- `src/gallery/schema.py`: item columns and `validate()`. Every stage uses it.
- `data/raw/{aic,nasa,wiki}.parquet`: per-source ingestion output (schema columns).
- `data/media/{aic,nasa}/<native_id>.jpg`: ≤768 px long edge. This is the image the models see.
- `data/thumbs/{aic,nasa}/<native_id>.jpg`: ≤320 px long edge. This is the image the UI shows.
- `data/items.parquet`: the merge of all three sources. This is the canonical row order for the whole build.
- `data/summaries.jsonl`: `{"id", "summary"}` per line, written by curation, append-only.
- `data/vectors.npy` + `data/vector_ids.json`: aligned to `items.parquet` order.
- `data/layout.npy`: (N, 3), aligned to `items.parquet` order.
- `config/retrieval.json`: written by eval_gap.
- `web/data/points.bin`, `web/data/items.json`, `web/thumbs/`: front-end bundle, built by `gallery.build_assets`. It only includes items that have a summary.

## Decisions after round 1
1. **Embedding doesn't depend on curation.** Order: ingest → merge → embed → layout → eval_gap → (curation runs in the background on the GPU) → build_assets.
2. **Weak threshold:** calibrated with 15 on-topic and 10 off-topic queries (`eval/queries.yaml`). It is set to the midpoint between the 10th-percentile top-1 similarity of the on-topic queries and the 90th percentile of the off-topic ones. It passes if ≥ 8 of 10 off-topic queries are weak and ≤ 2 of 15 on-topic queries are weak.
3. **three.js is vendored** in `web/vendor/`. No CDN at runtime.
4. **Time budget:** search ≤ 300 ms, fly-in 900 ms, lines ≤ 1000 ms, wall slide-in 300 ms, for ≤ 3.0 s total.
5. **Input:** disabled only while the request is in flight. A submit during an animation cancels it.
6. **Quota mode:** if `mode == "quota"`, the wall shows fine print "balanced across sources".
7. **Review gate:** Mason is away, so it becomes screenshots in `reports/screens/` for him to review on return.
8. **Local LLM serving:** llama.cpp (docker `ghcr.io/ggml-org/llama.cpp:server-cuda`) or vLLM. Never Ollama.
9. **Look and feel:** a museum. Light backgrounds, gentle hover states, pops of color that come from the works.

## Interface decisions (interface workstream, 2026-10-15)
- **Paths are env-overridable** (`gallery.paths`): `GALLERY_DATA_DIR` (default `data/`), `GALLERY_RETRIEVAL` (default `config/retrieval.json`), `GALLERY_ASSETS_DIR` (default `web/`). The app serves `/data` and `/thumbs` from the assets dir and everything else from `web/`.
- **Synthetic set:** `scripts/make_synthetic.py` writes `data/synthetic/` (5,000 invented items, keyword fake embedder centroids in `fake_topics.npz`, its own `retrieval.json`). `scripts/serve.sh --synthetic` runs it with `GALLERY_EMBEDDER=fake`. The UI shows a "synthetic data" note when `meta.json` says so.
- **points.bin layout:** all float32 xyz first (N×12 bytes), then all uint8 rgb (N×3 bytes); still N×15 bytes. Described in `web/data/meta.json`.
- **Asset index = search index.** `/search` only searches the items in `items.json`, so `neighbors[].index` indexes `items.json`/`points.bin`. The query position uses the xyz in `points.bin`, so the cloud and the API cannot disagree.
- **`build_assets --include-pending`** includes un-curated items with `raw_text` (≤420 chars) as the summary and `summary_pending: true`; the UI labels them. Without the flag only curated items are included (the contract).
- **Layout:** UMAP `n_neighbors=15, min_dist=0.1` (library defaults) plus the fixed `n_components=3, metric=cosine, random_state=42`. Re-run `gallery.layout` and `gallery.build_assets` whenever `vectors.npy` changes (e.g. if eval_gap switches to `image+title`).
- **Timings in the UI:** fly-in 900 ms, lines staggered 60 ms × 300 ms each (960 ms for k = 12), wall slide 300 ms. Camera auto-orbits only when idle; after a query the composition holds still unless dragged.
- **Billboards are spread in screen space** after the fly-in so hung thumbnails do not pile up; lines run to the spread position. The underlying dot is hidden, so this moves a piece by tens of pixels from its exact layout position.
- **Frames by source:** AIC pieces in gilt frames, NASA in thin black frames with a wide mat, Wikipedia as text cards. Gemma's note for an image appears as a wall label beside the piece on hover.
- **Fonts and three.js are vendored** (`web/vendor/`): three r170 (`three.module.min.js`, OrbitControls, Line2), Cormorant Garamond and Work Sans woff2 (OFL). No network at runtime.
