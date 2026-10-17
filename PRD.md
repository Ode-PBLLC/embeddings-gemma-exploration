# Embedding Gallery — PRD

Source of truth: `SPEC.md`. This PRD adds detail to it and must not contradict it.

Review status: round 1 had a Codex critique (14 issues). Gemini could not run because `GOOGLE_CLOUD_PROJECT` is not set. All issues were accepted and resolved as contracts in `BUILD.md`, which overrides this PRD where they differ. The debate stopped after round 1 because Mason asked for an immediate end-to-end build.

## Executive Summary

Embedding Gallery is an internal demo that makes semantic search visible. About 5,000
items (Art Institute of Chicago artworks, NASA images, and Wikipedia paragraphs) float as
a 3D cloud of colored points on an off-white background. The presenter types a text
query. EmbeddingGemma 2 embeds it, the query appears as a point in the cloud, and the
camera flies in. Lines spread from the query to its 12 nearest neighbors, the neighbors
turn into thumbnails, and a side panel hangs them as a gallery wall. Every item has a
one-paragraph summary that Gemma 4 E4B wrote offline. The wall can be expanded to a
full page.

The demo runs entirely on Mason's WSL2 machine (RTX 3070 Ti, 8 GB). Mason presents it to
Ode colleagues; it is not a self-serve product.

## Problem Statement

Embeddings are hard to explain to colleagues and clients. A table of cosine scores doesn't
show the idea. Ode needs a demo that lets people *see* a query land in a semantic space,
see what is near it across modalities, and see that "near" means "related".

## Target Users

- **Presenter (Mason):** runs a handful of scripted and ad-hoc queries in front of an
  audience. Needs reliability, speed and visual impact. Will tolerate rough edges
  outside the demo path.
- **Audience (Ode colleagues):** watch. They need to understand at a glance what
  happened: query → point → neighbors → gallery.

## Conventions used in this PRD

- Paths are relative to the repo root. Python is managed with `uv`. Tests use `pytest`.
- `data/` holds generated artifacts and is gitignored. `reports/` holds measured results
  as Markdown or JSON.
- Tests marked `@pytest.mark.gpu` need the real models. Tests marked `@pytest.mark.net`
  need internet. Plain tests run offline with fixtures.
- Front-end checks use Playwright (Python) against `http://localhost:8000`, started by
  `scripts/serve.sh`. The page exposes a read-only `window.__gallery` debug object
  (point count, current state, last search result, animation phase) for tests.

## User Stories

### US-001: Project scaffold
**As a** developer **I want** a uv project with a FastAPI app, static web dir and test setup **so that** every later story has a home.

**Acceptance Criteria:**
- [ ] `pyproject.toml` defines the package `gallery` (in `src/gallery/`) and the pytest markers `gpu` and `net`.
- [ ] `uv run pytest -q -m "not gpu and not net"` passes.
- [ ] `scripts/serve.sh` starts uvicorn on port 8000. `GET /healthz` returns 200 `{"ok": true}`, and `GET /` serves `web/index.html`.
- [ ] `.gitignore` excludes `data/`, `.venv/` and model caches.

**Verification:**
```bash
uv run pytest -q tests/test_smoke.py
```
**Priority:** P0

---

### US-002: EmbeddingGemma 2 feasibility
**As a** developer **I want** EmbeddingGemma 2 loaded locally and embedding text and images **so that** we know the core model works on this GPU before building on it.

**Acceptance Criteria:**
- [ ] `gallery.embed.Embedder` exposes `embed_texts(list[str]) -> np.ndarray` and `embed_images(list[Path], titles: list[str] | None) -> np.ndarray`. Both return float32 arrays of shape `(n, 768)` with rows L2-normalized (norms within 1e-3 of 1).
- [ ] The query and document prompt prefixes follow the model card. Which ones were used is recorded in `reports/phase0.md`.
- [ ] A sanity test: given a text "a spiral galaxy", the image `tests/fixtures/galaxy.jpg` scores higher cosine similarity than `tests/fixtures/still_life.jpg`, and the reverse holds for "a bowl of fruit".
- [ ] `reports/phase0.md` records peak VRAM, the loading path (HF or Ollama), and p50/p95 latency for a single text query over 50 runs.

**Verification:**
```bash
uv run pytest -q -m gpu tests/test_embed.py && test -f reports/phase0.md
```
**Priority:** P0

---

### US-003: Gemma 4 E4B feasibility
**As a** developer **I want** `gemma4:e4b` running in Ollama with image input **so that** offline curation is possible.

**Acceptance Criteria:**
- [ ] `gallery.curate.GemmaClient.summarize(item) -> str` calls Ollama at `http://localhost:11434` with model `gemma4:e4b`. For an image item it sends the image.
- [ ] For `tests/fixtures/galaxy.jpg` with a title, it returns 40–120 words, and the result mentions at least one of {galaxy, spiral, stars}.
- [ ] `reports/phase0.md` records seconds per image item (mean over 10).

**Verification:**
```bash
uv run pytest -q -m gpu tests/test_gemma.py
```
**Priority:** P0

---

### US-004: Item schema
**As a** developer **I want** one validated item schema **so that** all sources merge cleanly.

**Acceptance Criteria:**
- [ ] `gallery.schema` defines the item columns `id, source, modality, title, raw_text, summary, media_path, thumb_path, source_url, license, color_hex, topic` and a `validate(df)` function.
- [ ] `validate` rejects: duplicate `id`; `source` not in {aic, nasa, wiki}; `modality` not in {image, text}; image items without an existing `thumb_path`; `color_hex` not matching `^#[0-9a-f]{6}$`; empty `title` or `source_url`.
- [ ] Each rejection rule has a unit test.

**Verification:**
```bash
uv run pytest -q tests/test_schema.py
```
**Priority:** P0

---

### US-005: Ingest AIC
**As a** presenter **I want** about 1,700 CC0 artworks with color metadata **so that** the cloud has beautiful, varied art.

**Acceptance Criteria:**
- [ ] `uv run python -m gallery.ingest.aic` writes `data/raw/aic.parquet` and thumbnails to `data/thumbs/aic/`.
- [ ] Only items with `is_public_domain = true` and an `image_id` are kept. The item count is between 1,600 and 1,800.
- [ ] `color_hex` comes from the API's `color` (HSL), converted to hex.
- [ ] No single `department_title` holds more than 25% of the items. `topic` is set to the department.
- [ ] Requests are rate-limited to ≤ 1 per second and send a User-Agent that identifies the project. Failed downloads are retried up to 3 times and then logged to `reports/ingest_failures.jsonl`.
- [ ] The output passes `gallery.schema.validate`.

**Verification:**
```bash
uv run pytest -q tests/test_ingest_aic.py && uv run python -m gallery.check_corpus --source aic
```
**Priority:** P0

---

### US-006: Ingest NASA
**As a** presenter **I want** about 1,700 NASA images across subjects **so that** space imagery sits next to the art.

**Acceptance Criteria:**
- [ ] `uv run python -m gallery.ingest.nasa` queries the NASA Image and Video Library (`media_type=image`) over a fixed keyword list of at least 12 subjects (e.g. nebula, galaxy, Earth from orbit, Mars, astronaut, launch, Jupiter, aurora, Moon, Saturn, comet, ISS). It writes `data/raw/nasa.parquet` and thumbnails.
- [ ] The item count is between 1,600 and 1,800, and no single keyword holds more than 15%. `topic` is set to the keyword.
- [ ] `color_hex` is the dominant color from k-means (k=5) on the thumbnail. Near-black and near-white clusters are skipped when another cluster has ≥ 10% of the pixels.
- [ ] Items that are near-duplicates by NASA ID or exact title are dropped.
- [ ] The output passes `gallery.schema.validate`.

**Verification:**
```bash
uv run pytest -q tests/test_ingest_nasa.py && uv run python -m gallery.check_corpus --source nasa
```
**Priority:** P0

---

### US-007: Ingest Wikipedia
**As a** presenter **I want** about 1,700 Wikipedia paragraphs about the artists, movements, missions and objects in the corpus **so that** text cards land near related images.

**Acceptance Criteria:**
- [ ] `uv run python -m gallery.ingest.wiki` builds its article list from the AIC artist titles and style titles plus the NASA keywords, resolves them through the MediaWiki API, and writes `data/raw/wiki.parquet`.
- [ ] Paragraphs have at least 40 words. At most 6 paragraphs come from any one article. The total is between 1,600 and 1,800.
- [ ] `source_url` is the article URL with a section anchor where available. `license` = `CC BY-SA 4.0`. `color_hex` = the single neutral tone defined in `gallery.schema.WIKI_COLOR`.
- [ ] The output passes `gallery.schema.validate`.

**Verification:**
```bash
uv run pytest -q tests/test_ingest_wiki.py && uv run python -m gallery.check_corpus --source wiki
```
**Priority:** P0

---

### US-008: Merge corpus and mix report
**As a** developer **I want** one merged items table and a mix report **so that** later stages read a single file.

**Acceptance Criteria:**
- [ ] `uv run python -m gallery.merge` writes `data/items.parquet` with 4,800–5,400 rows that pass `validate`.
- [ ] Each source holds 30–36% of the rows.
- [ ] `reports/corpus_mix.md` lists counts by source and by topic.

**Verification:**
```bash
uv run python -m gallery.check_corpus --all
```
**Priority:** P0

---

### US-009: Offline Gemma curation
**As a** presenter **I want** every item to have a one-paragraph summary written by Gemma 4 **so that** the wall has text worth reading.

**Acceptance Criteria:**
- [ ] `uv run python -m gallery.curate` fills `summary` for every item. Progress is written to `data/summaries.jsonl` after each item.
- [ ] Restarting skips items that already have a summary (tested with a fake client: run, interrupt after 3, rerun, and assert each item is called exactly once).
- [ ] Each item is tried at most 3 times. Final failures are written to `reports/curate_failures.jsonl`, and those items are excluded from the merged output (tested with a fake client that always fails on one id).
- [ ] At least 98% of items end with a summary of 40–120 words.

**Verification:**
```bash
uv run pytest -q tests/test_curate.py && uv run python -m gallery.check_corpus --summaries
```
**Priority:** P0

---

### US-010: Embed corpus
**As a** developer **I want** a 768-d vector for every curated item **so that** search and layout work.

**Acceptance Criteria:**
- [ ] `uv run python -m gallery.embed_corpus` writes `data/vectors.npy` (float32, `(N, 768)`) and `data/vector_ids.json`, in the same order as `data/items.parquet`.
- [ ] All rows are L2-normalized. None contain NaN.
- [ ] Image items are embedded according to `config/retrieval.json` `image_input` (`image` or `image+title`). Text items are embedded from `raw_text` with the document prompt.

**Verification:**
```bash
uv run pytest -q tests/test_vectors.py
```
**Priority:** P0

---

### US-011: Modality-gap measurement and retrieval policy
**As a** presenter **I want** walls that mix sources **so that** the demo shows cross-modal search, and the method behind them is recorded.

**Acceptance Criteria:**
- [ ] `eval/queries.yaml` contains 15 queries spread over art, space, nature and abstract moods, including the 5 demo queries (US-023).
- [ ] `uv run python -m gallery.eval_gap` reports, for each query, the source counts in the raw top 12 and whether the wall is "mixed" (≥ 2 sources). It writes `reports/modality_gap.md`.
- [ ] It follows the SPEC §6 policy in order: raw → `image+title` re-embed → per-source quota (4/4/4). It stops at the first step where ≥ 12 of the 15 queries are mixed, and records each tried step's numbers.
- [ ] It writes `config/retrieval.json` with `{"mode": "raw"|"quota", "image_input": "image"|"image+title", "k": 12, "weak_threshold": <float>}`.
- [ ] `weak_threshold` is set from the data. It is the 5th percentile of the top-1 similarity over the 15 queries, minus 0.02. The value and how it was derived are in the report.

**Verification:**
```bash
uv run python -m gallery.eval_gap && test -f config/retrieval.json && uv run pytest -q tests/test_retrieval_config.py
```
**Priority:** P0

---

### US-012: 3D layout
**As a** presenter **I want** a stable 3D layout of the corpus **so that** the cloud looks the same every time.

**Acceptance Criteria:**
- [ ] `uv run python -m gallery.layout` runs UMAP (`n_components=3, metric="cosine", random_state=42`) and writes `data/layout.npy` `(N, 3)`, scaled so that each axis lies in [-1, 1].
- [ ] Two runs produce identical output (byte-equal).
- [ ] As a neighborhood check, for at least 60% of items, at least 3 of their 10 nearest 768-d neighbors are also among their 30 nearest 3D neighbors. The figure is recorded in `reports/layout.md`.

**Verification:**
```bash
uv run pytest -q tests/test_layout.py
```
**Priority:** P0

---

### US-013: Front-end assets
**As a** developer **I want** one compact asset bundle **so that** the page loads the cloud fast.

**Acceptance Criteria:**
- [ ] `uv run python -m gallery.build_assets` writes `web/data/points.bin` (float32 xyz + uint8 rgb per point), `web/data/items.json` (id, source, modality, title, summary, thumb, source_url, in the same order), and copies or symlinks thumbnails to `web/thumbs/`.
- [ ] `items.json` is at most 6 MB and `points.bin` is exactly N × 15 bytes.
- [ ] A test checks that the order matches `vector_ids.json`.

**Verification:**
```bash
uv run pytest -q tests/test_assets.py
```
**Priority:** P0

---

### US-014: Search API
**As a** presenter **I want** a `/search` endpoint **so that** the UI can place a query and get its neighbors.

**Acceptance Criteria:**
- [ ] `POST /search {"query": str}` returns `{"query": str, "position": [x,y,z], "neighbors": [{"index": int, "id": str, "similarity": float, "rank": int}], "weak": bool, "mode": str}` with exactly `k` neighbors sorted by similarity descending, ranks 1..k.
- [ ] `position` is the similarity-weighted centroid of the neighbors' layout coordinates. Weights are `max(sim, 0)`; if all weights are 0, it is the plain mean.
- [ ] If `mode = quota`, the endpoint returns the top 4 per source, merged and sorted by similarity.
- [ ] `weak = top1_similarity < weak_threshold`.
- [ ] A query that is empty or only whitespace, or longer than 500 characters, returns 422. Model errors return 500 with a JSON `detail`. Neither case is swallowed.
- [ ] The embedder loads once at startup. Server-side p95 latency is ≤ 300 ms over 50 queries (measured with the model warm), recorded in `reports/latency.md`.

**Verification:**
```bash
uv run pytest -q tests/test_search_api.py && uv run pytest -q -m gpu tests/test_search_latency.py
```
**Priority:** P0

---

### US-015: Cloud render
**As an** audience member **I want to** see the whole corpus as a slowly orbiting 3D cloud of colored dots **so that** the space feels real.

**Acceptance Criteria:**
- [ ] `web/index.html` renders `points.bin` with three.js (pinned version, loaded from a CDN or vendored) on an off-white background (`#f6f4ef` or another value defined as a CSS token).
- [ ] `window.__gallery.pointCount` equals N.
- [ ] When idle, the camera auto-orbits. Dragging orbits manually, the wheel zooms, and auto-orbit resumes 5 s after the last interaction.
- [ ] `?bench=1` shows an fps overlay and sets `window.__gallery.fps` (rolling 2 s median).
- [ ] `scripts/bench_fps.py` opens the page in headed Chromium. If the WebGL renderer string contains "SwiftShader" or "llvmpipe", it exits with code 2 (`UNVERIFIED: no GPU`). Otherwise it asserts a median fps ≥ 55 over 10 s of idle orbit.

**Verification:**
```bash
uv run pytest -q tests/e2e/test_cloud.py && uv run python scripts/bench_fps.py
```
**Priority:** P0

---

### US-016: Chat box
**As a** presenter **I want** a single input in front of the cloud **so that** I can type a query and press Enter.

**Acceptance Criteria:**
- [ ] The input is centered near the bottom, has focus on load, and Enter submits. The input is disabled while a search is in flight.
- [ ] On a 4xx or 5xx response, an inline error message appears under the input with the server's `detail`, and the cloud state does not change.
- [ ] Empty input does not send a request.

**Verification:**
```bash
uv run pytest -q tests/e2e/test_chatbox.py
```
**Priority:** P0

---

### US-017: Query point and fly-in
**As an** audience member **I want to** see the query appear as a point and the camera fly to it **so that** I see where my words land.

**Acceptance Criteria:**
- [ ] After a successful search, a distinct query marker (larger and dark) appears at `position`.
- [ ] The camera tweens over 1.0–1.4 s (eased) to a framing that contains the query and all k neighbors in view (checked by projecting them to screen space at the end of the tween).
- [ ] `window.__gallery.phase` moves through `idle → flying → lines → wall`.
- [ ] Screenshots after the fly-in for the 5 demo queries are saved to `reports/screens/flyin_*.png`.

**Verification:**
```bash
uv run pytest -q tests/e2e/test_flyin.py
```
**Priority:** P0

---

### US-018: Neighbor lines and thumbnails
**As an** audience member **I want to** see lines spread from the query to each neighbor, each neighbor turning into its picture **so that** the nearest-neighbor idea is visible.

**Acceptance Criteria:**
- [ ] After the fly-in, k lines grow from the query marker to each neighbor in rank order. Each line starts 60–90 ms after the previous one and takes 250–400 ms to grow.
- [ ] Line opacity is mapped linearly from similarity to the range [0.35, 1.0] across the k neighbors.
- [ ] When a line reaches its neighbor, that point is replaced by a billboard of its thumbnail. Wiki items get a small card glyph in the neutral tone. Non-neighbor points dim to ≤ 25% opacity.
- [ ] The whole lines phase finishes in ≤ 1.4 s for k = 12.
- [ ] Screenshots are saved to `reports/screens/lines_*.png`.

**Verification:**
```bash
uv run pytest -q tests/e2e/test_lines.py
```
**Priority:** P0

---

### REVIEW GATE (after US-018)
Ralph stops here and asks Mason to look at `reports/screens/` and the live page before
starting US-019. This is the "mockup with real data first" checkpoint from SPEC §5.

---

### US-019: Gallery wall panel
**As an** audience member **I want** the neighbors hung as a gallery wall **so that** I can look at what was found.

**Acceptance Criteria:**
- [ ] A right-side panel (about 40% of the viewport width at ≥ 1280 px) slides in after the lines phase. It shows the k items in rank order on a light background: image items as framed thumbnails with the title below, wiki items as text cards showing the title and summary.
- [ ] Hovering an image piece shows its summary in an overlay within 150 ms.
- [ ] Hovering any piece pulses its point and line in the cloud (`window.__gallery.pulsedIndex` equals that item's index while hovered).
- [ ] All k pieces are in the DOM with `data-rank` 1..k.

**Verification:**
```bash
uv run pytest -q tests/e2e/test_wall.py
```
**Priority:** P0

---

### US-020: Open as page
**As a** presenter **I want to** expand the wall into a full page **so that** I can show it at full size.

**Acceptance Criteria:**
- [ ] An "Open as page" button turns the panel into a full-viewport gallery page with larger pieces, in the same light style.
- [ ] The page footer lists each piece's source and links to its `source_url`.
- [ ] A "Back to cloud" button and the Esc key both return to the panel view with the cloud state intact.

**Verification:**
```bash
uv run pytest -q tests/e2e/test_page.py
```
**Priority:** P1

---

### US-021: Weak query handling
**As a** presenter **I want** off-topic queries to be shown honestly **so that** the demo doesn't pretend.

**Acceptance Criteria:**
- [ ] When `weak = true`, the fly-in and lines still run, a "Nothing close" badge appears near the input, and the wall header reads "Weak matches".
- [ ] A test with a stubbed `/search` response with `weak: true` checks both labels. A real-model test checks that "quarterly tax filing deadlines" returns `weak: true`, or it records the top-1 similarity in `reports/modality_gap.md` if it does not.

**Verification:**
```bash
uv run pytest -q tests/e2e/test_weak.py
```
**Priority:** P1

---

### US-022: Consecutive queries
**As a** presenter **I want** a new query to move smoothly from the current view **so that** I can run several queries in a row.

**Acceptance Criteria:**
- [ ] Submitting a new query while the wall is shown clears the previous lines, thumbnails, dimming and wall before the new fly-in starts, and the camera tweens from its current pose.
- [ ] Submitting during an animation cancels it cleanly: after the second query finishes, the scene has exactly k lines and k thumbnail billboards.

**Verification:**
```bash
uv run pytest -q tests/e2e/test_consecutive.py
```
**Priority:** P1

---

### US-023: Demo set and acceptance numbers
**As a** presenter **I want** 5 scripted queries and measured numbers **so that** I know the demo meets the bar before I present.

**Acceptance Criteria:**
- [ ] `eval/demo_queries.yaml` contains 5 queries. They are a subset of `eval/queries.yaml`.
- [ ] `scripts/demo_check.py` runs each one in the real UI (Playwright, warm server) and records: time from Enter to `phase == "wall"` (must be ≤ 3.0 s), and the source mix of the wall (≥ 4 of 5 walls must contain ≥ 2 sources).
- [ ] The results are written to `reports/demo_check.md` with the raw numbers. The script exits non-zero if either threshold fails.

**Verification:**
```bash
uv run python scripts/demo_check.py
```
**Priority:** P0

## Functional Requirements
SPEC F1–F22 map to stories as follows: F1–F5 → US-004..008; F6–F7 → US-003, US-009;
F8–F10 → US-002, US-010, US-011, US-012; F11–F15 → US-014, US-021; F16–F22 → US-015..020, US-022.

## Non-Functional Requirements
- **Latency:** Enter → wall visible ≤ 3.0 s (US-023), and server-side search p95 ≤ 300 ms (US-014).
- **Frame rate:** idle orbit median ≥ 55 fps in GPU-backed Chromium on this machine (US-015).
- **Mixed walls:** ≥ 4 of 5 demo walls contain ≥ 2 sources (US-023), and ≥ 12 of 15 eval queries are mixed under the chosen policy (US-011).
- **Offline:** after the build, no runtime network calls. three.js is vendored, or cached and pinned.
- **Errors:** visible in the UI and the logs. Pipeline failures are logged per item.

## Success Metrics
The demo_check report passes, bench_fps passes (or is verified manually by Mason if it can
only run without a GPU), and Mason signs off at the review gate.

## Scope
### In Scope
Everything in SPEC §2–§4.
### Out of Scope
SPEC §7: live Gemma generation at query time, chat memory, image/audio/video queries, audio/video items, multiple users, hosting, a one-command launcher, similarity and license shown on hover.

## Risks & Mitigations
- **Modality gap:** handled by the policy in US-011.
- **EmbeddingGemma 2 is gated or not in Ollama:** HF path with Mason's token. Ralph stops and asks if credentials are missing.
- **Curation takes a long time (hours):** it can be resumed. Run it in the background and continue on the front end with a placeholder summary marked `PENDING` in dev only. `check_corpus --summaries` blocks the final acceptance.
- **Headless WebGL has no GPU:** bench_fps reports UNVERIFIED instead of a false pass.
- **The API corpus changes over time:** raw API responses are cached in `data/raw/`, so re-runs are deterministic.

## Open Questions
- The exact neutral tone for wiki items (decided in US-015 against the real cloud).
- Whether instanced thumbnail billboards need a texture atlas (decided in US-018 if fps drops).
