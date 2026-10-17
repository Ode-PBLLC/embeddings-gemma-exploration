# SPEC — Embedding Gallery (EmbeddingGemma 2 × Gemma 4 demo)

Status: seeded 2026-10-15 from an interview with Mason. Supersedes the rough list in `TASKS.md`.

## 1. Overview

An internal demo interface that makes embedding search visible. About 5,000 items (art,
space imagery and encyclopedia text) float as a 3D cloud of colored points. The presenter
types a text query. EmbeddingGemma 2 embeds the query, the query point appears in the
cloud, and the camera flies in to it. Lines then spread out from the query to its 12
nearest neighbors, and those neighbors become thumbnails. A side panel hangs them as a
**gallery wall**, which can be expanded to a full page. Gemma 4 has already "read"
every item offline and written a one-paragraph summary for each, and that summary is
what appears on hover.

The story for the audience: **"Gemma read everything; EmbeddingGemma finds it."**

- Audience: Ode colleagues, shown by Mason as a presentation. It is not self-serve.
- Machine: Mason's WSL2 box (RTX 3070 Ti, 8 GB VRAM), opened in a Windows browser at `localhost`.

## 2. Functional requirements

### 2.1 Corpus (offline)
- F1. About 5,000 items, split in even thirds:
  - **AIC**: about 1,700 CC0 artworks from the Art Institute of Chicago API (image, title, artist, date, description, API color field).
  - **NASA**: about 1,700 images from the NASA Image and Video Library (image, title, description, date).
  - **Wiki**: about 1,700 paragraphs from English Wikipedia articles targeted at the artists, movements, missions and objects that appear in the AIC and NASA items. Each article is chunked into paragraphs, and paragraphs shorter than about 40 words are dropped.
- F2. Every item is stored in one table: `id, source (aic|nasa|wiki), modality (image|text), title, raw_text, summary, media_path, thumb_path, source_url, license, color_hex, x, y, z`.
- F3. `color_hex` gives each item's color in the cloud. AIC uses its API color. NASA images use a dominant color computed from the image (k-means or similar). Wiki items get a single neutral tone.
- F4. Topics should be mixed: cover a range of AIC departments and NASA subject areas, not just the most popular items. The ingestion scripts log the source and topic mix.
- F5. Thumbnails are stored locally (about 256–400 px on the long edge). The demo makes no network calls at runtime.

### 2.2 Offline Gemma curation
- F6. Gemma 4 E4B (Ollama, `gemma4:e4b`) writes a one-paragraph `summary` for every item:
  - image items: it is given the image and the metadata, and describes what is depicted and why it is notable;
  - wiki items: it condenses the paragraph to about 60–90 words.
- F7. The curation job can be resumed and does not redo items. It writes progress to disk, retries each item a bounded number of times, and logs failures as failures rather than skipping them. An item with a failed summary is excluded from the cloud and listed in a failures report.

### 2.3 Embeddings and layout (offline)
- F8. Every item is embedded with **EmbeddingGemma 2** (768-d). Image items are embedded from the image itself; the Phase 0 modality-gap check (§6) decides whether the title is interleaved with it. Wiki items are embedded from `raw_text`.
- F9. The vectors are L2-normalized and stored on disk. kNN is brute-force cosine over all items in numpy, which is trivial at 5k.
- F10. The layout is a 3D UMAP of the 768-d vectors (cosine metric), computed once, with a fixed seed, and stored.

### 2.4 Query (runtime)
- F11. `POST /search {query}` returns: the query's 3D position, the top 12 neighbors (id, similarity, rank), and a `weak` flag.
- F12. **Query placement:** the query point is the similarity-weighted centroid of its 12 neighbors' 3D positions. It is not placed with `umap.transform`.
- F13. **Weak queries:** if the best similarity is below a threshold set in Phase 0 from the observed distribution, `weak=true`. The UI still flies in, shows a "nothing close" badge, and labels the wall's matches as weak.
- F14. Queries are single-shot and text only. There is no conversation memory.
- F15. Only EmbeddingGemma runs at query time. Gemma 4 does not run at runtime.

### 2.5 Interface
- F16. **Cloud:** about 5k points in 3D on an off-white background. Each point is a dot in its `color_hex`. The camera orbits slowly when idle and can be dragged and zoomed.
- F17. **Chat box:** a single input in front of the cloud. Enter submits.
- F18. **Fly-in sequence** for each query:
  1. The query point appears in the cloud and the camera flies smoothly to frame it and its neighbors.
  2. **Neighbor lines:** thin lines grow outward from the query point to each of the 12 neighbors, staggered by rank (nearest first). Each line's opacity or weight scales with similarity.
  3. As each line reaches its point, the dot turns into the item's thumbnail (wiki items become a small text-card glyph). Other points dim.
  4. The gallery wall panel slides in.
- F19. **Gallery wall (panel):** the 12 neighbors hung in similarity order. Image items appear as framed images and wiki items as text cards showing the summary. Hovering a piece shows its summary paragraph and pulses its point and line in the cloud.
- F20. **Open as page:** a button expands the wall into a full-screen gallery page in the same light style, with a way back to the cloud. Each piece's `source_url` is kept for attribution (in the page footer or on click).
- F21. A new query flies from the current view to the new query point. The previous lines and thumbnails clear.
- F22. Visual style is light throughout: an off-white background, an editorial and calm feel, with color coming from the items.

## 3. Non-functional requirements
- N1. **Snappy:** from pressing Enter to the wall being visible takes ≤ 3 s, measured on this machine, with the model already loaded. The fly-in and line animation run about 1.5–2 s of that.
- N2. **Smooth:** the idle cloud orbits at about 60 fps in Chrome on this machine, and the fly-in doesn't drop frames visibly.
- N3. **Mixed walls:** most scripted queries (at least 4 of 5) return walls that contain at least two of the three sources.
- N4. Runs fully offline once the build is done.
- N5. Errors are not swallowed: backend errors show as a visible error state in the UI, and offline jobs fail loudly with item-level logs.

## 4. Architecture
- Python with `uv`. A FastAPI backend serves `/search` and the static front end, and keeps EmbeddingGemma 2 loaded on the GPU.
- Offline pipeline scripts run in order: `ingest_aic`, `ingest_nasa`, `ingest_wiki`, `curate` (Gemma 4 summaries), `embed`, `layout` (UMAP), then `build_assets` (a JSON or binary point buffer plus thumbnails for the front end).
- Front end: a single page using three.js for the 3D cloud (points, instanced thumbnail sprites, line geometry and camera tweening), plus plain HTML/CSS for the chat box, panel and page. No heavy framework.
- Models: EmbeddingGemma 2 through HF transformers or sentence-transformers. Ollama is only used if Phase 0 shows it supports the multimodal encoders. Gemma 4 E4B runs through Ollama.
- The two models never need to be on the GPU together: curation (Gemma 4) is offline, and the runtime only loads EmbeddingGemma.

## 5. Build phases
0. **Feasibility.** Load EmbeddingGemma 2 and embed text and images. Run `gemma4:e4b` with an image. Record VRAM and latency. Measure the modality gap (§6).
1. **Ingest.** Write the three ingestion scripts and the unified table, then report the corpus mix.
2. **Curate.** Gemma 4 writes summaries, with resume support and a failures report.
3. **Embed and layout.** Vectors, UMAP 3D, and sanity queries.
4. **Interface.** Cloud, chat box, fly-in, neighbor lines, thumbnails, wall panel, page view.
5. **Demo set.** 5 scripted queries, and checks of N1–N3 with measured numbers.

Note: the first look at the interface should be a mockup with real data (the cloud and a fly-in on real vectors) shown to Mason before it is polished.

## 6. Modality-gap decision (resolved policy)
Phase 0 and Phase 3 measure where text queries' neighbors come from (by source) over about 15 test queries.
1. If raw top-12 walls are mixed (N3 holds), use raw top-12.
2. Otherwise, re-embed image items as an interleaved image plus title, and measure again.
3. If walls are still not mixed, use per-source quotas (top 4 from each source) and note this in the UI's fine print.
The measured source mix is recorded for every step that is tried.

## 7. Out of scope
- Live Gemma generation at query time (titles, captions, query-aware notes).
- Conversational follow-ups and query history.
- Image, audio or video queries, and audio or video items.
- Multiple users, auth, hosting, packaging for other machines.
- A one-command launcher (nice to have, not required).
- Showing similarity scores and source/license on hover (the source is kept for attribution on the page view).

## 8. Assumptions and open questions
- A1. EmbeddingGemma 2 (released 2026-10-06, 740M parameters, 768-d, Apache 2.0) can be downloaded from Hugging Face. It may be gated, in which case Mason's HF token is needed.
- A2. `gemma4:e4b` is available in Ollama with image input (about 5 GB quantized), so it fits in 8 GB on its own.
- A3. Curating about 5k items with E4B takes on the order of hours. It runs unattended and can be resumed.
- A4. Wikipedia text is CC BY-SA, which is acceptable for internal use with source links kept.
- O1. The weak-query threshold is set from data in Phase 3.
- O2. Thumbnails versus the dots' texture budget: if instanced thumbnails for 12 neighbors plus 5k dots hurt fps, simplify the dots first.

## 9. Autonomy for the build
`/ralph` may do these without asking: download models (Ollama and HF), call the AIC, NASA and Wikipedia APIs with polite rate limits, download about 1–2 GB of thumbnails, run long GPU jobs, and install Python and JS dependencies. It must stop and ask before deleting data or before any step that needs credentials it doesn't have.
