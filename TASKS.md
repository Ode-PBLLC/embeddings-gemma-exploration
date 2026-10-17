# Embeddings × Gemma demo — tasks

Goal: an internal demo interface. A cloud of embedded items with a chat box in front.
A prompt is embedded and dropped into the cloud, the camera zooms to it, its nearest
neighbors light up, and Gemma writes a page or report from those neighbors.

Everything runs locally on an RTX 3070 Ti (8 GB).

Stack (proposed): Python + FastAPI backend, a single-page front end using deck.gl,
EmbeddingGemma 2 for embeddings, Gemma 4 E4B (Ollama) for the report.

## 0. Models run locally
- [ ] 0.1 Embed one text and one image with EmbeddingGemma 2 (Ollama if it supports the vision encoder, otherwise HF transformers). Check that cross-modal similarity makes sense.
- [ ] 0.2 Pull `gemma4` (E4B) in Ollama. Test one prompt that includes images.
- [ ] 0.3 Load both models together. Record VRAM use and the latency of embedding one query.

## 1. Corpus (just enough to make the cloud beautiful)
- [ ] 1.1 Pull about 3–5k CC0 items from the Art Institute of Chicago API (image, title, description, dominant color) and a NASA Image Library slice for range of topics.
- [ ] 1.2 Normalize everything to one table: `id, modality, title, text, media_path, source, license, color_hex, topic`.
- [ ] 1.3 Check the mix of topics and colors, and that thumbnails load.

## 2. Embed + layout (offline, run once)
- [ ] 2.1 Batch-embed every item and store the vectors (Parquet or LanceDB).
- [ ] 2.2 Run UMAP to 2D and save the coordinates. Color each point with its real `color_hex`.
- [ ] 2.3 Build kNN over the full 768-d vectors, not over the 2D layout.
- [ ] 2.4 Sanity-check about 15 hand-written queries by looking at their neighbors.

## 3. Interface (the core of the demo)
- [ ] 3.0 Look at Embedding Atlas, latent-scope and Nomic Atlas for about 15 minutes. Reuse one if it fits; otherwise build directly with deck.gl.
- [ ] 3.1 Render the full cloud with zoom and pan, and show the item on hover.
- [ ] 3.2 Add a chat box in front of the cloud. It calls `POST /search`, which returns the query embedding's top-k neighbors.
- [ ] 3.3 Place the query in the cloud at the similarity-weighted centroid of its neighbors' 2D positions.
- [ ] 3.4 Animate a camera fly-in to the query point. Show the neighbors as thumbnails and dim everything else.
- [ ] 3.5 Panel or page view: stream the report beside the cloud, or open it as its own page.

## 4. Report from neighbors
- [ ] 4.1 `POST /report`: send the neighbors' text and thumbnails to Gemma 4 and stream back a structured page (title, theme, short narrative, sections).
- [ ] 4.2 Every claim in the report cites a neighbor item, and clicking a citation highlights that item in the cloud.
- [ ] 4.3 Style the page with a palette taken from the neighbors' colors.

## 5. Demo-ready
- [ ] 5.1 Write 3–5 scripted queries that show off the cloud.
- [ ] 5.2 One command starts the whole thing (`uv run` plus a static front end).

## Stretch
- [ ] Audio (Freesound CC0, NASA audio) and video, all in the same space.
- [ ] Compare two queries: two fly-ins, and a report on how the two neighborhoods overlap.
