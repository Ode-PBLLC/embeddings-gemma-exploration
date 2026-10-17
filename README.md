# The Embedding Gallery

An internal Ode demo that makes embedding search visible. A few thousand works sit as a 3D cloud
of coloured points: paintings and objects from the Art Institute of Chicago, NASA images, video
clips and sounds, and Wikipedia paragraphs. You type a query, **EmbeddingGemma 2** embeds it, and
the camera flies to where the query lands. Lines run out to its 12 nearest neighbours, which are
then hung on a museum-style wall. **Gemma 4** read every work offline and wrote the label each one
carries.

A second, separate collection, **the Moon**, holds 1,024 tiles of a global lunar map named from the
IAU gazetteer. It opens as a globe built from the tiles themselves. One button breaks the globe
apart and flies every tile to its place in embedding space.

- Gallery: `http://localhost:8000/`. 5,376 works in the public data release: 3,400 images, 1,700 text passages, 240 video clips and 36 sounds.
- The Moon: `http://localhost:8000/moon/`. 1,024 tiles.

## Quick start

```bash
git clone git@github.com:Ode-PBLLC/embeddings-gemma-exploration.git
cd embeddings-gemma-exploration
scripts/setup.sh     # uv sync + the prebuilt data release (~460 MB) + EmbeddingGemma 2 weights
scripts/serve.sh     # http://localhost:8000/  and  http://localhost:8000/moon/
```

Setup needs [uv](https://docs.astral.sh/uv/), `curl` and `tar`. uv provides Python 3.12. Queries
are embedded live, so serving wants a CUDA GPU with bf16; the model takes about 1.5 GB of GPU
memory. It also runs on CPU, more slowly.

`scripts/setup.sh --rebuild` rebuilds everything from the original sources instead
(`scripts/rebuild.sh`). That needs a CUDA GPU, `ffmpeg`, network access and a Gemma 4 E4B
**llama.cpp** server, and it takes many hours. The script header gives the exact `llama-server`
command. Every stage is resumable.

## Using it

- **Ask:** type a query and press Enter. A query with nothing close is flagged *Nothing close*.
- **Explore the cloud:** drag to turn, right-drag to pan, scroll to zoom toward the pointer. Zoom
  in far enough and the neighbourhood flattens onto a pane of thumbnails. Click a dot or a
  picture to open its placard.
- **Scope:** the small menu under the search box restricts results by type (images, text, video,
  sound) or collection. These are exact metadata filters applied before the semantic ranking.
- **The Moon:** drag to turn the globe and click a tile to read about it. *Into embedding space*
  breaks the globe up; *Back to the globe* rebuilds it. A query starts from either view.

## How it works

| Stage | Module | Notes |
|---|---|---|
| Ingest | `gallery.ingest.{aic,nasa,wiki,nasa_video,nasa_audio,aic_audio,moon}` | Polite, cached API clients. One row per work in a shared schema (`gallery.schema`). |
| Merge | `gallery.merge` | Canonical row order for every later stage. |
| Transcribe | `gallery.transcribe` | faster-whisper word timings for the karaoke transcript on sound cards. |
| Curate | `gallery.curate`, `gallery.sound_captions` | Gemma 4 E4B on llama.cpp writes one label per work, from the image, video frames, transcript or passage. |
| Embed | `gallery.embed_corpus` | EmbeddingGemma 2 (768-d) via sentence-transformers. Images and video are embedded with their title interleaved; sounds from the audio itself; text in the model card's document format. |
| Calibrate | `gallery.eval_gap`, `gallery.calibrate_weak` | Measures the modality gap, chooses the retrieval policy and sets the *Nothing close* threshold. |
| Layout | `gallery.layout` | 3D UMAP (cosine, seeded). The Moon's layout is posed against its globe so the two spaces share one frame. |
| Bundle | `gallery.build_assets` | `items.json`, `points.bin` and thumbnails for the front end. |
| Serve | `gallery.app`, `gallery.search` | FastAPI with brute-force cosine search. The Moon is served at `/moon/` from the same process and shares the loaded model. |
| Front end | `web/` | three.js r170 (vendored), plain ES modules, no build step. |

The design and the decisions behind it are in `SPEC.md`, `PRD.md` and `BUILD.md`. Measurements
are in `reports/`, for example `reports/modality_gap.md`, `reports/layout.md`,
`reports/layout_moon.md` and `reports/weak_threshold_data_moon.md`. `SPEC.md` is the original
seed and still names Ollama for Gemma; the build uses llama.cpp instead (BUILD.md, decision 8).

## Tests

```bash
uv run pytest tests --ignore=tests/e2e     # unit and API tests
uv run playwright install chromium && uv run pytest tests/e2e   # browser tests (synthetic data, no GPU)
```

`test_scripts/` holds ad hoc checks against a running server, for example
`uv run python test_scripts/moon_ui_check.py`.

## Data sources and licences

| Collection | Source | Terms |
|---|---|---|
| Artworks | [Art Institute of Chicago API](https://api.artic.edu/docs/) | Images CC0 (public-domain works only); text CC BY 4.0 |
| AIC audio guides | AIC API `sounds` | © AIC, noncommercial educational use. **Not in the public data release**; `--rebuild` fetches them from AIC. |
| Space images, clips, sounds | [NASA Image and Video Library](https://images.nasa.gov/) | NASA media guidelines (generally not copyrighted) |
| Encyclopedia text | English Wikipedia | CC BY-SA 4.0 |
| The Moon | [OpenPlanetaryMap](https://www.openplanetary.org/) Moon Hillshaded Albedo (LOLA / USGS); names from the [IAU/USGS Gazetteer of Planetary Nomenclature](https://planetarynames.wr.usgs.gov/) | Tile licence not stated by OpenPlanetaryMap; attribution kept |
| Models | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), Gemma 4 E4B ([GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)) | Gemma terms of use |

Gemma's labels are machine-written and can be wrong. This is a demo, not a reference.
