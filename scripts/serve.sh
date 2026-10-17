#!/usr/bin/env bash
# Start the gallery on :8000 (override with PORT).
#   scripts/serve.sh               real data (data/, config/retrieval.json, web/), EmbeddingGemma on the GPU,
#                                  plus the Moon set at /moon/ when web_moon/ and data_moon/ exist
#   scripts/serve.sh --synthetic   synthetic data (data/synthetic), keyword fake embedder, no GPU
# Build the synthetic set first with: uv run python scripts/make_synthetic.py
#   then: GALLERY_DATA_DIR=data/synthetic GALLERY_ASSETS_DIR=data/synthetic/web uv run python -m gallery.build_assets
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--synthetic" ]]; then
  export GALLERY_DATA_DIR=data/synthetic
  export GALLERY_ASSETS_DIR=data/synthetic/web
  export GALLERY_RETRIEVAL=data/synthetic/retrieval.json
  export GALLERY_EMBEDDER=fake
fi
# the Moon set is served at /moon/ whenever it has been built or downloaded (GALLERY_MOUNTS="" turns it off)
if [[ -z "${GALLERY_MOUNTS+set}" && -f web_moon/data/items.json && -f data_moon/vectors.npy ]]; then
  export GALLERY_MOUNTS=moon
fi
exec uv run uvicorn gallery.app:create_app --factory --host 0.0.0.0 --port "${PORT:-8000}"
