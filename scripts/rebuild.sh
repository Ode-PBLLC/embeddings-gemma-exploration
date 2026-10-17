#!/usr/bin/env bash
# Rebuild both datasets from their original sources, in pipeline order. Every stage is resumable
# (downloads and Gemma summaries are cached), so a failed run can simply be started again.
#
#   scripts/rebuild.sh            the main gallery, then the Moon set
#   scripts/rebuild.sh main|moon  just one of them
#
# Needs: a CUDA GPU with bf16 (EmbeddingGemma 2, Whisper), ffmpeg, network access, and a Gemma 4
# llama.cpp server for the offline summaries (never Ollama), reachable at GEMMA_BASE_URL
# (default http://localhost:8081/v1). For example:
#   huggingface-cli download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_0.gguf mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models/gemma
#   llama-server -m models/gemma/gemma-4-E4B-it-Q4_0.gguf --mmproj models/gemma/mmproj-gemma-4-E4B-it-Q8_0.gguf \
#     --alias gemma-4-E4B-it -ngl 99 -c 32768 -np 4 --jinja --port 8081
# Stop scripts/serve.sh first on a small GPU: the embedding stages need the memory it holds.
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH="${1:-all}"
export GEMMA_BASE_URL="${GEMMA_BASE_URL:-http://localhost:8081/v1}"

run() { echo; echo "==> $*"; "$@"; }

health="${GEMMA_BASE_URL%/v1}/health"
if ! curl -fsS -m 5 "$health" >/dev/null; then
  echo "No Gemma server at $health — start llama-server first (see the top of this script)." >&2
  exit 1
fi

if [[ "$WHICH" == all || "$WHICH" == main ]]; then
  for src in aic nasa wiki nasa_video nasa_audio aic_audio; do
    run uv run python -m "gallery.ingest.$src"
  done
  run uv run python -m gallery.merge
  run uv run python -m gallery.transcribe           # word-timed transcripts for the sounds (Whisper)
  run uv run python -m gallery.curate --workers 4   # Gemma 4 summaries
  run uv run python -m gallery.sound_captions       # short captions for wordless sounds
  run uv run python -m gallery.check_summaries
  run uv run python -m gallery.embed_corpus         # EmbeddingGemma 2
  run uv run python -m gallery.eval_gap --keep-policy   # re-measure the mix, recalibrate the weak threshold
  run uv run python -m gallery.layout
  run uv run python -m gallery.build_assets
fi

if [[ "$WHICH" == all || "$WHICH" == moon ]]; then
  run uv run python -m gallery.ingest.moon
  export GALLERY_DATA_DIR=data_moon GALLERY_RETRIEVAL=data_moon/retrieval.json
  [[ -f data_moon/retrieval.json ]] || echo '{"mode": "raw", "image_input": "image+title", "k": 12}' > data_moon/retrieval.json
  run uv run python -m gallery.curate --workers 4
  run uv run python -m gallery.embed_corpus --chunk 128
  run uv run python -m gallery.layout --data-dir data_moon --report reports/layout_moon.md --align-globe
  run uv run python -m gallery.calibrate_weak --data-dir data_moon --queries eval/moon_queries.yaml
  run uv run python -m gallery.build_assets --data-dir data_moon --out web_moon
fi
echo; echo "Rebuilt. Start it with: scripts/serve.sh"
