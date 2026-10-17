#!/usr/bin/env bash
# One-step setup for the Embedding Gallery.
#
#   scripts/setup.sh              install, download the prebuilt data release (~460 MB) and the model
#   scripts/setup.sh --rebuild    install, then rebuild every dataset from its original sources
#                                 (scripts/rebuild.sh: many hours, a CUDA GPU, a Gemma 4 llama.cpp server)
#   scripts/setup.sh --no-data    install only
#
# Then: scripts/serve.sh  ->  http://localhost:8000/ (the gallery) and http://localhost:8000/moon/ (the Moon)
#
# Needs: uv (https://docs.astral.sh/uv/), curl, tar; ffmpeg too for --rebuild. Python 3.12 comes via uv.
# A CUDA GPU with bf16 is strongly recommended for serving (queries are embedded live); CPU works, slowly.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="Ode-PBLLC/embeddings-gemma-exploration"
DATA_TAG="${GALLERY_DATA_TAG:-data-v1}"
ASSET="gallery-data.tar.gz"
MODE="${1:-download}"
case "$MODE" in
  download|--rebuild|--no-data) ;;
  -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
  *) echo "unknown option: $MODE (try --help)" >&2; exit 2 ;;
esac

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1 — $2" >&2; exit 1; }; }
need uv "install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
need curl "install curl"
need tar "install tar"

echo "==> Python environment (uv sync)"
uv sync

if [[ "$MODE" == "download" ]]; then
  if [[ -f web/data/items.json && -f data/vectors.npy ]]; then
    echo "==> data already present (web/data, data/); delete them to download again"
  else
    base="https://github.com/$REPO/releases/download/$DATA_TAG"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    echo "==> downloading $ASSET ($DATA_TAG)"
    curl -fL --progress-bar -o "$tmp/$ASSET" "$base/$ASSET"
    curl -fsSL -o "$tmp/$ASSET.sha256" "$base/$ASSET.sha256"
    (cd "$tmp" && sha256sum -c "$ASSET.sha256")
    echo "==> unpacking into $(pwd)"
    tar -xzf "$tmp/$ASSET" -C .
  fi
fi

if [[ "$MODE" != "--no-data" ]]; then
  echo "==> EmbeddingGemma 2 weights (Hugging Face cache; ~1.5 GB the first time)"
  uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/embeddinggemma-2'))"
fi

if [[ "$MODE" == "--rebuild" ]]; then
  need ffmpeg "install ffmpeg (the video and sound ingesters use it)"
  scripts/rebuild.sh
fi

echo
echo "Done. Start it with:  scripts/serve.sh"
echo "  gallery: http://localhost:8000/      the Moon: http://localhost:8000/moon/"
