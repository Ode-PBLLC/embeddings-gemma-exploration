#!/usr/bin/env bash
# Start the labelling page on :8010 (override with PORT): http://localhost:8010/label/
# Real data (data/, web/); EmbeddingGemma loads in the background and is needed only for "split by words".
# Labels are saved to data/labels/ (labels.json, label_rows.csv).
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run uvicorn gallery.label_app:create_app --factory --host 0.0.0.0 --port "${PORT:-8010}"
