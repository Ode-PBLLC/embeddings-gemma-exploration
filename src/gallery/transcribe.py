"""Word-timed transcripts for the sound works (faster-whisper), written into each sound's sidecar JSON.

uv run python -m gallery.transcribe [--model small.en]

Adds to data/media/nasa_audio/<id>.json:
  "speech": true/false   - whether the sound carries enough clear speech to follow along
  "words": [[word, start_s, end_s], ...]   (only when speech is true)
A sound is "wordless" when Whisper hears fewer than MIN_WORDS confident words: those cards show a
short static caption instead of a transcript (gallery.sound_captions).
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from gallery.ingest.nasa_audio import audio_paths
from gallery.schema import DATA

log = logging.getLogger("gallery.transcribe")
MIN_WORDS = 12         # fewer confident words than this: treat as wordless (wind, roar, rumble)
MIN_PROB = 0.5         # a word counts as confident at this probability or more


def preload_cuda12() -> None:
    """faster-whisper's CTranslate2 links CUDA 12 cuBLAS/cuDNN; torch here ships CUDA 13. Load the
    CUDA 12 libraries from their pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12) before the model."""
    import ctypes
    import glob
    import os

    import nvidia

    base = os.path.dirname(nvidia.__path__[0] + "/")
    for pattern in ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn*.so.9"):
        for lib in sorted(glob.glob(os.path.join(base, pattern))):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


def transcribe(model, wav) -> tuple[bool, list]:
    import soundfile as sf

    # decode ourselves: faster-whisper's own loader (PyAV) is incompatible with the installed av
    audio, sr = sf.read(str(wav), dtype="float32")
    if sr != 16000:
        raise ValueError(f"{wav}: expected 16 kHz, got {sr}")
    segments, _info = model.transcribe(audio, language="en", word_timestamps=True, vad_filter=True,
                                       condition_on_previous_text=False)
    words = [[w.word.strip(), round(w.start, 2), round(w.end, 2), w.probability]
             for seg in segments for w in (seg.words or []) if w.word.strip()]
    confident = [w for w in words if w[3] >= MIN_PROB]
    speech = len(confident) >= MIN_WORDS
    return speech, [w[:3] for w in words] if speech else []


def main(argv: list[str] | None = None) -> int:
    from faster_whisper import WhisperModel

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    items = pd.read_parquet(DATA / "items.parquet")
    sounds = items[items["modality"] == "audio"]
    if args.device == "cuda":
        preload_cuda12()
    model = WhisperModel(args.model, device=args.device, compute_type="int8_float16" if args.device == "cuda" else "int8")
    n_speech = 0
    for r in sounds.itertuples(index=False):
        wav, side = audio_paths(r.id)
        meta = json.loads(side.read_text())
        speech, words = transcribe(model, wav)
        meta["speech"], meta["words"] = speech, words
        side.write_text(json.dumps(meta))
        n_speech += speech
        log.info("%-60s %s (%d words)", r.title[:60], "speech" if speech else "wordless", len(words))
    log.info("done: %d sounds, %d with speech, %d wordless", len(sounds), n_speech, len(sounds) - n_speech)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
