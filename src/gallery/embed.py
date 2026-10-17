"""EmbeddingGemma 2 wrapper: one model on the GPU, text and image embeddings in one 768-d space.

Model card facts this follows (google/embeddinggemma-2, README):
- prompts: query -> "task: search result | query: ", document -> "title: none | text: ".
  Titled documents are formatted by hand as "title: {title} | text: {content}".
- images are passed without any prefix, as {"image": path}.
- interleaved inputs use the "<|image|>" placeholder inside "text", filled from "image".
- video is {"video": <path | list of PIL frames>}; "<|video|>" interleaves it with text.
- run in bfloat16 or float32, never float16 (NaN / degraded embeddings).
- the audio encoder is optional; we skip it with config_kwargs={"audio_config": None} unless with_audio.
- audio is {"audio": <path>} (mono 16 kHz, ~25 tokens/s, up to ~327 s); "<|audio|>" interleaves it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MODEL_ID = "google/embeddinggemma-2"
DIM = 768
QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT_NO_TITLE = "title: none | text: "

log = logging.getLogger(__name__)


def doc_text(content: str, title: str | None = None) -> str:
    """Document-side text in the model card's retrieval format."""
    t = (title or "").strip() or "none"
    return f"title: {t} | text: {content}"


def interleaved_image_input(path: str | Path, title: str) -> dict:
    """One input that holds the image and its title together (model-card interleaving)."""
    return {"text": f"title: {title} | <|image|>", "image": str(path)}


class Embedder:
    """Loads EmbeddingGemma 2 once. All outputs are float32 (n, 768), L2-normalized."""

    def __init__(self, device: str | None = None, dtype: str = "bfloat16",
                 text_batch: int = 32, image_batch: int = 8, with_audio: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 not supported on this GPU; refusing float16 (model card: NaNs)")
        torch_dtype = getattr(torch, dtype)
        if torch_dtype == torch.float16:
            raise ValueError("float16 is unsafe for EmbeddingGemma 2 (model card); use bfloat16/float32")
        self.dtype = dtype
        self.text_batch = text_batch
        self.image_batch = image_batch
        self.model = SentenceTransformer(
            MODEL_ID,
            device=self.device,
            model_kwargs={"torch_dtype": torch_dtype},
            **({} if with_audio else {"config_kwargs": {"audio_config": None}}),
        )
        self.with_audio = with_audio
        self.model.eval()

    def _encode(self, inputs: list, batch_size: int, prompt: str | None = None) -> np.ndarray:
        if not inputs:
            return np.zeros((0, DIM), dtype=np.float32)
        kw = {"prompt": prompt} if prompt is not None else {}
        out = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kw,
        )
        out = np.asarray(out, dtype=np.float32)
        if out.shape != (len(inputs), DIM):
            raise RuntimeError(f"unexpected embedding shape {out.shape}, wanted {(len(inputs), DIM)}")
        if not np.isfinite(out).all():
            raise RuntimeError("non-finite values in embeddings")
        # renormalize in float32 (bf16 compute leaves norms a hair off 1)
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out

    def embed_texts(self, texts: Sequence[str], kind: str = "document",
                    titles: Sequence[str | None] | None = None) -> np.ndarray:
        """kind="query": search-query prompt. kind="document": "title: {title|none} | text: ..."."""
        texts = list(texts)
        if kind == "query":
            return self._encode(texts, self.text_batch, prompt=QUERY_PROMPT)
        if kind != "document":
            raise ValueError(f"kind must be 'query' or 'document', got {kind!r}")
        if titles is None:
            return self._encode(texts, self.text_batch, prompt=DOC_PROMPT_NO_TITLE)
        if len(titles) != len(texts):
            raise ValueError("titles and texts differ in length")
        return self._encode([doc_text(c, t) for c, t in zip(texts, titles)], self.text_batch)

    def embed_images(self, paths: Sequence[str | Path],
                     titles: Sequence[str] | None = None) -> np.ndarray:
        """Images alone ({"image": path}, no prefix), or interleaved image+title if titles given."""
        paths = [str(p) for p in paths]
        for p in paths:
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        if titles is None:
            inputs = [{"image": p} for p in paths]
        else:
            if len(titles) != len(paths):
                raise ValueError("titles and paths differ in length")
            inputs = [interleaved_image_input(p, t) for p, t in zip(paths, titles)]
        return self._encode(inputs, self.image_batch)

    def embed_audio(self, wavs: Sequence[str | Path], titles: Sequence[str] | None = None) -> np.ndarray:
        """One embedding per sound (16 kHz mono WAV), alone or interleaved with its title."""
        if not self.with_audio:
            raise RuntimeError("Embedder was loaded without the audio encoder (with_audio=False)")
        wavs = [str(w) for w in wavs]
        for w in wavs:
            if not Path(w).is_file():
                raise FileNotFoundError(w)
        if titles is None:
            inputs = [{"audio": w} for w in wavs]
        else:
            inputs = [{"text": f"title: {t} | <|audio|>", "audio": w} for t, w in zip(titles, wavs)]
        return self._encode(inputs, 1)

    def embed_videos(self, frame_dirs: Sequence[str | Path],
                     titles: Sequence[str] | None = None) -> np.ndarray:
        """One embedding per clip from its sampled frames (fNN.jpg, in order), alone or with its title.
        One clip at a time: a clip is up to 16 frames through the vision encoder."""
        from PIL import Image

        inputs = []
        for k, d in enumerate(frame_dirs):
            files = sorted(Path(d).glob("f*.jpg"))
            if not files:
                raise FileNotFoundError(f"no frames in {d}")
            frames = [Image.open(f).convert("RGB") for f in files]
            if titles is None:
                inputs.append({"video": frames})
            else:
                inputs.append({"text": f"title: {titles[k]} | <|video|>", "video": frames})
        return self._encode(inputs, 1)
