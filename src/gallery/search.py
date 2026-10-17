"""Brute-force cosine search over the asset items (US-014).

The searchable set is exactly the items in the front-end bundle (web/data/items.json), so a
neighbour's `index` is its index in items.json and points.bin.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

MAX_QUERY_CHARS = 500
SOURCES = ("aic", "nasa", "wiki", "moon")
MODALITIES = ("image", "text", "video", "audio")
# Metadata a query may be restricted to. Deterministic pre-filters applied before the nearest-neighbour
# ranking: the semantic part of a query stays the embedding; these only decide what is eligible.
FILTER_FIELDS = {"modality": MODALITIES, "source": SOURCES}

EmbedFn = Callable[[str], np.ndarray]


class QueryError(ValueError):
    """The query itself is invalid (maps to 422)."""


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str
    k: int
    weak_threshold: float

    @classmethod
    def load(cls, path: Path) -> "RetrievalConfig":
        if not path.is_file():
            raise FileNotFoundError(f"retrieval config not found: {path} (written by gallery.eval_gap)")
        cfg = json.loads(path.read_text())
        mode = cfg.get("mode", "raw")
        if mode not in ("raw", "quota"):
            raise ValueError(f"{path}: mode must be raw|quota, got {mode!r}")
        if "weak_threshold" not in cfg:
            raise ValueError(f"{path}: missing weak_threshold")
        return cls(mode=mode, k=int(cfg.get("k", 12)), weak_threshold=float(cfg["weak_threshold"]))


def validate_filters(filters: dict | None) -> dict[str, list[str]]:
    """Normalise {field: [allowed values]}; unknown fields or values are errors, empty means no filter."""
    if not filters:
        return {}
    if not isinstance(filters, dict):
        raise QueryError("filters must be an object of {field: [values]}")
    out = {}
    for field, values in filters.items():
        if field not in FILTER_FIELDS:
            raise QueryError(f"unknown filter field {field!r}; allowed: {sorted(FILTER_FIELDS)}")
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise QueryError(f"filter {field!r} must be a list of strings")
        bad = sorted(set(values) - set(FILTER_FIELDS[field]))
        if bad:
            raise QueryError(f"unknown {field} value(s) {bad}; allowed: {list(FILTER_FIELDS[field])}")
        if values:
            out[field] = sorted(set(values), key=FILTER_FIELDS[field].index)
    return out


def validate_query(q: str) -> str:
    if not isinstance(q, str) or not q.strip():
        raise QueryError("query is empty")
    if len(q) > MAX_QUERY_CHARS:
        raise QueryError(f"query is longer than {MAX_QUERY_CHARS} characters")
    return q.strip()


class Searcher:
    def __init__(self, vectors: np.ndarray, layout: np.ndarray, sources: np.ndarray, ids: list[str],
                 config: RetrievalConfig, embed: EmbedFn, modalities: np.ndarray | None = None):
        if not (len(vectors) == len(layout) == len(sources) == len(ids)):
            raise ValueError(f"misaligned inputs: vectors {len(vectors)}, layout {len(layout)}, "
                             f"sources {len(sources)}, ids {len(ids)}")
        if len(vectors) < config.k:
            raise ValueError(f"only {len(vectors)} items, fewer than k={config.k}")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.layout = np.asarray(layout, dtype=np.float32)
        self.sources = np.asarray(sources)
        self.modalities = None if modalities is None else np.asarray(modalities)
        if self.modalities is not None and len(self.modalities) != len(ids):
            raise ValueError(f"misaligned modalities: {len(self.modalities)} vs {len(ids)} items")
        self.ids = list(ids)
        self.config = config
        self.embed = embed

    def eligible(self, filters: dict[str, list[str]]) -> np.ndarray:
        """Boolean mask of the items a filtered query may return."""
        mask = np.ones(len(self.ids), dtype=bool)
        if "source" in filters:
            mask &= np.isin(self.sources, filters["source"])
        if "modality" in filters:
            if self.modalities is None:
                raise RuntimeError("searcher was built without modality metadata")
            mask &= np.isin(self.modalities, filters["modality"])
        return mask

    def _pick(self, sims: np.ndarray, mask: np.ndarray) -> np.ndarray:
        k = self.config.k
        cand = np.flatnonzero(mask)
        if self.config.mode == "raw":
            kk = min(k, len(cand))
            top = cand[np.argpartition(-sims[cand], kk - 1)[:kk]]
        else:
            present = [s for s in SOURCES if (mask & (self.sources == s)).any()]
            per = k // len(present)
            picks = []
            for s in present:
                idx = np.flatnonzero(mask & (self.sources == s))
                picks.extend(idx[np.argsort(-sims[idx], kind="stable")[:per]].tolist())
            top = np.array(picks, dtype=np.int64)
        return top[np.argsort(-sims[top], kind="stable")]

    def search(self, query: str, filters: dict | None = None) -> dict:
        q = validate_query(query)
        flt = validate_filters(filters)
        mask = self.eligible(flt)
        n_eligible = int(mask.sum())
        if n_eligible == 0:
            raise QueryError("no works match these filters")
        qv = np.asarray(self.embed(q), dtype=np.float32).reshape(-1)
        if qv.shape[0] != self.vectors.shape[1] or not np.isfinite(qv).all():
            raise RuntimeError(f"embedder returned a bad vector: shape {qv.shape}")
        qv = qv / np.linalg.norm(qv)
        sims = self.vectors @ qv
        top = self._pick(sims, mask)
        s = sims[top]
        w = np.maximum(s, 0.0)
        pos = (w[:, None] * self.layout[top]).sum(0) / w.sum() if w.sum() > 0 else self.layout[top].mean(0)
        return {
            "query": q,
            "position": [float(x) for x in pos],
            "neighbors": [{"index": int(i), "id": self.ids[i], "similarity": float(v), "rank": r + 1}
                          for r, (i, v) in enumerate(zip(top, s))],
            "weak": bool(float(sims[mask].max()) < self.config.weak_threshold),
            "mode": self.config.mode,
            "filters": flt,
            "eligible": n_eligible,
        }


# ---------- embedders ----------

class KeywordEmbedder:
    """Deterministic stand-in for EmbeddingGemma on the synthetic dataset.

    Sums the centroids of every synthetic topic whose keywords occur in the query, plus a small
    hash-seeded noise vector. Queries that match no topic land near nothing (and come back weak).
    """

    def __init__(self, topics_npz: Path):
        z = np.load(topics_npz)
        self.centroids = z["centroids"].astype(np.float32)
        self.keywords = [set(k.split()) for k in z["keywords"].tolist()]

    def embed(self, text: str) -> np.ndarray:
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).normal(0, 0.5 / np.sqrt(self.centroids.shape[1]), self.centroids.shape[1])
        for c, kws in zip(self.centroids, self.keywords):
            if tokens & kws:
                v = v + c
        return (v / np.linalg.norm(v)).astype(np.float32)


def gemma_embed_fn() -> EmbedFn:
    """The real query embedder. Imported lazily so tests never load the model."""
    from gallery.embed import Embedder  # owned by the models workstream

    model = Embedder()

    def embed(text: str) -> np.ndarray:
        return np.asarray(model.embed_texts([text], kind="query"))[0]

    return embed
