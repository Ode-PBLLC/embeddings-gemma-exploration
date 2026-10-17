"""Gemma 4 E4B client for offline curation, via llama.cpp's OpenAI-compatible server.

The server is the `gallery-gemma4` docker container (see reports/phase0.md for the run command),
listening on http://localhost:8081/v1. Never Ollama (BUILD.md decision 8).
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import httpx

from gallery.schema import ROOT

BASE_URL = os.environ.get("GEMMA_BASE_URL", "http://localhost:8081/v1")
MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-E4B-it")
MIN_WORDS, MAX_WORDS = 40, 120

IMAGE_SYSTEM = (
    "You write wall labels for a museum. Write exactly one paragraph of 50 to 90 words in plain "
    "prose. Describe what is depicted and what makes the work notable. Use the metadata for "
    "names, dates and context, and what you see in the image for the description. Do not invent "
    "facts that are neither visible nor in the metadata. No preamble, no title line, no markdown, "
    "no lists, no quotation marks around the paragraph."
)
VIDEO_SYSTEM = (
    "You write wall labels for a museum. The images are frames sampled in order from one short "
    "video. Write exactly one paragraph of 50 to 90 words in plain prose: what the video shows and "
    "how it unfolds, and what makes it notable. Use the metadata (and any narration) for names, "
    "dates and context, and the frames for what is seen. Do not invent facts that are neither "
    "visible nor in the metadata. No preamble, no title line, no markdown, no lists."
)
AUDIO_SYSTEM = (
    "You write wall labels for a museum's listening station. The work is a sound recording. Write "
    "exactly one paragraph of 50 to 90 words in plain prose: what the listener hears and where the "
    "sound comes from, and what makes it notable. Use only the metadata and the transcript (if any); "
    "you cannot hear the recording, so do not describe sounds the metadata does not mention. No "
    "preamble, no title line, no markdown, no lists."
)
SOUND_CAPTION_SYSTEM = (
    "Write a short caption for a sound recording, the way a museum labels a listening station: a "
    "natural phrase of 3 to 6 words saying what is heard and where. Good: 'Wind on Mars', 'A Saturn V "
    "lifts off', 'Engines roar at Stennis', 'A meteoroid strikes Mars'. Bad: lists of keywords such as "
    "'Mars wind marsquake robotic arm'. Use only the metadata. Reply with the phrase only: no quotes, "
    "no full stop."
)
MOON_SYSTEM = (
    "You write captions for a lunar atlas. The image is one tile of a global map of the Moon (a "
    "relief hillshade blended with surface brightness). Write exactly one paragraph of 50 to 90 words "
    "in plain prose describing what this particular tile shows: the size and crowding of its craters, "
    "any smooth dark plains, rims, ridges, rilles or mountains, and how it differs from ordinary "
    "cratered ground. Describe only what is visible; if the tile is plain, say so briefly. Name "
    "features only from the metadata, and do not invent ages, missions or facts that are neither "
    "visible nor in the metadata. Do not mention the map, the hillshade or the tile itself. No "
    "preamble, no title line, no markdown, no lists."
)
VIDEO_FRAMES = 6   # of the 16 sampled, evenly spaced; enough to follow the clip, small enough to be quick
TEXT_SYSTEM = (
    "You condense encyclopedia passages. Rewrite the passage as exactly one paragraph of 50 to 90 "
    "words in plain prose. Stay faithful to the passage and add no facts that are not in it. "
    "No preamble, no title line, no markdown, no lists."
)


class SummaryError(RuntimeError):
    pass


def word_count(s: str) -> int:
    return len(s.split())


def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*(\*\*|#+\s*)", "", s)
    s = s.replace("**", "")
    s = re.sub(r"\s+", " ", s)
    if len(s) > 1 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _meta_block(item: dict) -> str:
    lines = [f"Title: {item.get('title', '')}"]
    if item.get("creator"):
        lines.append(f"Creator: {item['creator']}")
    if item.get("date"):
        lines.append(f"Date: {item['date']}")
    src = {"aic": "Art Institute of Chicago", "nasa": "NASA Image and Video Library",
           "moon": "OpenPlanetaryMap Moon basemap (LRO LOLA / USGS)"}.get(
        item.get("source", ""), item.get("source", ""))
    if src:
        lines.append(f"Collection: {src}")
    raw = (item.get("raw_text") or "").strip()
    if raw:
        lines.append(f"Catalogue description: {raw[:2500]}")
    return "\n".join(lines)


def _data_url(path: Path) -> str:
    b = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


def _transcript(item: dict) -> str:
    from gallery.ingest.nasa_audio import audio_paths

    import json as _json
    side = audio_paths(item["id"])[1]
    meta = _json.loads(side.read_text()) if side.is_file() else {}
    return " ".join(w[0] for w in meta.get("words") or [])


def build_messages(item: dict) -> list[dict]:
    if item.get("modality") == "audio":
        tr = _transcript(item)
        body = _meta_block(item) + (f"\nTranscript of the recording: {tr[:3000]}" if tr else
                                    "\nThe recording has no speech.")
        return [{"role": "system", "content": AUDIO_SYSTEM},
                {"role": "user", "content": body + "\n\nWrite the wall label now."}]
    if item.get("modality") == "video":
        from gallery.ingest.nasa_video import frames_dir

        files = sorted(frames_dir(item["id"]).glob("f*.jpg"))
        if not files:
            raise FileNotFoundError(f"frames for {item.get('id')}: {frames_dir(item['id'])}")
        pick = [files[round(k * (len(files) - 1) / (VIDEO_FRAMES - 1))] for k in range(VIDEO_FRAMES)]
        user = [{"type": "image_url", "image_url": {"url": _data_url(f)}} for f in pick]
        user.append({"type": "text", "text": _meta_block(item) + "\n\nWrite the wall label now."})
        return [{"role": "system", "content": VIDEO_SYSTEM}, {"role": "user", "content": user}]
    if item.get("modality") == "image":
        mp = item.get("media_path") or ""
        path = Path(mp) if Path(mp).is_absolute() else ROOT / mp
        if not path.is_file():
            raise FileNotFoundError(f"media for {item.get('id')}: {path}")
        user = [
            {"type": "image_url", "image_url": {"url": _data_url(path)}},
            {"type": "text", "text": _meta_block(item) + "\n\nWrite the wall label now."},
        ]
        system = MOON_SYSTEM if item.get("source") == "moon" else IMAGE_SYSTEM
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw = (item.get("raw_text") or "").strip()
    if not raw:
        raise ValueError(f"text item {item.get('id')} has empty raw_text")
    user = f"Article: {item.get('title', '')}\n\nPassage:\n{raw[:6000]}\n\nWrite the paragraph now."
    return [{"role": "system", "content": TEXT_SYSTEM}, {"role": "user", "content": user}]


class GemmaClient:
    def __init__(self, base_url: str = BASE_URL, model: str = MODEL, timeout: float = 180.0,
                 max_attempts: int = 3, temperature: float = 0.4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.http.close()

    def health(self) -> bool:
        r = self.http.get(self.base_url.removesuffix("/v1") + "/health")
        return r.status_code == 200

    def complete(self, messages: list[dict], temperature: float | None = None) -> str:
        r = self.http.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": 300,
                # Gemma 4 can think; we want the label only.
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if r.status_code != 200:
            raise SummaryError(f"HTTP {r.status_code}: {r.text[:300]}")
        msg = r.json()["choices"][0]["message"]
        return msg.get("content") or ""

    def summarize(self, item: dict) -> str:
        """One 40-120 word paragraph. Retries off-length output up to max_attempts total calls."""
        messages = build_messages(item)
        last = ""
        for attempt in range(self.max_attempts):
            text = _clean(self.complete(messages, temperature=self.temperature + 0.15 * attempt))
            n = word_count(text)
            if MIN_WORDS <= n <= MAX_WORDS:
                return text
            last = f"{n} words: {text[:200]!r}"
        raise SummaryError(f"no valid summary for {item.get('id')} after {self.max_attempts} tries; last {last}")
