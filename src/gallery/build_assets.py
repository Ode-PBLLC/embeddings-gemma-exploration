"""Front-end bundle (US-013).

    uv run python -m gallery.build_assets [--include-pending]

Reads (GALLERY_DATA_DIR, default data/): items.parquet, summaries.jsonl, vector_ids.json, layout.npy.
Writes (GALLERY_ASSETS_DIR, default web/):
  data/points.bin  N*15 bytes: float32 xyz for all points (N*12 bytes), then uint8 rgb (N*3 bytes)
  data/items.json  list, index-aligned with points.bin
  data/meta.json   counts, flags and the binary layout description
  thumbs/<source>/<file>.jpg  copies of the UI thumbnails
  data/<atlas>     only for a dataset with <data>/globe.json (the Moon): the globe texture; each
                   item then carries "tile": [x, y] and meta.json a "globe" block

Only items with a summary are included. --include-pending also includes items still waiting for
curation, with their raw_text truncated as the summary and `summary_pending: true`.
Asset order is items.parquet order restricted to the included items, so it is a subsequence of
vector_ids.json; the search API uses the same subset, and a neighbour's `index` is the asset index.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gallery.paths import assets_dir, data_dir
from gallery.schema import ROOT

PENDING_CHARS = 420
MAX_ITEMS_JSON = 6 * 1024 * 1024


def load_summaries(path: Path) -> dict[str, str]:
    """Last line wins (the file is append-only). Blank summaries count as missing."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open() as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: bad JSON line: {e}") from e
            s = (rec.get("summary") or "").strip()
            if s:
                out[rec["id"]] = s
            else:
                out.pop(rec["id"], None)
    return out


def truncate(text: str, n: int = PENDING_CHARS) -> str:
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:.") + "…"


def _rgb(hex_colors: pd.Series) -> np.ndarray:
    return np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hex_colors], dtype=np.uint8)


def build(data: Path, out: Path, include_pending: bool = False) -> dict:
    items = pd.read_parquet(data / "items.parquet")
    ids = json.loads((data / "vector_ids.json").read_text())
    layout = np.load(data / "layout.npy")
    if items["id"].tolist() != ids:
        raise ValueError("vector_ids.json is not in items.parquet order")
    if layout.shape != (len(items), 3):
        raise ValueError(f"layout.npy shape {layout.shape} != ({len(items)}, 3)")

    globe = json.loads((data / "globe.json").read_text()) if (data / "globe.json").is_file() else None
    summaries = load_summaries(data / "summaries.jsonl")
    has = items["id"].isin(summaries)
    keep = np.ones(len(items), bool) if include_pending else has.to_numpy()
    rows = np.flatnonzero(keep)
    if len(rows) == 0:
        raise ValueError(f"no items to include: {len(summaries)} summaries match none of {len(items)} items "
                         "(use --include-pending while curation runs)")
    sel = items.iloc[rows].reset_index(drop=True)

    (out / "data").mkdir(parents=True, exist_ok=True)
    thumbs_out = out / "thumbs"
    records, copied = [], 0
    videos_out = out / "videos"
    audio_out = out / "audio"
    for r in sel.itertuples(index=False):
        thumb = video = ""
        sound = None
        if r.modality == "audio":
            from gallery.ingest.nasa_audio import audio_paths

            asrc = Path(r.media_path)
            asrc = asrc if asrc.is_absolute() else ROOT / asrc
            side = audio_paths(r.id)[1]
            if not asrc.is_file() or not side.is_file():
                raise FileNotFoundError(f"sound file or sidecar missing for {r.id}: {asrc}, {side}")
            arel = Path(r.source) / asrc.name
            adst = audio_out / arel
            if not adst.is_file() or adst.stat().st_size != asrc.stat().st_size:
                adst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(asrc, adst)
                copied += 1
            meta_s = json.loads(side.read_text())
            if "speech" not in meta_s:
                raise ValueError(f"{r.id} has not been transcribed (run gallery.transcribe)")
            if not meta_s["speech"] and not meta_s.get("caption"):
                raise ValueError(f"{r.id} is wordless but has no caption (run gallery.sound_captions)")
            sound = {"audio": f"audio/{arel.as_posix()}", "duration": meta_s["duration"], "peaks": meta_s["peaks"],
                     "words": meta_s.get("words") or [], "caption": meta_s.get("caption", ""), "license": r.license}
        if r.modality == "video":
            vsrc = Path(r.media_path)
            vsrc = vsrc if vsrc.is_absolute() else ROOT / vsrc
            if not vsrc.is_file():
                raise FileNotFoundError(f"video loop missing for {r.id}: {vsrc}")
            vrel = Path(r.source) / vsrc.name
            vdst = videos_out / vrel
            if not vdst.is_file() or vdst.stat().st_size != vsrc.stat().st_size:
                vdst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(vsrc, vdst)
                copied += 1
            video = f"videos/{vrel.as_posix()}"
        if r.modality in ("image", "video"):
            src = Path(r.thumb_path)
            src = src if src.is_absolute() else ROOT / src
            if not src.is_file():
                raise FileNotFoundError(f"thumbnail missing for {r.id}: {src}")
            rel = Path(r.source) / src.name
            dst = thumbs_out / rel
            if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied += 1
            thumb = f"thumbs/{rel.as_posix()}"
        pending = r.id not in summaries
        rec = {
            "id": r.id, "source": r.source, "modality": r.modality, "title": r.title,
            "creator": r.creator or "", "date": r.date or "",
            "summary": truncate(r.raw_text) if pending else summaries[r.id],
            "thumb": thumb, "source_url": r.source_url, "topic": r.topic or "", "color": r.color_hex,
        }
        if video:
            rec["video"] = video   # short silent loop (the wall and placard play it)
        if sound:
            rec.update(sound)      # playable sound, waveform, timed words or a static caption
        if globe:
            rec["tile"] = globe["tiles"][r.id]   # [x, y] in the globe's XYZ grid
        if pending:
            rec["summary_pending"] = True
        records.append(rec)

    xyz = layout[rows].astype("<f4")
    rgb = _rgb(sel["color_hex"])
    blob = xyz.tobytes() + rgb.tobytes()
    assert len(blob) == len(rows) * 15
    (out / "data" / "points.bin").write_bytes(blob)

    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    if len(payload.encode()) > MAX_ITEMS_JSON:
        raise ValueError(f"items.json would be {len(payload.encode())} bytes (> {MAX_ITEMS_JSON})")
    (out / "data" / "items.json").write_text(payload, encoding="utf-8")

    if globe:
        shutil.copyfile(data / globe["atlas"], out / "data" / globe["atlas"])
    meta = {
        "n": len(rows),
        "n_items_total": len(items),
        "n_pending": int((~has.to_numpy()[rows]).sum()),
        "sources": sel["source"].value_counts().to_dict(),
        "points_layout": "float32 xyz [N*3] little-endian, then uint8 rgb [N*3]",
        "synthetic": (data / "fake_topics.npz").is_file(),
        "data_dir": str(data),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if globe:
        meta["globe"] = {"z": globe["z"], "cell": globe["cell"], "atlas": f"data/{globe['atlas']}"}
    (out / "data" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    meta["thumbs_copied"] = copied
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="assets root (default GALLERY_ASSETS_DIR or web/)")
    ap.add_argument("--include-pending", action="store_true",
                    help="dev: include items without a summary (raw_text truncated, flagged summary_pending)")
    args = ap.parse_args()
    meta = build(args.data_dir or data_dir(), args.out or assets_dir(), args.include_pending)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
