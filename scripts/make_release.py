"""Package the built gallery (main collection + the Moon set) as one tarball for a GitHub Release.

    uv run python scripts/make_release.py [--out dist]

What a fresh clone needs to serve both galleries, relative to the repo root:
  web/data/ web/thumbs/ web/videos/ web/audio/nasa/      the main bundle
  data/vectors.npy data/vector_ids.json data/media/{aic,nasa}/*.jpg   search + placard pictures
  web_moon/   data_moon/{vectors.npy,vector_ids.json,retrieval.json,media/moon/}

Left out: the Art Institute of Chicago audio guides (their audio and transcripts are AIC's own
material, licensed for noncommercial educational use, not for redistribution here). Those items are
removed from items.json, points.bin, vectors.npy and vector_ids.json together, so the bundle stays
consistent; `scripts/rebuild.sh` fetches them from AIC directly. Writes <out>/gallery-data.tar.gz
and gallery-data.tar.gz.sha256, then re-opens the archive and checks every reference resolves.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = "gallery-data.tar.gz"


def is_aic_audio(it: dict) -> bool:
    return it["source"] == "aic" and it["modality"] == "audio"


def filtered_bundle(web: Path) -> tuple[bytes, bytes, bytes, set[str]]:
    """items.json, points.bin and meta.json without the AIC audio guides; returns the dropped ids too."""
    items = json.loads((web / "data" / "items.json").read_text())
    blob = (web / "data" / "points.bin").read_bytes()
    n = len(items)
    xyz = np.frombuffer(blob, dtype="<f4", count=n * 3).reshape(n, 3)
    rgb = np.frombuffer(blob, dtype=np.uint8, offset=n * 12, count=n * 3).reshape(n, 3)
    keep = np.array([not is_aic_audio(it) for it in items])
    dropped = {it["id"] for it in items if is_aic_audio(it)}
    kept = [it for it, k in zip(items, keep) if k]
    points = xyz[keep].astype("<f4").tobytes() + rgb[keep].tobytes()
    meta = json.loads((web / "data" / "meta.json").read_text())
    meta["n"] = len(kept)
    meta["sources"] = {s: sum(it["source"] == s for it in kept) for s in meta["sources"]}
    meta["release_note"] = f"{len(dropped)} AIC audio guides left out of the public data release"
    return (json.dumps(kept, ensure_ascii=False, separators=(",", ":")).encode(), points,
            (json.dumps(meta, indent=2) + "\n").encode(), dropped)


def filtered_vectors(data: Path, dropped: set[str]) -> tuple[bytes, bytes]:
    ids = json.loads((data / "vector_ids.json").read_text())
    vec = np.load(data / "vectors.npy")
    keep = [i for i, v in enumerate(ids) if v not in dropped]
    buf = io.BytesIO()
    np.save(buf, vec[keep])
    return buf.getvalue(), json.dumps([ids[i] for i in keep]).encode()


def add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def add_tree(tar: tarfile.TarFile, path: Path, pattern: str = "*") -> int:
    files = sorted(p for p in path.rglob(pattern) if p.is_file())
    for p in files:
        tar.add(p, arcname=p.relative_to(ROOT).as_posix())
    return len(files)


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    target = out / NAME
    web, data = ROOT / "web", ROOT / "data"
    items, points, meta, dropped = filtered_bundle(web)
    vectors, vector_ids = filtered_vectors(data, dropped)
    with tarfile.open(target, "w:gz") as tar:
        add_bytes(tar, "web/data/items.json", items)
        add_bytes(tar, "web/data/points.bin", points)
        add_bytes(tar, "web/data/meta.json", meta)
        add_bytes(tar, "data/vectors.npy", vectors)
        add_bytes(tar, "data/vector_ids.json", vector_ids)
        counts = {
            "thumbs": add_tree(tar, web / "thumbs"),
            "videos": add_tree(tar, web / "videos"),
            "nasa sounds": add_tree(tar, web / "audio" / "nasa"),
            "placard pictures": sum(add_tree(tar, data / "media" / s, "*.jpg") for s in ("aic", "nasa")),
            "moon bundle": add_tree(tar, ROOT / "web_moon"),
            "moon media": add_tree(tar, ROOT / "data_moon" / "media" / "moon"),
        }
        for f in ("vectors.npy", "vector_ids.json", "retrieval.json"):
            tar.add(ROOT / "data_moon" / f, arcname=f"data_moon/{f}")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (out / f"{NAME}.sha256").write_text(f"{digest}  {NAME}\n")
    print(f"{target}: {target.stat().st_size / 2**20:.0f} MiB, sha256 {digest[:16]}…; "
          f"dropped {len(dropped)} AIC audio guides; {counts}")
    return target


def check(target: Path) -> None:
    """Every file the two bundles reference is in the archive; vectors cover every item; no AIC audio."""
    with tarfile.open(target, "r:gz") as tar:
        names = set(tar.getnames())
        read = lambda n: tar.extractfile(n).read()  # noqa: E731
        for web, data in (("web", "data"), ("web_moon", "data_moon")):
            items = json.loads(read(f"{web}/data/items.json"))
            assert len(read(f"{web}/data/points.bin")) == 15 * len(items), f"{web}: points.bin size"
            ids = set(json.loads(read(f"{data}/vector_ids.json")))
            vec = np.load(io.BytesIO(read(f"{data}/vectors.npy")))
            assert vec.shape == (len(ids), 768), f"{data}: vectors {vec.shape} vs {len(ids)} ids"
            missing = [it["id"] for it in items if it["id"] not in ids]
            assert not missing, f"{web}: items without vectors, e.g. {missing[:3]}"
            refs = [f"{web}/{it[k]}" for it in items for k in ("thumb", "video", "audio") if it.get(k)]
            refs += [f"{data}/{it['thumb'].replace('thumbs/', 'media/', 1)}" for it in items
                     if it.get("thumb") and it["modality"] == "image"]
            gone = [r for r in refs if r not in names]
            assert not gone, f"{web}: {len(gone)} referenced files missing, e.g. {gone[:3]}"
            assert not any(is_aic_audio(it) for it in items), f"{web}: AIC audio guide left in"
            if web == "web_moon":
                atlas = json.loads(read("web_moon/data/meta.json"))["globe"]["atlas"]
                assert f"web_moon/{atlas}" in names, "moon atlas missing"
            print(f"check {web}: {len(items)} items, {len(refs)} referenced files present")
        assert not any(n.startswith("web/audio/aic") for n in names), "AIC audio files in archive"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    args = ap.parse_args()
    check(build(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
