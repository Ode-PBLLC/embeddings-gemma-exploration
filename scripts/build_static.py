"""Build the static site (GitHub Pages): the gallery and the Moon with search running in the browser.

    python scripts/build_static.py [--root .] [--out dist/site]

`--root` holds the built bundles in the repo layout (web/, data/, web_moon/, data_moon/): the repo
itself after a build, or the extracted public data release (gallery-data.tar.gz) in CI. The front-end
code, the front page and the precomputed demo queries always come from this checkout. Needs only numpy.

    <out>/index.html            front page (static/front/)
    <out>/gallery/, <out>/moon/ the front end (web/) with its bundle, plus for in-browser search:
        data/vectors.bin        float32 little-endian, items.json order, 768 per item
        data/demo_queries.json  precomputed query vectors (scripts/embed_demo_queries.py)
        data/meta.json          + "static": model (static/model.json) and retrieval config

Left out, as in the public data release: the AIC audio guides (scripts/make_release.py). Also left out:
the larger placard images (data/media, ~800 MB); the placard shows the thumbnail. Fails if any
reference is missing, a file passes 100 MB, or the site passes 1 GB (GitHub Pages limits).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_release import filtered_bundle, is_aic_audio  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
DIM = 768
MAX_FILE = 100 * 2**20
MAX_SITE = 2**30
FRONT_END = ("index.html", "css", "js", "vendor")


def aligned_vectors(data: Path, items: list[dict]) -> np.ndarray:
    """vectors.npy rows in items.json order (gallery.app.load_searcher does the same)."""
    vectors = np.load(data / "vectors.npy", mmap_mode="r")
    ids = json.loads((data / "vector_ids.json").read_text())
    row = {v: i for i, v in enumerate(ids)}
    missing = [it["id"] for it in items if it["id"] not in row]
    if missing:
        raise ValueError(f"{data}: {len(missing)} items have no vector, e.g. {missing[:3]}")
    out = np.asarray(vectors[[row[it["id"]] for it in items]], dtype="<f4")
    if out.shape != (len(items), DIM) or not np.isfinite(out).all():
        raise ValueError(f"{data}: bad vectors {out.shape}")
    return out


def retrieval(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if cfg.get("mode", "raw") not in ("raw", "quota") or "weak_threshold" not in cfg:
        raise ValueError(f"{path}: bad retrieval config {cfg}")
    return {"mode": cfg.get("mode", "raw"), "k": int(cfg.get("k", 12)), "weak_threshold": float(cfg["weak_threshold"])}


def copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_set(name: str, web: Path, data: Path, retrieval_path: Path, out: Path,
              model: dict, demos: list[dict]) -> int:
    """One collection at <out>/<name>/. Returns its item count."""
    dst = out / name
    for part in FRONT_END:
        copy_tree(CODE / "web" / part, dst / part)
    items_b, points_b, meta_b, _dropped = filtered_bundle(web)
    items = json.loads(items_b)
    meta = json.loads(meta_b)
    meta.pop("data_dir", None)   # a local path; not for publishing
    note = json.loads((web / "data" / "meta.json").read_text()).get("release_note")
    if note:   # built from the release, which already left them out: keep its count, not "0"
        meta["release_note"] = note
    meta["static"] = {"model": model, "retrieval": retrieval(retrieval_path), "placards": False}
    (dst / "data").mkdir(parents=True, exist_ok=True)
    (dst / "data" / "items.json").write_bytes(items_b)
    (dst / "data" / "points.bin").write_bytes(points_b)
    (dst / "data" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (dst / "data" / "vectors.bin").write_bytes(aligned_vectors(data, items).tobytes())
    (dst / "data" / "demo_queries.json").write_text(json.dumps(demos))
    for extra in sorted((web / "data").iterdir()):   # e.g. the Moon's globe atlas
        if extra.name not in ("items.json", "points.bin", "meta.json"):
            copy_tree(extra, dst / "data" / extra.name)
    for it in items:   # only the media the kept items use: never the AIC audio guides
        for key in ("thumb", "video", "audio"):
            if it.get(key):
                copy_tree(web / it[key], dst / it[key])
    return len(items)


def check(out: Path) -> dict:
    """Every reference resolves, no AIC audio, vectors match items, Pages size limits hold."""
    sizes = {}
    for name in ("gallery", "moon"):
        d = out / name
        items = json.loads((d / "data" / "items.json").read_text())
        assert not any(is_aic_audio(it) for it in items), f"{name}: AIC audio guide in items"
        assert (d / "data" / "points.bin").stat().st_size == 15 * len(items), f"{name}: points.bin"
        assert (d / "data" / "vectors.bin").stat().st_size == 4 * DIM * len(items), f"{name}: vectors.bin"
        refs = [it[k] for it in items for k in ("thumb", "video", "audio") if it.get(k)]
        gone = [r for r in refs if not (d / r).is_file()]
        assert not gone, f"{name}: {len(gone)} missing files, e.g. {gone[:3]}"
        meta = json.loads((d / "data" / "meta.json").read_text())
        if meta.get("globe"):
            assert (d / meta["globe"]["atlas"]).is_file(), f"{name}: globe atlas missing"
        demos = json.loads((d / "data" / "demo_queries.json").read_text())
        assert demos, f"{name}: no demo queries"
    assert not list(out.rglob("audio/aic")), "AIC audio directory published"
    files = [p for p in out.rglob("*") if p.is_file()]
    big = [(p, p.stat().st_size) for p in files if p.stat().st_size > MAX_FILE]
    assert not big, f"files over 100 MB: {big}"
    total = sum(p.stat().st_size for p in files)
    assert total < MAX_SITE, f"site is {total / 2**20:.0f} MiB, over the 1 GB Pages limit"
    sizes["files"], sizes["mib"] = len(files), round(total / 2**20, 1)
    return sizes


def build(root: Path, out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model = json.loads((CODE / "static" / "model.json").read_text())
    demos = json.loads((CODE / "static" / "demo_queries.json").read_text())
    counts = {
        "gallery": build_set("gallery", root / "web", root / "data", CODE / "config" / "retrieval.json",
                             out, model, demos["gallery"]),
        "moon": build_set("moon", root / "web_moon", root / "data_moon", root / "data_moon" / "retrieval.json",
                          out, model, demos["moon"]),
    }
    for f in sorted((CODE / "static" / "front").iterdir()):
        copy_tree(f, out / f.name)
    mb = {k: round(sum(v["files"].values()) / 1e6) for k, v in model["paths"].items()}
    front = (out / "index.html").read_text()
    for key, value in {"gallery_n": counts["gallery"], "moon_n": counts["moon"],
                       "webgpu_mb": mb["webgpu"], "wasm_mb": mb["wasm"]}.items():
        front = front.replace("{{" + key + "}}", f"{value:,}")
    if "{{" in front:
        raise ValueError(f"front page has unfilled placeholders: {front[front.index('{{'):][:40]}")
    (out / "index.html").write_text(front)
    (out / ".nojekyll").write_text("")
    return {"items": counts, **check(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=CODE)
    ap.add_argument("--out", type=Path, default=CODE / "dist" / "site")
    args = ap.parse_args()
    print(json.dumps(build(args.root.resolve(), args.out.resolve())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
