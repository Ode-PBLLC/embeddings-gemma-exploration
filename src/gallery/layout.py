"""3D UMAP layout of the item vectors (US-012).

    uv run python -m gallery.layout [--data-dir DIR] [--report PATH] [--align-globe]

Reads <data>/vectors.npy, writes <data>/layout.npy (float32, (N, 3), each axis in [-1, 1])
and a neighbourhood-preservation report (default reports/layout.md).
Deterministic: UMAP random_state=42 (single-threaded) gives byte-equal output across runs.

--align-globe (a dataset with <data>/globe.json, i.e. the Moon): instead of scaling each axis to
[-1, 1], pose the layout against the tiles' places on the unit globe: centred on the globe, scaled
uniformly so its RMS radius equals the globe's (the two spaces span the same range), and rotated
(or reflected) by orthogonal Procrustes so each tile travels as little as possible in the
globe -> cloud morph. Distances within the layout keep their ratios. (A least-squares scale is not
used: where the embedding carries little geography it shrinks the cloud toward the centre.)
The report states how much the rotation helps against tiles in shuffled order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gallery.paths import data_dir
from gallery.schema import ROOT

SEED = 42
K_HIGH = 10   # nearest neighbours in 768-d
K_LOW = 30    # nearest neighbours in 3D
MIN_OVERLAP = 3
PASS_SHARE = 0.60


def umap_3d(vectors: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    import umap  # slow import (numba); keep it out of module import

    reducer = umap.UMAP(
        n_components=3,
        metric="cosine",
        random_state=SEED,
        n_neighbors=min(n_neighbors, len(vectors) - 1),
        min_dist=min_dist,
    )
    return reducer.fit_transform(vectors.astype(np.float32))


def scale_unit(xyz: np.ndarray) -> np.ndarray:
    """Scale each axis independently to [-1, 1]."""
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return ((xyz - lo) / span * 2.0 - 1.0).astype(np.float32)


def globe_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Unit-sphere position, y up, longitude 0 facing +z (web/js/globe.js uses the same convention)."""
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=1)


def tile_centres(globe: dict, ids: list[str]) -> np.ndarray:
    """Globe position of the centre of each item's XYZ web-mercator tile."""
    n = 2 ** globe["z"]
    xy = np.array([globe["tiles"][i] for i in ids], dtype=np.float64)
    lon = (xy[:, 0] + 0.5) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (xy[:, 1] + 0.5) / n))))
    return globe_xyz(lon, lat)


def align_similarity(src: np.ndarray, dst: np.ndarray, scale: str = "match") -> tuple[np.ndarray, dict]:
    """s * src @ R + t ~ dst: R by orthogonal Procrustes; s matching RMS spreads ("match") or least squares ("lsq")."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    u, sv, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    s = np.sqrt((b ** 2).sum() / (a ** 2).sum()) if scale == "match" else sv.sum() / (a ** 2).sum()
    out = s * a @ r + dst.mean(0)
    rms = float(np.sqrt(((out - dst) ** 2).sum(1).mean()))
    return out.astype(np.float32), {"scale": float(s), "rms": rms, "reflected": bool(np.linalg.det(r) < 0)}


def neighbourhood_overlap(vectors: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Per item: how many of its K_HIGH cosine neighbours are among its K_LOW 3D neighbours."""
    from sklearn.neighbors import NearestNeighbors

    n = len(vectors)
    kh, kl = min(K_HIGH, n - 1), min(K_LOW, n - 1)
    hi = NearestNeighbors(n_neighbors=kh + 1, metric="cosine").fit(vectors)
    lo = NearestNeighbors(n_neighbors=kl + 1).fit(layout)
    _, hi_idx = hi.kneighbors(vectors)
    _, lo_idx = lo.kneighbors(layout)
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        h = set(hi_idx[i][hi_idx[i] != i][:kh].tolist())
        l = set(lo_idx[i][lo_idx[i] != i][:kl].tolist())
        out[i] = len(h & l)
    return out


def write_report(path: Path, overlap: np.ndarray, src: Path) -> float:
    share = float((overlap >= MIN_OVERLAP).mean())
    verdict = "PASS" if share >= PASS_SHARE else "FAIL"
    hist = np.bincount(overlap, minlength=K_HIGH + 1)
    lines = [
        "# Layout neighbourhood check",
        "",
        f"Vectors: `{src}` ({len(overlap)} items). UMAP n_components=3, metric=cosine, random_state={SEED}.",
        "",
        f"Share of items with >= {MIN_OVERLAP} of their {K_HIGH} nearest 768-d neighbours among their "
        f"{K_LOW} nearest 3D neighbours: **{share:.3f}** (threshold {PASS_SHARE:.2f}) -> **{verdict}**",
        "",
        f"Mean overlap: {overlap.mean():.2f} of {K_HIGH}. Median: {int(np.median(overlap))}.",
        "",
        "| overlap | items |",
        "|---:|---:|",
        *[f"| {i} | {int(c)} |" for i, c in enumerate(hist)],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return share


def run(data: Path, report: Path, align_globe: bool = False) -> tuple[np.ndarray, float]:
    import json

    vectors = np.load(data / "vectors.npy")
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError(f"{data / 'vectors.npy'}: expected finite (N, D) array, got {vectors.shape}")
    raw = umap_3d(vectors)
    fit = None
    if align_globe:
        globe = json.loads((data / "globe.json").read_text())
        ids = json.loads((data / "vector_ids.json").read_text())
        target = tile_centres(globe, ids)
        layout, fit = align_similarity(raw, target)
        # the same fit with no embedding signal: tiles in random order (what "no alignment" scores)
        rng = np.random.default_rng(SEED)
        _, base = align_similarity(raw, target[rng.permutation(len(target))])
        fit["rms_shuffled"] = base["rms"]
        fit["extent"] = [float(v) for v in np.abs(layout).max(0)]
    else:
        layout = scale_unit(raw)
    np.save(data / "layout.npy", layout)
    share = write_report(report, neighbourhood_overlap(vectors, layout), data / "vectors.npy")
    if fit:
        with report.open("a") as f:
            f.write("\n## Globe alignment\n\n"
                    f"Layout posed against the tiles' unit-globe positions (RMS spread matched, Procrustes "
                    f"rotation): scale {fit['scale']:.3f}, "
                    f"reflected {fit['reflected']}, RMS distance to the tile's globe place **{fit['rms']:.3f}** "
                    f"(same fit against shuffled tiles: {fit['rms_shuffled']:.3f}; the unit globe's RMS radius is 1). "
                    f"Max |coordinate| per axis: {', '.join(f'{v:.2f}' for v in fit['extent'])}.\n")
        print(f"globe fit: {fit}")
    return layout, share


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "layout.md")
    ap.add_argument("--align-globe", action="store_true", help="pose the layout to fit <data>/globe.json")
    args = ap.parse_args()
    data = args.data_dir or data_dir()
    layout, share = run(data, args.report, args.align_globe)
    print(f"layout {layout.shape} -> {data / 'layout.npy'}; neighbourhood share {share:.3f}; report {args.report}")


if __name__ == "__main__":
    main()
