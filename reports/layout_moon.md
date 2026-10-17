# Layout neighbourhood check

Vectors: `data_moon/vectors.npy` (1024 items). UMAP n_components=3, metric=cosine, random_state=42.

Share of items with >= 3 of their 10 nearest 768-d neighbours among their 30 nearest 3D neighbours: **0.946** (threshold 0.60) -> **PASS**

Mean overlap: 6.91 of 10. Median: 7.

| overlap | items |
|---:|---:|
| 0 | 0 |
| 1 | 14 |
| 2 | 41 |
| 3 | 61 |
| 4 | 78 |
| 5 | 103 |
| 6 | 128 |
| 7 | 128 |
| 8 | 136 |
| 9 | 135 |
| 10 | 200 |

## Globe alignment

Layout posed against the tiles' unit-globe positions (RMS spread matched, Procrustes rotation): scale 0.329, reflected False, RMS distance to the tile's globe place **1.289** (same fit against shuffled tiles: 1.378; the unit globe's RMS radius is 1). Max |coordinate| per axis: 2.59, 2.04, 2.00.
