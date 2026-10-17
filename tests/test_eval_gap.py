import numpy as np

from gallery.eval_gap import K, measure, top_k


def test_quota_takes_four_per_source_sorted():
    rng = np.random.default_rng(0)
    sources = np.array(["aic"] * 20 + ["nasa"] * 20 + ["wiki"] * 20)
    sims = rng.random(60)
    sims[:20] += 1.0  # aic dominates raw ranking
    raw = top_k(sims, sources, "raw")
    assert set(sources[raw]) == {"aic"}
    quota = top_k(sims, sources, "quota")
    assert len(quota) == K
    assert {s: int((sources[quota] == s).sum()) for s in ("aic", "nasa", "wiki")} == {"aic": 4, "nasa": 4, "wiki": 4}
    assert np.all(np.diff(sims[quota]) <= 0)


def test_measure_flags_mixed_walls():
    sources = np.array(["aic"] * 12 + ["wiki"] * 12)
    vectors = np.eye(24, dtype=np.float32)
    q = np.zeros((2, 24), dtype=np.float32)
    q[0, :12] = 1.0           # only aic close -> not mixed
    q[1, 6:18] = 1.0          # spans both -> mixed
    df = measure(q, vectors, sources, ["a", "b"], "raw")
    assert df["mixed"].tolist() == [False, True]
