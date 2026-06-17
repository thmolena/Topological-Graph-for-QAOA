"""Family-holdout integrity (no leakage) and metric sanity."""
import numpy as np
import pytest

from topoqaoa import metrics
from topoqaoa import graph_generators as gg
from topoqaoa.splits import build_dataset, family_holdout, has_leakage


def test_family_holdout_no_leakage():
    rng = np.random.default_rng(11)
    data = build_dataset(gg.FAMILIES, n_per_family=3, sizes=[6, 8], rng=rng)
    for held in gg.FAMILIES:
        split = family_holdout(data, held)
        assert all(d.family == held for d in split["test"])
        assert all(d.family != held for d in split["train"])
        assert not has_leakage(split["train"], split["test"])


def test_metrics_bounds():
    assert metrics.approximation_ratio(3, 4) == 0.75
    assert metrics.regret(0.9) == pytest.approx(0.1)
    hist = [(1, 0.5), (2, 0.8), (3, 0.96)]
    assert metrics.queries_to_target(hist, 0.95) == 3
    assert metrics.queries_to_target(hist, 0.99) == -1
