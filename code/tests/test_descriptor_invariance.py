"""Topology descriptors must be invariant to node relabeling."""
import numpy as np

from topoqaoa import descriptors
from topoqaoa import graph_generators as gg


def test_relabeling_invariance():
    rng = np.random.default_rng(3)
    for family in gg.FAMILIES:
        g = gg.generate(family, 10, rng)
        d0 = descriptors.describe(g)
        for _ in range(3):
            gp = gg.relabel(g, rng)
            dp = descriptors.describe(gp)
            assert np.allclose(d0, dp, atol=1e-8), family


def test_descriptor_length_matches_names():
    rng = np.random.default_rng(1)
    g = gg.generate("erdos_renyi", 9, rng)
    assert len(descriptors.describe(g)) == len(descriptors.FEATURE_NAMES)
