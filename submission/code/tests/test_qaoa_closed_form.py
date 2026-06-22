"""The analytic p=1 formula must match the exact statevector simulator.

This is the central correctness guarantee: the O(|E|) closed form that powers
the fast demo agrees with the unambiguous O(2**n) statevector evaluator to
numerical precision, across random graphs from every family.
"""
import networkx as nx
import numpy as np
import pytest

from topoqaoa import graph_generators as gg
from topoqaoa import qaoa


@pytest.mark.parametrize("family", gg.FAMILIES)
def test_closed_form_matches_statevector(family):
    rng = np.random.default_rng(7)
    for _ in range(4):
        g = gg.generate(family, 8, rng)
        for gamma, beta in [(0.3, 0.4), (1.1, 0.7), (2.0, 1.2), (0.8, 0.2)]:
            cf = qaoa.expectation_p1_closed_form(g, gamma, beta)
            sv = qaoa.expectation_statevector(g, [gamma], [beta])
            assert abs(cf - sv) < 1e-9, (family, gamma, beta, cf, sv)


def test_single_edge_p1_is_exact():
    # p=1 QAOA solves a single edge exactly: max <C> = 1.
    g = nx.Graph([(0, 1)])
    val, _, _ = qaoa.best_p1_on_grid(g, n_gamma=64, n_beta=32)
    assert val > 0.999
