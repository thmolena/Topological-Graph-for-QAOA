"""Exact MaxCut oracle against analytically known small cases."""
import networkx as nx

from topoqaoa.maxcut_exact import cut_value, max_cut


def test_triangle():
    # K3: any bipartition cuts at most 2 of 3 edges.
    g = nx.cycle_graph(3)
    val, assign = max_cut(g)
    assert val == 2
    assert cut_value(g, assign) == 2


def test_square_is_bipartite():
    # C4 is bipartite -> all 4 edges cut.
    g = nx.cycle_graph(4)
    val, _ = max_cut(g)
    assert val == 4


def test_complete_k4():
    # K4 max cut = 4 (balanced 2-2 split).
    g = nx.complete_graph(4)
    val, _ = max_cut(g)
    assert val == 4


def test_star_all_cut():
    # Star S5: bipartite, all 5 edges cut.
    g = nx.star_graph(5)
    val, _ = max_cut(g)
    assert val == 5
