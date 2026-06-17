"""Graph-family generators for MaxCut benchmarks.

Six families spanning the regimes that arise in molecular and biomedical
optimization graphs: Erdos-Renyi (unstructured), random regular (degree
homogeneous), Barabasi-Albert (scale-free, hub structure like PPI networks),
Watts-Strogatz (small-world, like metabolic/contact graphs), 2D grid (locally
tree-like / lattice) and stochastic block (community structure, like cell-type
graphs).

All graphs are returned as ``networkx.Graph`` with integer node labels and unit
edge weights. Generators are seeded for exact reproducibility.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import networkx as nx
import numpy as np

FAMILIES: List[str] = [
    "erdos_renyi",
    "regular",
    "barabasi_albert",
    "watts_strogatz",
    "grid",
    "stochastic_block",
]


def _seed_int(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2 ** 31 - 1))


def erdos_renyi(n: int, rng: np.random.Generator, p: float = 0.5) -> nx.Graph:
    return nx.gnp_random_graph(n, p, seed=_seed_int(rng))


def regular(n: int, rng: np.random.Generator, d: int = 3) -> nx.Graph:
    d = min(d, n - 1)
    if (n * d) % 2 != 0:  # n*d must be even for a d-regular graph
        d -= 1
    return nx.random_regular_graph(d, n, seed=_seed_int(rng))


def barabasi_albert(n: int, rng: np.random.Generator, m: int = 2) -> nx.Graph:
    m = max(1, min(m, n - 1))
    return nx.barabasi_albert_graph(n, m, seed=_seed_int(rng))


def watts_strogatz(n: int, rng: np.random.Generator, k: int = 4, p: float = 0.3) -> nx.Graph:
    k = min(k, n - 1)
    if k % 2 != 0:
        k -= 1
    k = max(2, k)
    return nx.watts_strogatz_graph(n, k, p, seed=_seed_int(rng))


def grid(n: int, rng: np.random.Generator) -> nx.Graph:
    side = max(2, int(round(np.sqrt(n))))
    g = nx.grid_2d_graph(side, side)
    return nx.convert_node_labels_to_integers(g)


def stochastic_block(n: int, rng: np.random.Generator, blocks: int = 2) -> nx.Graph:
    blocks = max(2, min(blocks, n // 2))
    sizes = [n // blocks] * blocks
    sizes[-1] += n - sum(sizes)
    p_in, p_out = 0.7, 0.1
    probs = [[p_in if i == j else p_out for j in range(blocks)] for i in range(blocks)]
    g = nx.stochastic_block_model(sizes, probs, seed=_seed_int(rng))
    return nx.Graph(g)


_GENERATORS: Dict[str, Callable[..., nx.Graph]] = {
    "erdos_renyi": erdos_renyi,
    "regular": regular,
    "barabasi_albert": barabasi_albert,
    "watts_strogatz": watts_strogatz,
    "grid": grid,
    "stochastic_block": stochastic_block,
}


def generate(family: str, n: int, rng: np.random.Generator) -> nx.Graph:
    """Generate one connected graph from ``family`` with ``n`` nodes."""
    if family not in _GENERATORS:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")
    for _ in range(50):
        g = _GENERATORS[family](n, rng)
        g = nx.convert_node_labels_to_integers(g)
        if g.number_of_edges() == 0:
            continue
        # Use the largest connected component so MaxCut is well posed.
        if not nx.is_connected(g):
            comp = max(nx.connected_components(g), key=len)
            g = nx.convert_node_labels_to_integers(g.subgraph(comp).copy())
        if g.number_of_nodes() >= 2 and g.number_of_edges() > 0:
            return g
    raise RuntimeError(f"failed to generate a valid {family} graph with n={n}")


def relabel(graph: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Return an isomorphic copy with randomly permuted node labels.

    Used to test that learned policies and topology descriptors are invariant
    to node ordering (a hard requirement for graph-structured inputs).
    """
    perm = rng.permutation(graph.number_of_nodes())
    mapping = {i: int(perm[i]) for i in range(graph.number_of_nodes())}
    return nx.relabel_nodes(graph, mapping, copy=True)
