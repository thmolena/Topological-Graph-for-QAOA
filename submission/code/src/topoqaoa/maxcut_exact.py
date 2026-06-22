"""Exact MaxCut oracle by brute-force enumeration (small graphs only).

For ``n`` nodes we enumerate the ``2**(n-1)`` distinct bipartitions (node 0 is
fixed to break the global Z2 symmetry). This is the ground-truth used to compute
*exact* approximation ratios, so it must be correct, not approximate. Guarded to
``n <= MAX_EXACT_NODES`` to keep runtime bounded on commodity hardware.
"""
from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np

MAX_EXACT_NODES = 20


def edge_array(graph: nx.Graph) -> np.ndarray:
    """Return an ``(m, 2)`` int array of edges with contiguous node ids."""
    return np.array([(int(u), int(v)) for u, v in graph.edges()], dtype=np.int64)


def cut_value(graph: nx.Graph, assignment: np.ndarray) -> int:
    """Number of edges crossing the partition given a {0,1} assignment."""
    edges = edge_array(graph)
    if edges.size == 0:
        return 0
    a = assignment.astype(np.int64)
    return int(np.sum(a[edges[:, 0]] != a[edges[:, 1]]))


def max_cut(graph: nx.Graph) -> Tuple[int, np.ndarray]:
    """Return ``(max_cut_value, optimal_assignment)`` by exact enumeration."""
    n = graph.number_of_nodes()
    if n > MAX_EXACT_NODES:
        raise ValueError(
            f"exact MaxCut limited to n<={MAX_EXACT_NODES}; got n={n}. "
            "Use a heuristic oracle for larger graphs."
        )
    edges = edge_array(graph)
    best_val, best_assign = -1, np.zeros(n, dtype=np.int64)
    # Fix node 0 in partition 0; enumerate the other n-1 bits.
    for mask in range(2 ** (n - 1)):
        assign = np.zeros(n, dtype=np.int64)
        for b in range(n - 1):
            assign[b + 1] = (mask >> b) & 1
        val = int(np.sum(assign[edges[:, 0]] != assign[edges[:, 1]])) if edges.size else 0
        if val > best_val:
            best_val, best_assign = val, assign.copy()
    return best_val, best_assign


def random_cut_expectation(graph: nx.Graph) -> float:
    """Expected cut of a uniformly random assignment (= |E|/2)."""
    return graph.number_of_edges() / 2.0
