"""QAOA MaxCut objective: exact statevector and closed-form depth-1.

Two independent evaluators of the QAOA expected cut

    <C(gamma, beta)> ,  C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2 :

* ``expectation_statevector`` builds the full QAOA state for arbitrary depth p
  and measures <C>. Unambiguously correct; used for full-depth runs and as the
  reference oracle in tests. Cost O(p * n * 2**n), fine for n <~ 16.

* ``expectation_p1_closed_form`` evaluates the analytic per-edge formula of
  Wang et al. (PRA 97, 022304, 2018) for depth p=1. Cost O(|E|), so it scales to
  graphs far beyond statevector reach and powers the in-browser demo. The test
  suite asserts it matches the statevector evaluator to ~1e-9 on random graphs.

No heavyweight quantum dependency is required; everything is numpy.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------- #
# Exact statevector evaluator (arbitrary depth)                               #
# --------------------------------------------------------------------------- #
def _cost_diagonal(graph: nx.Graph, n: int) -> np.ndarray:
    x = np.arange(2 ** n, dtype=np.int64)
    bit_shifts = np.arange(n - 1, -1, -1)
    bits = (x[:, None] >> bit_shifts[None, :]) & 1  # qubit j = column j (MSB-first)
    cost = np.zeros(2 ** n)
    for u, v in graph.edges():
        cost += (bits[:, int(u)] != bits[:, int(v)]).astype(float)
    return cost


def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = np.cos(beta), np.sin(beta)
    tensor = psi.reshape((2,) * n)
    for j in range(n):
        tensor = c * tensor - 1j * s * np.flip(tensor, axis=j)
    return tensor.reshape(-1)


def qaoa_state(graph: nx.Graph, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    n = graph.number_of_nodes()
    cost = _cost_diagonal(graph, n)
    psi = np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        psi = np.exp(-1j * gamma * cost) * psi
        psi = _apply_mixer(psi, beta, n)
    return psi


def expectation_statevector(
    graph: nx.Graph, gammas: Sequence[float], betas: Sequence[float]
) -> float:
    n = graph.number_of_nodes()
    cost = _cost_diagonal(graph, n)
    psi = qaoa_state(graph, gammas, betas)
    return float(np.sum(cost * np.abs(psi) ** 2))


# --------------------------------------------------------------------------- #
# Closed-form depth-1 evaluator (Wang et al. 2018)                            #
# --------------------------------------------------------------------------- #
def expectation_p1_closed_form(graph: nx.Graph, gamma: float, beta: float) -> float:
    """Analytic <C> for p=1 QAOA on MaxCut.

    Per edge (u, v) with d_u, d_v = (degree - 1) [neighbours other than the
    partner] and f = common neighbours (triangles on the edge):

        <C_uv> = 1/2
               + 1/4 sin(4b) sin(g) [cos(g)^d_u + cos(g)^d_v]
               - 1/4 sin(2b)^2 cos(g)^(d_u+d_v-2f) [1 - cos(2g)^f]
    """
    adj = {v: set(graph.neighbors(v)) for v in graph.nodes()}
    cg, c2g = np.cos(gamma), np.cos(2 * gamma)
    s4b, s2b = np.sin(4 * beta), np.sin(2 * beta)
    total = 0.0
    for u, v in graph.edges():
        du = len(adj[u]) - 1
        dv = len(adj[v]) - 1
        f = len(adj[u] & adj[v])
        term1 = 0.25 * s4b * np.sin(gamma) * (cg ** du + cg ** dv)
        term2 = 0.25 * (s2b ** 2) * (cg ** (du + dv - 2 * f)) * (1.0 - c2g ** f)
        total += 0.5 + term1 - term2
    return float(total)


# --------------------------------------------------------------------------- #
# Convenience: grid search over p=1 angles (used by demo / topology policy)   #
# --------------------------------------------------------------------------- #
def best_p1_on_grid(
    graph: nx.Graph, n_gamma: int = 24, n_beta: int = 12
) -> Tuple[float, float, float]:
    """Return ``(best_value, gamma, beta)`` from a closed-form grid search."""
    gammas = np.linspace(0, np.pi, n_gamma, endpoint=False)
    betas = np.linspace(0, np.pi / 2, n_beta, endpoint=False)
    best = (-1.0, 0.0, 0.0)
    for g in gammas:
        for b in betas:
            val = expectation_p1_closed_form(graph, g, b)
            if val > best[0]:
                best = (val, float(g), float(b))
    return best
