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


# --------------------------------------------------------------------------- #
# Depth-p angle schedules: caching, split convention, fast cached evaluator    #
# --------------------------------------------------------------------------- #
# A depth-p warm start is a flat schedule  theta = [gamma_1..gamma_p, beta_1..beta_p].
# The cost diagonal depends only on the graph, so we precompute it once per graph
# and reuse it across the many objective calls an optimizer or refiner makes.

def cost_diagonal(graph: nx.Graph) -> np.ndarray:
    """Public handle on the per-basis-state cut values (cached by the caller)."""
    return _cost_diagonal(graph, graph.number_of_nodes())


def split_schedule(theta: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Split a flat length-``2p`` schedule into ``(gammas, betas)``."""
    theta = np.asarray(theta, dtype=float)
    p = theta.size // 2
    return theta[:p], theta[p:]


def expectation_from_cost(
    cost: np.ndarray, n: int, gammas: Sequence[float], betas: Sequence[float]
) -> float:
    """``<C>`` for a precomputed cost diagonal (statevector, arbitrary depth)."""
    psi = np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        psi = np.exp(-1j * gamma * cost) * psi
        psi = _apply_mixer(psi, beta, n)
    return float(np.sum(cost * np.abs(psi) ** 2))


def expectation_schedule(
    graph: nx.Graph, theta: Sequence[float], cost: np.ndarray | None = None
) -> float:
    """``<C>`` for a flat depth-p schedule on ``graph`` (statevector)."""
    n = graph.number_of_nodes()
    if cost is None:
        cost = _cost_diagonal(graph, n)
    gammas, betas = split_schedule(theta)
    return expectation_from_cost(cost, n, gammas, betas)


# --------------------------------------------------------------------------- #
# INTERP-seeded schedule optimization (training-label oracle)                  #
# --------------------------------------------------------------------------- #
def _interp_seed(theta_p: np.ndarray) -> np.ndarray:
    """Linear-interpolation (INTERP, Zhou et al. 2020) seed from depth p to p+1.

    The optimized depth-p schedule is resampled onto a depth-(p+1) grid, which
    places the (p+1)-layer search inside the basin of the p-layer optimum and is
    the standard way to grow good QAOA schedules with depth.
    """
    gammas, betas = split_schedule(theta_p)
    p = gammas.size
    if p == 1:
        return np.concatenate([np.full(2, gammas[0]), np.full(2, betas[0])])
    xp = np.linspace(0.0, 1.0, p)
    xnew = np.linspace(0.0, 1.0, p + 1)
    return np.concatenate([np.interp(xnew, xp, gammas), np.interp(xnew, xp, betas)])


def optimize_schedule(
    graph: nx.Graph,
    p: int,
    rng: np.random.Generator | None = None,
    restarts: int = 2,
) -> Tuple[float, np.ndarray]:
    """Best depth-``p`` schedule found by INTERP growth plus random restarts.

    Used only to *label* training graphs with near-optimal angle schedules; the
    learned policies must then predict these from topology alone. Returns
    ``(best_expected_cut, theta)`` with ``theta`` a flat length-``2p`` vector.
    For p=1 the closed-form grid optimum seeds the search (globally reliable).
    """
    from scipy.optimize import minimize

    n = graph.number_of_nodes()
    cost = _cost_diagonal(graph, n)

    def neg(theta: np.ndarray) -> float:
        gammas, betas = theta[: theta.size // 2], theta[theta.size // 2 :]
        return -expectation_from_cost(cost, n, gammas, betas)

    def polish(seed: np.ndarray) -> np.ndarray:
        res = minimize(
            neg,
            seed,
            method="Nelder-Mead",
            options={"maxiter": 80 * seed.size, "xatol": 1e-4, "fatol": 1e-6},
        )
        return res.x

    # Seed at depth 1 from the exact closed-form grid, then grow by INTERP.
    _, g1, b1 = best_p1_on_grid(graph)
    cur = polish(np.array([g1, b1])) if p == 1 else np.array([g1, b1])
    for _ in range(2, p + 1):
        cur = polish(_interp_seed(cur))

    candidates = [cur]
    if rng is not None and restarts > 0:
        for _ in range(restarts):
            seed = np.concatenate(
                [rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)]
            )
            candidates.append(polish(seed))
    vals = [-neg(c) for c in candidates]
    j = int(np.argmax(vals))
    return float(vals[j]), np.asarray(candidates[j], dtype=float)
