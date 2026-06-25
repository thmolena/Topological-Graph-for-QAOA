"""General-depth QAOA engine: arbitrary-p objective, angle oracle, budget env.

This module generalizes the depth-one machinery in :mod:`qaoa` and :mod:`env`
to arbitrary depth ``p`` *without* disturbing the validated p=1 code path. It
provides:

* :class:`QAOAObjective` -- precomputes the (optionally weighted) cost diagonal
  once per graph and evaluates the exact statevector expectation for any depth.
  Precomputation is what makes repeated optimization affordable: the dominant
  ``O(|E| 2**n)`` cost is paid a single time, not per query.
* :func:`optimize_angles` -- multi-restart Nelder--Mead "oracle" that returns the
  best ``(gammas, betas)`` found at depth ``p``; used both to label training
  graphs for the learned policy and to report best-achievable ratios.
* :class:`DepthBudgetEnv` / :func:`refine_pd` -- a query-counted, ``2p``-angle
  coordinate-ascent refiner, the depth-p analogue of :func:`env.refine`, so that
  warm-start policies can be compared at matched query budgets at any depth.

Everything is numpy + scipy; no quantum dependency. Exact statevector cost is
``O(p n 2**n)`` per evaluation, fine for ``n <~ 16``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from .maxcut_exact import max_cut


# --------------------------------------------------------------------------- #
# Cost diagonal and mixer (weighted-aware)                                    #
# --------------------------------------------------------------------------- #
def cost_diagonal(graph: nx.Graph, weighted: bool = False) -> np.ndarray:
    """Diagonal of the MaxCut cost operator over all ``2**n`` basis states."""
    n = graph.number_of_nodes()
    x = np.arange(2 ** n, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1)
    bits = (x[:, None] >> shifts[None, :]) & 1  # qubit j = column j (MSB-first)
    cost = np.zeros(2 ** n)
    for u, v in graph.edges():
        w = float(graph[u][v].get("weight", 1.0)) if weighted else 1.0
        cost += w * (bits[:, int(u)] != bits[:, int(v)])
    return cost


def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = np.cos(beta), np.sin(beta)
    tensor = psi.reshape((2,) * n)
    for j in range(n):
        tensor = c * tensor - 1j * s * np.flip(tensor, axis=j)
    return tensor.reshape(-1)


class QAOAObjective:
    """Exact arbitrary-depth QAOA expected cut with a precomputed cost diagonal.

    The weighted MaxCut optimum is obtained by brute force here as well (the
    unweighted oracle in :mod:`maxcut_exact` ignores weights), so approximation
    ratios are exact ground truth in both the weighted and unweighted cases.
    """

    def __init__(self, graph: nx.Graph, weighted: bool = False):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.cost = cost_diagonal(graph, weighted)
        if weighted:
            self.max_cut_value = float(max(1e-9, self.cost.max()))
        else:
            mc, _ = max_cut(graph)
            self.max_cut_value = float(max(1, mc))

    def expect(self, gammas: Sequence[float], betas: Sequence[float]) -> float:
        n = self.n
        psi = np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=complex)
        for gamma, beta in zip(gammas, betas):
            psi = np.exp(-1j * gamma * self.cost) * psi
            psi = _apply_mixer(psi, beta, n)
        return float(np.sum(self.cost * np.abs(psi) ** 2))

    def ratio(self, gammas: Sequence[float], betas: Sequence[float]) -> float:
        return self.expect(gammas, betas) / self.max_cut_value


# --------------------------------------------------------------------------- #
# Angle oracle: multi-restart optimization at depth p                         #
# --------------------------------------------------------------------------- #
def optimize_angles(
    graph: nx.Graph,
    p: int,
    rng: np.random.Generator,
    n_restarts: int = 12,
    maxiter: int = 400,
    weighted: bool = False,
    obj: "QAOAObjective | None" = None,
) -> dict:
    """Best ``(gammas, betas)`` at depth ``p`` from multi-restart Nelder--Mead.

    Returns a dict with ``value``, ``ratio``, ``gammas`` (length p), ``betas``
    (length p), and ``n_restarts``. This is an unbudgeted "oracle": it is used
    to produce regression targets and best-achievable references, never as a
    warm-start policy competing under the query budget.
    """
    obj = obj or QAOAObjective(graph, weighted)

    def neg(x: np.ndarray) -> float:
        return -obj.expect(x[:p], x[p:])

    best_val = -np.inf
    best_x = None
    for _ in range(n_restarts):
        x0 = np.concatenate(
            [rng.uniform(0.0, np.pi, p), rng.uniform(0.0, np.pi / 2.0, p)]
        )
        res = minimize(
            neg, x0, method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-7},
        )
        val = -float(res.fun)
        if val > best_val:
            best_val, best_x = val, res.x.copy()
    gammas, betas = best_x[:p], best_x[p:]
    return {
        "value": best_val,
        "ratio": best_val / obj.max_cut_value,
        "gammas": gammas.tolist(),
        "betas": betas.tolist(),
        "n_restarts": n_restarts,
    }


# --------------------------------------------------------------------------- #
# INTERP angle finder (Zhou et al., PRX 2020) -- robust depth-p oracle and     #
# the INTERP competitor warm start in one procedure.                           #
# --------------------------------------------------------------------------- #
def _interp_extend(theta: np.ndarray) -> np.ndarray:
    """Linear INTERP interpolation: optimal length-p vector -> length-(p+1) init.

    ``theta^{(p+1)}_i = (i-1)/p * theta^{(p)}_{i-1} + (p-i+1)/p * theta^{(p)}_i``
    with the conventional boundary terms ``theta^{(p)}_0 = theta^{(p)}_{p+1} = 0``
    (Zhou et al., Phys. Rev. X 10, 021067 (2020)).
    """
    p = len(theta)
    padded = np.concatenate([[0.0], theta, [0.0]])  # indices 0..p+1
    out = np.zeros(p + 1)
    for i in range(1, p + 2):
        out[i - 1] = ((i - 1) / p) * padded[i - 1] + ((p - i + 1) / p) * padded[i]
    return out


def interp_optimize(
    graph: nx.Graph,
    P: int,
    rng: np.random.Generator,
    restarts_p1: int = 16,
    maxiter: int = 400,
    weighted: bool = False,
    obj: "QAOAObjective | None" = None,
) -> dict:
    """INTERP: optimize depth 1 from many restarts, then climb depth-by-depth,
    seeding each level by interpolating the level below. Returns a dict keyed by
    depth ``p -> {gammas, betas, value, ratio, init_gammas, init_betas}`` where
    the ``init_*`` are the pre-optimization interpolated angles (the warm start a
    pure-INTERP *policy* would propose at that depth).
    """
    obj = obj or QAOAObjective(graph, weighted)
    o1 = optimize_angles(graph, 1, rng, n_restarts=restarts_p1, maxiter=maxiter, obj=obj)
    gammas, betas = np.array(o1["gammas"]), np.array(o1["betas"])
    levels = {
        1: {"gammas": gammas.copy(), "betas": betas.copy(), "value": o1["value"],
            "ratio": o1["ratio"], "init_gammas": gammas.copy(), "init_betas": betas.copy()}
    }
    for p in range(1, P):
        ig, ib = _interp_extend(gammas), _interp_extend(betas)
        x0 = np.concatenate([ig, ib])

        def neg(x: np.ndarray, _p=p) -> float:
            return -obj.expect(x[: _p + 1], x[_p + 1:])

        res = minimize(neg, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-7})
        gammas, betas = res.x[: p + 1], res.x[p + 1:]
        val = -float(res.fun)
        levels[p + 1] = {"gammas": gammas.copy(), "betas": betas.copy(), "value": val,
                         "ratio": val / obj.max_cut_value, "init_gammas": ig, "init_betas": ib}
    return levels


def oracle_ratio(
    graph: nx.Graph, p: int, rng: np.random.Generator, weighted: bool = False,
    obj: "QAOAObjective | None" = None, random_restarts: int = 8,
) -> dict:
    """Best-achievable depth-p angles: the better of INTERP and random restarts.

    Combining the two guards against either getting stuck: INTERP exploits angle
    smoothness, random restarts catch the occasional INTERP miss.
    """
    obj = obj or QAOAObjective(graph, weighted)
    levels = interp_optimize(graph, p, rng, weighted=weighted, obj=obj)
    best = levels[p]
    rnd = optimize_angles(graph, p, rng, n_restarts=random_restarts, weighted=weighted, obj=obj)
    if rnd["value"] > best["value"]:
        best = {"gammas": np.array(rnd["gammas"]), "betas": np.array(rnd["betas"]),
                "value": rnd["value"], "ratio": rnd["ratio"]}
    return best


# --------------------------------------------------------------------------- #
# Query-counted depth-p environment and refiner                               #
# --------------------------------------------------------------------------- #
@dataclass
class DepthBudgetEnv:
    """Counts arbitrary-depth objective queries against a per-graph budget."""

    obj: QAOAObjective
    p: int
    queries: int = 0
    history: List[Tuple[int, float]] = field(default_factory=list)
    _best: float = 0.0

    @classmethod
    def from_graph(cls, graph: nx.Graph, p: int, weighted: bool = False) -> "DepthBudgetEnv":
        return cls(obj=QAOAObjective(graph, weighted), p=p)

    def objective(self, gammas: Sequence[float], betas: Sequence[float]) -> float:
        self.queries += 1
        val = self.obj.expect(gammas, betas)
        ratio = val / self.obj.max_cut_value
        if ratio > self._best:
            self._best = ratio
        self.history.append((self.queries, self._best))
        return val

    @property
    def best_ratio(self) -> float:
        return self._best


def refine_pd(
    env: DepthBudgetEnv,
    init_gammas: Sequence[float],
    init_betas: Sequence[float],
    budget: int,
    step0: float = 0.3,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Coordinate-ascent over ``2p`` angles from a warm start, capped at budget.

    Direct generalization of :func:`env.refine` from 2 to ``2p`` coordinates:
    each axis is probed in +/- directions, the step halves when no axis
    improves, and every objective call spends one query.
    """
    p = env.p
    g = np.array(init_gammas, dtype=float)
    b = np.array(init_betas, dtype=float)
    best_val = env.objective(g, b)
    step = step0
    while env.queries < budget and step > 1e-3:
        improved = False
        for axis in range(p):
            for arr in (g, b):
                for sign in (+1.0, -1.0):
                    if env.queries >= budget:
                        break
                    trial = arr.copy()
                    trial[axis] += sign * step
                    cg, cb = (trial, b) if arr is g else (g, trial)
                    val = env.objective(cg, cb)
                    if val > best_val:
                        best_val = val
                        if arr is g:
                            g = trial
                        else:
                            b = trial
                        improved = True
        if not improved:
            step *= 0.5
    return best_val, g, b
