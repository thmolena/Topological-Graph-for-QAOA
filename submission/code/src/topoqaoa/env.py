"""Budgeted QAOA parameter-search environment (arbitrary depth p).

In practice each QAOA objective evaluation costs circuit executions / shots, so
the relevant question is not "what is the optimum" but "how good a cut can a
policy reach within a *fixed query budget*". This environment wraps the exact
statevector depth-p objective, counts every query, caches the cost diagonal
once per graph, and exposes a local refiner over the full ``2p``-dimensional
angle schedule so that warm-start policies (which propose the initial schedule)
can be compared at matched budgets. The reward is the exact approximation ratio
against the brute-force MaxCut oracle.

At depth one the statevector objective equals the analytic closed form to
floating-point precision (asserted in the test suite to ~1e-9), so the depth-1
results remain analytically cross-verified; for p >= 2 the statevector evaluator
is the unambiguous reference, and the angle schedule has 2p degrees of freedom
that a single spectral scalar cannot set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np

from . import qaoa
from .maxcut_exact import max_cut


@dataclass
class BudgetEnv:
    """Counts depth-p objective queries against a per-graph budget."""

    graph: nx.Graph
    max_cut_value: int
    p: int = 1
    cost: np.ndarray = field(default=None, repr=False)
    n: int = 0
    queries: int = 0
    history: List[Tuple[int, float]] = field(default_factory=list)  # (query#, best ratio)
    _best: float = 0.0
    _budget: int = 10 ** 9

    @classmethod
    def from_graph(cls, graph: nx.Graph, p: int = 1) -> "BudgetEnv":
        mc, _ = max_cut(graph)
        n = graph.number_of_nodes()
        return cls(
            graph=graph,
            max_cut_value=max(1, mc),
            p=p,
            cost=qaoa.cost_diagonal(graph),
            n=n,
        )

    def objective(self, theta: Sequence[float]) -> float:
        """One query: returns the depth-p QAOA expected cut, logs running best."""
        self.queries += 1
        theta = np.asarray(theta, dtype=float)
        gammas, betas = theta[: self.p], theta[self.p :]
        val = qaoa.expectation_from_cost(self.cost, self.n, gammas, betas)
        ratio = val / self.max_cut_value
        if ratio > self._best:
            self._best = ratio
        self.history.append((self.queries, self._best))
        return val

    @property
    def best_ratio(self) -> float:
        return self._best


def refine(
    env: BudgetEnv,
    init_theta: Sequence[float],
    budget: int,
    rng: np.random.Generator,
    step0: float = 0.3,
) -> Tuple[float, np.ndarray]:
    """Coordinate-ascent local search over the full schedule, capped at ``budget``.

    Returns ``(best_value, theta)``. Each call to ``env.objective`` spends one
    query; the loop stops as soon as the budget is exhausted, so policies that
    start closer to a good basin reach higher ratios per query. The dimension is
    ``2p``: at p=1 this is the classic (gamma, beta) refiner, and for p >= 2 the
    same local search must tune the whole schedule, which is exactly why a
    full-schedule warm start (rather than a single spectral angle) helps.
    """
    theta = np.asarray(init_theta, dtype=float).copy()
    best_val = env.objective(theta)
    step = step0
    dim = theta.size
    while env.queries < budget and step > 1e-3:
        improved = False
        for axis in range(dim):
            for sign in (+1.0, -1.0):
                if env.queries >= budget:
                    break
                cand = theta.copy()
                cand[axis] += sign * step
                val = env.objective(cand)
                if val > best_val:
                    best_val, theta = val, cand
                    improved = True
            if env.queries >= budget:
                break
        if not improved:
            step *= 0.5
    return best_val, theta
