"""Budgeted QAOA parameter-search environment.

In practice each QAOA objective evaluation costs circuit executions / shots, so
the relevant question is not "what is the optimum" but "how good a cut can a
policy reach within a *fixed query budget*". This environment wraps the
closed-form depth-1 objective, counts every query, and exposes a local refiner
so that warm-start policies (which propose the initial angles) can be compared
at matched budgets. The reward is the exact approximation ratio against the
brute-force MaxCut oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import networkx as nx
import numpy as np

from . import qaoa
from .maxcut_exact import max_cut


@dataclass
class BudgetEnv:
    """Counts objective queries against a per-graph budget."""

    graph: nx.Graph
    max_cut_value: int
    queries: int = 0
    history: List[Tuple[int, float]] = field(default_factory=list)  # (query#, best ratio)
    _best: float = 0.0

    @classmethod
    def from_graph(cls, graph: nx.Graph) -> "BudgetEnv":
        mc, _ = max_cut(graph)
        return cls(graph=graph, max_cut_value=max(1, mc))

    def objective(self, gamma: float, beta: float) -> float:
        """One query: returns the QAOA expected cut and logs the running best."""
        self.queries += 1
        val = qaoa.expectation_p1_closed_form(self.graph, gamma, beta)
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
    init_gamma: float,
    init_beta: float,
    budget: int,
    rng: np.random.Generator,
    step0: float = 0.3,
) -> Tuple[float, float, float]:
    """Coordinate-ascent local search from a warm start, capped at ``budget``.

    Returns ``(best_value, gamma, beta)``. Each call to ``env.objective`` spends
    one query; the loop stops as soon as the budget is exhausted, so policies
    that start closer to a good basin reach higher ratios per query.
    """
    g, b = float(init_gamma), float(init_beta)
    best_val = env.objective(g, b)
    step = step0
    while env.queries < budget and step > 1e-3:
        improved = False
        for axis in (0, 1):
            for sign in (+1.0, -1.0):
                if env.queries >= budget:
                    break
                cg = g + sign * step if axis == 0 else g
                cb = b + sign * step if axis == 1 else b
                val = env.objective(cg, cb)
                if val > best_val:
                    best_val, g, b = val, cg, cb
                    improved = True
        if not improved:
            step *= 0.5
    return best_val, g, b
