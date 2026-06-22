"""Warm-start policies that propose initial QAOA angles for a graph.

All policies expose ``propose(graph) -> (gamma, beta)``. They differ only in how
much structure they exploit:

  RandomPolicy       angles drawn uniformly (no structure)
  SpectralPolicy     angle set from the Laplacian spectrum / density
  TopologyPolicy     fixed-angle heuristic interpolated by average degree
  GraphConditioned   learned map descriptor -> good angles (the contribution);
                     CPU reference is a descriptor regressor (a graph-invariant
                     surrogate for a message-passing GNN; an optional torch GNN
                     backend can be dropped in via ``set_backend``)
  RLPolicy           cross-entropy refinement seeded by GraphConditioned

The learned policies see *only* the relabeling-invariant descriptor vector, so
they are node-ordering invariant by construction.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np

from . import descriptors, qaoa


class RandomPolicy:
    name = "random"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def propose(self, graph: nx.Graph) -> Tuple[float, float]:
        return float(self.rng.uniform(0, np.pi)), float(self.rng.uniform(0, np.pi / 2))


class SpectralPolicy:
    name = "spectral"

    def propose(self, graph: nx.Graph) -> Tuple[float, float]:
        try:
            lam = np.sort(nx.laplacian_spectrum(graph))
            algebraic_conn = lam[1] if len(lam) > 1 else 1.0
        except Exception:
            algebraic_conn = 1.0
        gamma = float(np.clip(np.pi / (2.0 * (1.0 + algebraic_conn)), 0, np.pi))
        return gamma, float(np.pi / 8)


class TopologyPolicy:
    """Non-learned structural heuristic: fixed angles scaled by mean degree."""

    name = "topology"

    def propose(self, graph: nx.Graph) -> Tuple[float, float]:
        deg = np.mean([d for _, d in graph.degree()])
        gamma = float(np.arctan(1.0 / np.sqrt(max(1.0, deg))))  # shrinks with degree
        return gamma, float(np.pi / 8)


class GraphConditionedPolicy:
    """Learned descriptor -> angle map (this project's core contribution).

    Default backend is a k-nearest-neighbour regressor over standardized
    relabeling-invariant descriptors -- a deterministic, dependency-light
    surrogate for a message-passing GNN. ``fit`` builds the angle bank from
    *training* graphs only (held-out families never leak in).
    """

    name = "graph_conditioned"

    def __init__(self, k: int = 3):
        self.k = k
        self._X: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def fit(self, train_graphs: Sequence[nx.Graph]) -> "GraphConditionedPolicy":
        X, Y = [], []
        for g in train_graphs:
            _, gamma, beta = qaoa.best_p1_on_grid(g)
            X.append(descriptors.describe(g))
            Y.append([gamma, beta])
        self._X = np.array(X)
        self._Y = np.array(Y)
        self._mu = self._X.mean(axis=0)
        self._sd = self._X.std(axis=0) + 1e-8
        return self

    def propose(self, graph: nx.Graph) -> Tuple[float, float]:
        if self._X is None:
            raise RuntimeError("GraphConditionedPolicy must be fit() before use")
        z = (descriptors.describe(graph) - self._mu) / self._sd
        zb = (self._X - self._mu) / self._sd
        d = np.linalg.norm(zb - z, axis=1)
        idx = np.argsort(d)[: self.k]
        gamma, beta = self._Y[idx].mean(axis=0)
        return float(gamma), float(beta)


class RLPolicy:
    """Cross-entropy-method refinement seeded by a base warm start.

    Treats angle selection as a 1-step contextual bandit: sample a Gaussian over
    (gamma, beta), keep the elite fraction, re-fit. Each sample is one query, so
    this is directly comparable to the other policies under the budget env.
    """

    name = "rl"

    def __init__(self, base: GraphConditionedPolicy, rng: np.random.Generator,
                 pop: int = 8, elite: int = 3, iters: int = 4):
        self.base = base
        self.rng = rng
        self.pop, self.elite, self.iters = pop, elite, iters

    def optimize(self, graph: nx.Graph, env) -> Tuple[float, float, float]:
        mu = np.array(self.base.propose(graph))
        sigma = np.array([0.4, 0.2])
        best = (-1.0, mu[0], mu[1])
        for _ in range(self.iters):
            samples = self.rng.normal(mu, sigma, size=(self.pop, 2))
            scored = []
            for g, b in samples:
                if env.queries >= env_budget(env):
                    break
                scored.append((env.objective(float(g), float(b)), g, b))
            if not scored:
                break
            scored.sort(reverse=True, key=lambda t: t[0])
            if scored[0][0] > best[0]:
                best = scored[0]
            elites = np.array([[g, b] for _, g, b in scored[: self.elite]])
            mu, sigma = elites.mean(axis=0), elites.std(axis=0) + 1e-3
        return best


# Budget is stored on the env by the runner; helper keeps RLPolicy decoupled.
def env_budget(env) -> int:
    return getattr(env, "_budget", 10 ** 9)


def build_policies(rng: np.random.Generator) -> List[str]:
    return ["random", "spectral", "topology", "graph_conditioned", "rl"]
