"""Warm-start policies that propose an initial depth-p QAOA angle schedule.

All policies expose ``propose(graph, p) -> theta`` where ``theta`` is a flat
length-``2p`` schedule ``[gamma_1..gamma_p, beta_1..beta_p]``. They differ only
in how much structure they exploit:

  RandomPolicy       schedule drawn uniformly (no structure)
  SpectralPolicy     annealing-ramp schedule anchored by the Laplacian spectrum
                     (Sack & Serbyn 2021); at p=1 it reduces to the one-line
                     spectral angle gamma = pi/(2(1+lambda_2)), beta = pi/8
  TopologyPolicy     annealing-ramp schedule anchored by the average degree
  GraphConditioned   learned k-NN map descriptor -> optimal schedule (a
                     dependency-light surrogate for a message-passing GNN)
  STKPolicy          learned spectral-truncation-kernel ridge map
                     graph -> optimal schedule (this package's core contribution)

The non-learned ramp policies can set only a single physical scale, so at depth
one they are near-optimal but at depth ``p`` they cannot adapt the 2p schedule to
the instance. The learned policies predict the *whole* schedule from invariant
topology, which is where the depth-``p`` advantage comes from. Every learned
policy sees only relabeling-invariant features, so it is node-ordering invariant
by construction.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np

from . import descriptors, kernels, qaoa


def _ramp(gamma_scale: float, p: int) -> np.ndarray:
    """Discretized adiabatic (annealing) schedule of depth ``p``.

    Cost angles ramp up and mixer angles ramp down across the layers, following
    the quantum-annealing initialization of Sack & Serbyn (2021). The midpoint
    convention makes the p=1 case reduce exactly to ``(gamma_scale, pi/8)``.
    """
    frac = (np.arange(1, p + 1) - 0.5) / p  # midpoints in (0, 1)
    gammas = frac * (2.0 * gamma_scale)
    betas = (1.0 - frac) * (np.pi / 4.0)
    return np.concatenate([gammas, betas])


class RandomPolicy:
    name = "random"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def propose(self, graph: nx.Graph, p: int) -> np.ndarray:
        return np.concatenate(
            [self.rng.uniform(0, np.pi, p), self.rng.uniform(0, np.pi / 2, p)]
        )


class SpectralPolicy:
    """Annealing ramp anchored by the algebraic connectivity (strong baseline)."""

    name = "spectral"

    def propose(self, graph: nx.Graph, p: int) -> np.ndarray:
        try:
            lam = np.sort(nx.laplacian_spectrum(graph))
            algebraic_conn = lam[1] if len(lam) > 1 else 1.0
        except Exception:
            algebraic_conn = 1.0
        gamma_scale = float(np.clip(np.pi / (2.0 * (1.0 + algebraic_conn)), 0, np.pi))
        return _ramp(gamma_scale, p)


class TopologyPolicy:
    """Non-learned structural heuristic: annealing ramp scaled by mean degree."""

    name = "topology"

    def propose(self, graph: nx.Graph, p: int) -> np.ndarray:
        deg = np.mean([d for _, d in graph.degree()])
        gamma_scale = float(np.arctan(1.0 / np.sqrt(max(1.0, deg))))
        return _ramp(gamma_scale, p)


class GraphConditionedPolicy:
    """Learned descriptor -> schedule map via k-nearest-neighbour regression.

    A deterministic, dependency-light surrogate for a message-passing GNN.
    ``fit`` builds the schedule bank from *training* graphs only (held-out
    families never leak in); ``propose`` averages the schedules of the ``k``
    nearest standardized descriptors.
    """

    name = "graph_conditioned"

    def __init__(self, k: int = 3):
        self.k = k

    def fit(
        self, train_graphs: Sequence[nx.Graph], schedules: Sequence[np.ndarray]
    ) -> "GraphConditionedPolicy":
        self._X = np.array([descriptors.describe(g) for g in train_graphs])
        self._Y = np.asarray(schedules, dtype=float)
        self._mu = self._X.mean(axis=0)
        self._sd = self._X.std(axis=0) + 1e-8
        return self

    def propose(self, graph: nx.Graph, p: int) -> np.ndarray:
        z = (descriptors.describe(graph) - self._mu) / self._sd
        zb = (self._X - self._mu) / self._sd
        d = np.linalg.norm(zb - z, axis=1)
        idx = np.argsort(d)[: self.k]
        return self._Y[idx].mean(axis=0)


class STKPolicy:
    """Spectral-truncation-kernel schedule transfer (the core contribution).

    Copies the optimized depth-p schedule of the training graph that is most
    similar under a relabeling-invariant, positive-definite kernel on the
    truncated graph Laplacian spectrum (see :mod:`topoqaoa.kernels`). Transfer of
    a single real optimum -- rather than averaging several -- preserves basin
    consistency, which is why this policy separates from both the adiabatic ramp
    and the averaging-based ``graph_conditioned`` baseline as depth grows.
    """

    name = "stk"

    def __init__(self, r: int = 6, ridge: float = 1e-2, use_descriptor: bool = True,
                 mode: str = "transfer"):
        self._reg = kernels.SpectralTruncationKernelTransfer(
            r=r, ridge=ridge, use_descriptor=use_descriptor, mode=mode
        )

    def fit(
        self, train_graphs: Sequence[nx.Graph], schedules: Sequence[np.ndarray]
    ) -> "STKPolicy":
        self._reg.fit(list(train_graphs), list(schedules))
        return self

    def propose(self, graph: nx.Graph, p: int) -> np.ndarray:
        return self._reg.predict(graph)


# Policy roster: random < {spectral, topology} < {graph_conditioned, stk}.
POLICIES = ["random", "spectral", "topology", "graph_conditioned", "stk"]
LEARNED = {"graph_conditioned", "stk"}


def build_policies(rng: np.random.Generator) -> List[str]:
    return list(POLICIES)
