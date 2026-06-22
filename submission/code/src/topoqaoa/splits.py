"""Dataset construction and family-held-out splits.

The headline generalization claim is *transfer to unseen graph families*. To
test it honestly we train policies on a subset of families and evaluate on a
family that was never seen during fitting -- and we assert (in the test suite)
that no test graph appears in the training set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np

from . import graph_generators as gg


@dataclass
class GraphInstance:
    family: str
    n: int
    graph: nx.Graph
    key: str  # canonical fingerprint for leakage checks


def _fingerprint(family: str, g: nx.Graph) -> str:
    deg_seq = tuple(sorted(d for _, d in g.degree()))
    return f"{family}:{g.number_of_nodes()}:{g.number_of_edges()}:{hash(deg_seq) & 0xffffffff}"


def build_dataset(
    families: List[str], n_per_family: int, sizes: List[int], rng: np.random.Generator
) -> List[GraphInstance]:
    data: List[GraphInstance] = []
    for fam in families:
        for _ in range(n_per_family):
            n = int(rng.choice(sizes))
            g = gg.generate(fam, n, rng)
            data.append(GraphInstance(fam, g.number_of_nodes(), g, _fingerprint(fam, g)))
    return data


def family_holdout(
    data: List[GraphInstance], held_out: str
) -> Dict[str, List[GraphInstance]]:
    train = [d for d in data if d.family != held_out]
    test = [d for d in data if d.family == held_out]
    return {"train": train, "test": test}


def has_leakage(train: List[GraphInstance], test: List[GraphInstance]) -> bool:
    train_keys = {d.key for d in train}
    return any(d.key in train_keys for d in test)
