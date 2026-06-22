"""Relabeling-invariant topology descriptors.

The descriptor vector is the only information a topology-guided policy is allowed
to see about a graph. Every feature is a *graph invariant*: it is computed from
multiset/spectral quantities that do not depend on node ordering, so two
isomorphic graphs map to an identical descriptor (verified in the test suite).

Feature blocks:
  degree     : mean, std, max, min, density
  motif      : triangle density, clustering coefficient, square (C4) density
  wl         : hashed Weisfeiler-Lehman color-refinement histogram (k rounds)
  cycle      : girth-proxy, fraction of nodes in any triangle
  laplacian  : smallest non-zero eigenvalue (algebraic connectivity), spectral
               gap, normalized-Laplacian spectrum moments
  connectivity: average shortest path proxy, assortativity
"""
from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from typing import Dict, List

import networkx as nx
import numpy as np

WL_ROUNDS = 2
WL_BINS = 8

FEATURE_NAMES: List[str] = [
    "deg_mean", "deg_std", "deg_max", "deg_min", "density",
    "triangle_density", "clustering", "c4_density",
    *[f"wl{r}_bin{b}" for r in range(WL_ROUNDS) for b in range(WL_BINS)],
    "tri_node_frac", "girth_inv",
    "algebraic_connectivity", "spectral_gap", "lap_mean", "lap_var",
    "assortativity", "avg_clustering_global",
]


def _degree_block(g: nx.Graph) -> List[float]:
    n = g.number_of_nodes()
    degs = np.array([d for _, d in g.degree()], dtype=float)
    density = (2.0 * g.number_of_edges()) / (n * (n - 1)) if n > 1 else 0.0
    return [degs.mean(), degs.std(), degs.max(), degs.min(), density]


def _motif_block(g: nx.Graph) -> List[float]:
    n = g.number_of_nodes()
    triangles = sum(nx.triangles(g).values()) / 3.0
    max_tri = n * (n - 1) * (n - 2) / 6.0 if n >= 3 else 1.0
    clustering = nx.average_clustering(g)
    # Count 4-cycles via trace identity on the adjacency matrix.
    A = nx.to_numpy_array(g)
    A2 = A @ A
    paths2 = np.sum(A2) - np.trace(A2)
    c4 = (np.trace(A2 @ A2) - 2 * paths2 - np.trace(A2)) / 8.0
    max_c4 = max(1.0, n * (n - 1) * (n - 2) * (n - 3) / 8.0) if n >= 4 else 1.0
    return [triangles / max_tri, clustering, max(0.0, c4) / max_c4]


def _wl_block(g: nx.Graph) -> List[float]:
    """Hashed Weisfeiler-Lehman color histogram, order-invariant by construction."""
    labels: Dict[int, str] = {v: str(g.degree(v)) for v in g.nodes()}
    out: List[float] = []
    for _ in range(WL_ROUNDS):
        new_labels: Dict[int, str] = {}
        for v in g.nodes():
            neigh = sorted(labels[u] for u in g.neighbors(v))
            sig = labels[v] + "|" + ",".join(neigh)
            new_labels[v] = hashlib.md5(sig.encode()).hexdigest()[:8]
        labels = new_labels
        hist = np.zeros(WL_BINS)
        for lab in labels.values():
            hist[int(lab, 16) % WL_BINS] += 1
        hist = hist / max(1.0, hist.sum())
        out.extend(hist.tolist())
    return out


def _cycle_block(g: nx.Graph) -> List[float]:
    n = g.number_of_nodes()
    tri = nx.triangles(g)
    tri_node_frac = sum(1 for v in g if tri[v] > 0) / max(1, n)
    try:
        girth = nx.girth(g) if hasattr(nx, "girth") else _girth_fallback(g)
    except Exception:
        girth = _girth_fallback(g)
    girth_inv = 1.0 / girth if girth and girth < float("inf") else 0.0
    return [tri_node_frac, girth_inv]


def _girth_fallback(g: nx.Graph) -> float:
    best = float("inf")
    for cycle in nx.minimum_cycle_basis(g):
        best = min(best, len(cycle))
    return best


def _laplacian_block(g: nx.Graph) -> List[float]:
    n = g.number_of_nodes()
    try:
        lap_spec = np.sort(nx.laplacian_spectrum(g))
    except Exception:
        lap_spec = np.zeros(n)
    algebraic_conn = float(lap_spec[1]) if len(lap_spec) > 1 else 0.0
    spectral_gap = float(lap_spec[-1] - lap_spec[-2]) if len(lap_spec) > 1 else 0.0
    norm_spec = nx.normalized_laplacian_spectrum(g)
    return [algebraic_conn, spectral_gap, float(np.mean(norm_spec)), float(np.var(norm_spec))]


def _connectivity_block(g: nx.Graph) -> List[float]:
    try:
        # Degree-homogeneous graphs (e.g. regular, grid) give 0/0 inside
        # networkx's correlation formula, raising a harmless RuntimeWarning and
        # returning NaN; we map NaN to 0.0, so suppressing the warning here does
        # not change any descriptor value.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            assort = nx.degree_assortativity_coefficient(g)
        assort = 0.0 if np.isnan(assort) else float(assort)
    except Exception:
        assort = 0.0
    return [assort, nx.average_clustering(g)]


def describe(g: nx.Graph) -> np.ndarray:
    """Return the fixed-length, relabeling-invariant descriptor vector."""
    feats: List[float] = []
    feats += _degree_block(g)
    feats += _motif_block(g)
    feats += _wl_block(g)
    feats += _cycle_block(g)
    feats += _laplacian_block(g)
    feats += _connectivity_block(g)
    vec = np.array(feats, dtype=float)
    vec[~np.isfinite(vec)] = 0.0
    assert len(vec) == len(FEATURE_NAMES), (len(vec), len(FEATURE_NAMES))
    return vec
