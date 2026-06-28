"""Spectral-truncation graph kernel for QAOA warm-start regression.

This module is the methodological core of the package. To transfer a *full*
depth-p QAOA angle schedule across unseen graph families, we need a similarity
between graphs that (i) is invariant to node relabeling and (ii) captures the
low-frequency structure that controls MaxCut landscapes.

We obtain it by *spectral truncation* of the graph operator. Each graph is
mapped to the multiset of the ``r`` smallest non-zero eigenvalues of its
normalized Laplacian -- a finite, low-rank window of the operator spectrum, in
the spirit of spectral-truncation kernels for operators (Hashimoto et al., 2024)
adapted from C*-algebraic operators to the graph Laplacian. Because Laplacian
eigenvalues are similarity invariants, the truncated spectrum is identical for
isomorphic graphs (proved in the manuscript), so any kernel built from it is a
relabeling-invariant, positive-definite kernel on isomorphism classes.

The warm-start map is kernel ridge regression (KRR): the predicted schedule is a
kernel-weighted combination of the optimal schedules of the training graphs. The
kernel is a product of a truncated-spectrum RBF and an invariant-descriptor RBF,

    k(G, G') = exp(-||sigma_r(G) - sigma_r(G')||^2 / 2 ell_s^2)
             * exp(-||phi(G)    - phi(G')   ||^2 / 2 ell_d^2),

a product of two positive-definite kernels (hence positive definite) whose
bandwidths are set by the median heuristic on the training set. No quantum or
deep-learning dependency is required; everything is numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np

from . import descriptors


def truncated_spectrum(graph: nx.Graph, r: int) -> np.ndarray:
    """The ``r`` smallest non-zero normalized-Laplacian eigenvalues of ``graph``.

    This is the spectral-truncation feature map sigma_r(G). The zero eigenvalue
    (one per connected component) is dropped; if the graph has fewer than ``r``
    non-zero eigenvalues the vector is right-padded with its last entry so that
    every graph maps to a fixed length-``r`` vector. Relabeling invariant because
    the Laplacian spectrum is a similarity invariant.
    """
    try:
        spec = np.sort(np.real(nx.normalized_laplacian_spectrum(graph)))
    except Exception:
        spec = np.zeros(graph.number_of_nodes())
    nz = spec[spec > 1e-9]
    vec = nz[:r]
    if vec.size < r:
        pad = vec[-1] if vec.size > 0 else 0.0
        vec = np.concatenate([vec, np.full(r - vec.size, pad)])
    return vec.astype(float)


def _median_bandwidth(feats: np.ndarray) -> float:
    """Median pairwise Euclidean distance (the classic RBF bandwidth heuristic)."""
    n = feats.shape[0]
    if n < 2:
        return 1.0
    diffs = feats[:, None, :] - feats[None, :, :]
    d = np.sqrt(np.maximum(0.0, np.sum(diffs ** 2, axis=2)))
    iu = np.triu_indices(n, k=1)
    med = float(np.median(d[iu]))
    return med if med > 1e-8 else 1.0


@dataclass
class SpectralTruncationKernelTransfer:
    """Spectral-truncation-kernel transfer of QAOA angle schedules.

    The key empirical fact this class encodes is that optimal depth-p QAOA
    schedules must be *transferred*, not *averaged*: distinct graphs reach their
    optima in different, symmetry-related basins, so a kernel-ridge or k-NN mean
    of several schedules lands between basins and underperforms even a generic
    adiabatic ramp. We therefore predict the schedule of the single training
    graph that is *most similar* under the spectral-truncation kernel -- a real
    optimum copied wholesale -- which preserves basin consistency.

    The similarity is the product kernel

        k(G, G') = exp(-||sigma_r(G)-sigma_r(G')||^2 / 2 ell_s^2)   (truncated spectrum)
                 * exp(-||phi(G)   -phi(G')   ||^2 / 2 ell_d^2),    (invariant descriptor)

    a product of two positive-definite RBF kernels (hence positive definite) on
    standardized, relabeling-invariant features, with median-heuristic
    bandwidths. ``mode='krr'`` exposes the kernel-ridge mean as a documented
    ablation (the averaging that fails); ``mode='transfer'`` (default) returns
    the kernel-nearest donor's schedule.

    Parameters
    ----------
    r : int
        Spectral-truncation order (number of smallest non-zero normalized-
        Laplacian eigenvalues retained).
    ridge : float
        Tikhonov lambda used only by the ``krr`` ablation.
    use_descriptor : bool
        If True the kernel multiplies in the invariant-descriptor RBF; if False
        it is the pure truncated-spectrum kernel (ablation).
    mode : str
        ``'transfer'`` (kernel-nearest schedule) or ``'krr'`` (ridge mean).
    """

    r: int = 6
    ridge: float = 1e-2
    use_descriptor: bool = True
    mode: str = "transfer"

    def fit(
        self, graphs: Sequence[nx.Graph], schedules: Sequence[np.ndarray]
    ) -> "SpectralTruncationKernelTransfer":
        self._spec = np.array([truncated_spectrum(g, self.r) for g in graphs])
        self._desc = np.array([descriptors.describe(g) for g in graphs])
        # Standardize each feature block so the median heuristic is well scaled.
        self._spec_mu, self._spec_sd = self._spec.mean(0), self._spec.std(0) + 1e-8
        self._desc_mu, self._desc_sd = self._desc.mean(0), self._desc.std(0) + 1e-8
        zs = (self._spec - self._spec_mu) / self._spec_sd
        zd = (self._desc - self._desc_mu) / self._desc_sd
        self._ell_s = _median_bandwidth(zs)
        self._ell_d = _median_bandwidth(zd)
        self._Y = np.asarray(schedules, dtype=float)
        self._zs_train, self._zd_train = zs, zd
        if self.mode == "krr":
            K = self._gram(zs, zs, zd, zd)
            n = K.shape[0]
            self._alpha = np.linalg.solve(K + self.ridge * np.eye(n), self._Y)
        return self

    def _gram(self, zs_a, zs_b, zd_a, zd_b) -> np.ndarray:
        ds = (
            np.sum(zs_a ** 2, 1)[:, None]
            + np.sum(zs_b ** 2, 1)[None, :]
            - 2 * zs_a @ zs_b.T
        )
        k_spec = np.exp(-np.maximum(0.0, ds) / (2 * self._ell_s ** 2))
        if not self.use_descriptor:
            return k_spec
        dd = (
            np.sum(zd_a ** 2, 1)[:, None]
            + np.sum(zd_b ** 2, 1)[None, :]
            - 2 * zd_a @ zd_b.T
        )
        k_desc = np.exp(-np.maximum(0.0, dd) / (2 * self._ell_d ** 2))
        return k_spec * k_desc

    def kernel_to_train(self, graph: nx.Graph) -> np.ndarray:
        zs = ((truncated_spectrum(graph, self.r) - self._spec_mu) / self._spec_sd)[None, :]
        zd = ((descriptors.describe(graph) - self._desc_mu) / self._desc_sd)[None, :]
        return self._gram(zs, self._zs_train, zd, self._zd_train).ravel()  # (n_train,)

    def predict(self, graph: nx.Graph) -> np.ndarray:
        k = self.kernel_to_train(graph)
        if self.mode == "krr":
            return (k @ self._alpha).ravel()
        return self._Y[int(np.argmax(k))]  # transfer: kernel-nearest donor schedule


# Backwards-compatible alias.
SpectralTruncationKernelRegressor = SpectralTruncationKernelTransfer
