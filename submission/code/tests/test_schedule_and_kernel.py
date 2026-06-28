"""Depth-p schedule objective, spectral-truncation kernel, and transfer.

These tests pin the three guarantees the multi-depth contribution relies on:
  * the cached depth-p statevector objective equals the reference evaluators
    (and, at p=1, the analytic closed form) to numerical precision;
  * the truncated Laplacian spectrum and the spectral-truncation kernel are
    relabeling invariant, and the kernel Gram matrix is symmetric positive
    semi-definite (a valid Mercer kernel);
  * spectral-truncation-kernel transfer is a deterministic, relabeling-invariant
    function of the test graph.
"""
import numpy as np
import pytest

from topoqaoa import descriptors, kernels, qaoa
from topoqaoa import graph_generators as gg
from topoqaoa.baselines import _ramp


@pytest.mark.parametrize("family", gg.FAMILIES)
def test_depth_p_objective_matches_statevector(family):
    rng = np.random.default_rng(5)
    for p in (1, 2, 3):
        g = gg.generate(family, 8, rng)
        theta = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        cached = qaoa.expectation_schedule(g, theta)
        gammas, betas = theta[:p], theta[p:]
        sv = qaoa.expectation_statevector(g, gammas, betas)
        assert abs(cached - sv) < 1e-9, (family, p)
        if p == 1:
            cf = qaoa.expectation_p1_closed_form(g, float(theta[0]), float(theta[1]))
            assert abs(cached - cf) < 1e-9, (family, "p1 closed form")


def test_truncated_spectrum_relabeling_invariant():
    rng = np.random.default_rng(2)
    for family in gg.FAMILIES:
        g = gg.generate(family, 10, rng)
        s0 = kernels.truncated_spectrum(g, 6)
        for _ in range(3):
            gp = gg.relabel(g, rng)
            assert np.allclose(s0, kernels.truncated_spectrum(gp, 6), atol=1e-8), family


def test_stk_kernel_is_symmetric_psd():
    rng = np.random.default_rng(0)
    graphs = [gg.generate(f, 9, rng) for f in gg.FAMILIES for _ in range(2)]
    sched = [np.zeros(4) for _ in graphs]  # labels irrelevant for the Gram test
    reg = kernels.SpectralTruncationKernelTransfer(r=6).fit(graphs, sched)
    zs, zd = reg._zs_train, reg._zd_train
    K = reg._gram(zs, zs, zd, zd)
    assert np.allclose(K, K.T, atol=1e-10)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > -1e-8  # positive semi-definite


def test_stk_transfer_invariant_and_deterministic():
    rng = np.random.default_rng(1)
    train = [gg.generate(f, 9, rng) for f in gg.FAMILIES for _ in range(3)]
    sched = [
        np.concatenate([rng.uniform(0, np.pi, 2), rng.uniform(0, np.pi / 2, 2)])
        for _ in train
    ]
    reg = kernels.SpectralTruncationKernelTransfer(r=6).fit(train, sched)
    g = gg.generate("watts_strogatz", 10, rng)
    p0 = reg.predict(g)
    assert np.allclose(p0, reg.predict(g))  # deterministic
    for _ in range(3):
        assert np.allclose(p0, reg.predict(gg.relabel(g, rng)), atol=1e-8)  # invariant


def test_optimizer_oracle_improves_with_depth():
    rng = np.random.default_rng(4)
    g = gg.generate("stochastic_block", 10, rng)
    v1, _ = qaoa.optimize_schedule(g, 1, rng, restarts=1)
    v2, _ = qaoa.optimize_schedule(g, 2, rng, restarts=1)
    # A deeper circuit can represent the shallower one, so the optimum cannot
    # decrease (allowing a small optimizer slack).
    assert v2 >= v1 - 1e-6
