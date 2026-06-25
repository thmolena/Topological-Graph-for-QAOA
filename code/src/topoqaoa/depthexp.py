"""Depth-p warm-start policies and the matched-budget head-to-head experiment.

This is the depth-generalized successor to :mod:`runner`. Each policy proposes
depth-``p`` warm-start angles ``(gammas, betas)`` for a test graph; the SAME
query-counted refiner (:func:`depth.refine_pd`) then spends a fixed budget, so
policies are compared fairly at matched cost. The scientific question is whether
the *topology-conditioned* learned policy beats a single transferred angle set
(Galda--Shaydulin) and the physics-motivated TQA schedule under that budget.

Policies
--------
random       uniform angles (floor)
tqa          Trotterized-quantum-annealing linear ramp; dt tuned on train
transfer     Galda--Shaydulin fixed-angle transfer: median oracle angles over
             the training graphs, identical for every test graph (NO conditioning)
interp       INTERP init (Zhou et al.) interpolated up from a transferred depth-1
             optimum (no per-test optimization)
graph_cond   learned: kNN over relabeling-invariant descriptors -> the depth-p
             oracle angles of the nearest training graphs (THE contribution)

Per-graph depth-``p`` oracle angles (used for transfer/graph_cond labels and as a
best-achievable reference) are computed once via INTERP and cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from . import descriptors
from .depth import (QAOAObjective, DepthBudgetEnv, refine_pd, interp_optimize,
                    _interp_extend)
from .splits import build_dataset, GraphInstance


def tqa_angles(p: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Trotterized quantum annealing linear ramp (Sack & Serbyn 2021).

    Midpoint discretization ``s_i = (i-1/2)/p`` so neither schedule endpoint
    collapses to zero: ``gamma_i = s_i dt``, ``beta_i = (1-s_i) dt``.
    """
    s = (np.arange(1, p + 1) - 0.5) / p
    return s * dt, (1.0 - s) * dt


@dataclass
class GraphRecord:
    inst: GraphInstance
    obj: QAOAObjective
    desc: np.ndarray
    oracle_g: np.ndarray   # depth-p oracle gammas
    oracle_b: np.ndarray   # depth-p oracle betas
    oracle_ratio: float


def build_records(data: List[GraphInstance], p: int, rng: np.random.Generator,
                  restarts_p1: int = 12) -> List[GraphRecord]:
    """Precompute objective, descriptor, and cached depth-p oracle angles once."""
    recs: List[GraphRecord] = []
    for inst in data:
        obj = QAOAObjective(inst.graph)
        lv = interp_optimize(inst.graph, p, rng, restarts_p1=restarts_p1, obj=obj)
        top = lv[p]
        recs.append(GraphRecord(
            inst=inst, obj=obj, desc=descriptors.describe(inst.graph),
            oracle_g=np.array(top["gammas"]), oracle_b=np.array(top["betas"]),
            oracle_ratio=float(top["ratio"]),
        ))
    return recs


def _tune_tqa_dt(train: List[GraphRecord], p: int,
                 dts: np.ndarray | None = None) -> float:
    """Pick the annealing time dt maximizing the mean warm-start ratio on train."""
    dts = np.linspace(0.2, 1.4, 13) if dts is None else dts
    best_dt, best_mean = float(dts[0]), -np.inf
    for dt in dts:
        g, b = tqa_angles(p, float(dt))
        m = float(np.mean([r.obj.ratio(g, b) for r in train]))
        if m > best_mean:
            best_mean, best_dt = m, float(dt)
    return best_dt


def _knn_angles(desc: np.ndarray, train: List[GraphRecord],
                mu: np.ndarray, sd: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    z = (desc - mu) / sd
    Z = np.array([(r.desc - mu) / sd for r in train])
    d = np.linalg.norm(Z - z, axis=1)
    idx = np.argsort(d)[:k]
    g = np.mean([train[i].oracle_g for i in idx], axis=0)
    b = np.mean([train[i].oracle_b for i in idx], axis=0)
    return g, b


def _propose(policy: str, rec: GraphRecord, p: int, ctx: dict,
             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if policy == "random":
        return rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)
    if policy == "tqa":
        return tqa_angles(p, ctx["tqa_dt"])
    if policy == "transfer":
        return ctx["transfer_g"].copy(), ctx["transfer_b"].copy()
    if policy == "interp":
        g, b = ctx["transfer_g1"].copy(), ctx["transfer_b1"].copy()
        for _ in range(p - 1):
            g, b = _interp_extend(g), _interp_extend(b)
        return g, b
    if policy == "graph_cond":
        return _knn_angles(rec.desc, ctx["train"], ctx["mu"], ctx["sd"], ctx["k"])
    raise ValueError(policy)


POLICIES = ["random", "tqa", "transfer", "interp", "graph_cond"]


def run_depth_experiment(
    families: List[str], n_per_family: int, sizes: List[int], p: int,
    budget: int, seed: int, restarts_p1: int = 12, k: int = 3,
) -> Dict:
    """Family-held-out, matched-budget depth-p head-to-head. Returns a summary
    dict with per-policy ratios, first-query ratios, the best-achievable oracle,
    and the paired (graph_cond - transfer) and (graph_cond - tqa) differences."""
    rng = np.random.default_rng(seed)
    data = build_dataset(families, n_per_family, sizes, rng)
    recs = build_records(data, p, rng, restarts_p1=restarts_p1)
    by_key = {id(r.inst): r for r in recs}

    ratios: Dict[str, List[float]] = {pol: [] for pol in POLICIES}
    firstq: Dict[str, List[float]] = {pol: [] for pol in POLICIES}
    oracle_ratios: List[float] = []
    paired_gc_transfer: List[float] = []
    paired_gc_tqa: List[float] = []

    for held in families:
        train = [r for r in recs if r.inst.family != held]
        test = [r for r in recs if r.inst.family == held]
        if not train or not test:
            continue
        # Transfer (Galda-Shaydulin): median oracle angles over training graphs.
        transfer_g = np.median([r.oracle_g for r in train], axis=0)
        transfer_b = np.median([r.oracle_b for r in train], axis=0)
        # Depth-1 transfer optimum (median of first-layer oracle angle) for INTERP.
        transfer_g1 = np.array([np.median([r.oracle_g[0] for r in train])])
        transfer_b1 = np.array([np.median([r.oracle_b[0] for r in train])])
        D = np.array([r.desc for r in train])
        mu, sd = D.mean(0), D.std(0) + 1e-8
        ctx = {"tqa_dt": _tune_tqa_dt(train, p), "transfer_g": transfer_g,
               "transfer_b": transfer_b, "transfer_g1": transfer_g1,
               "transfer_b1": transfer_b1, "train": train, "mu": mu, "sd": sd, "k": k}

        for rec in test:
            oracle_ratios.append(rec.oracle_ratio)
            per: Dict[str, float] = {}
            for pol in POLICIES:
                g0, b0 = _propose(pol, rec, p, ctx, rng)
                env = DepthBudgetEnv(obj=rec.obj, p=p)
                fq = rec.obj.ratio(g0, b0)
                refine_pd(env, g0, b0, budget)
                ratios[pol].append(env.best_ratio)
                firstq[pol].append(fq)
                per[pol] = env.best_ratio
            paired_gc_transfer.append(per["graph_cond"] - per["transfer"])
            paired_gc_tqa.append(per["graph_cond"] - per["tqa"])

    def summ(xs: List[float]) -> Dict[str, float]:
        a = np.array(xs)
        n = len(a)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if n > 1 else 0.0,
                "ci95": float(1.96 * a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0, "n": n}

    return {
        "p": p, "budget": budget, "seed": seed, "n_graphs": len(data),
        "policies": {pol: summ(ratios[pol]) for pol in POLICIES},
        "first_query": {pol: summ(firstq[pol]) for pol in POLICIES},
        "oracle": summ(oracle_ratios),
        "paired_gc_minus_transfer": summ(paired_gc_transfer),
        "paired_gc_minus_tqa": summ(paired_gc_tqa),
    }
