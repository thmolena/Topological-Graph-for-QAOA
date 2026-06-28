"""End-to-end experiment driver (multi-depth).

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. The protocol, repeated at each requested QAOA depth ``p``:

  label every graph with a near-optimal depth-p schedule (INTERP oracle)
  for each held-out family F:
      fit the learned policies on graphs NOT in F  (no leakage)
      for each test graph in F:
          for each policy: warm-start schedule -> budgeted refine -> exact ratio

Aggregates, per depth: mean approximation ratio and first-query ratio per policy,
the query-budget frontier, per-family held-out ratios, and -- the headline -- the
*paired* advantage of each learned policy over the spectral baseline. The
top-level ``advantage_vs_depth`` block tracks how that advantage grows with depth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from . import baselines, metrics, qaoa
from .config import Config
from .env import BudgetEnv, refine
from .seed import RunProvenance, set_seed
from .splits import build_dataset, family_holdout, has_leakage

POLICIES = baselines.POLICIES


def _make_policy(name: str, rng, gc, stk):
    if name == "random":
        return baselines.RandomPolicy(rng)
    if name == "spectral":
        return baselines.SpectralPolicy()
    if name == "topology":
        return baselines.TopologyPolicy()
    if name == "graph_conditioned":
        return gc
    if name == "stk":
        return stk
    raise ValueError(name)


def _eval_policy(policy, graph, p, budget, rng) -> BudgetEnv:
    env = BudgetEnv.from_graph(graph, p)
    env._budget = budget
    theta0 = policy.propose(graph, p)
    refine(env, theta0, budget, rng)
    return env


def _frontier_curve(histories: List[List[tuple]], budget: int) -> List[list]:
    """Average running-best ratio across graphs onto a 1..budget grid.

    Each grid point is ``[q, mean, ci95]`` with ``ci95`` the 95% confidence
    interval of the mean (1.96 s / sqrt(n)) across graphs.
    """
    grid = np.arange(1, budget + 1)
    stacked = []
    for hist in histories:
        if not hist:
            continue
        qs = np.array([q for q, _ in hist])
        bs = np.array([b for _, b in hist])
        curve = np.interp(grid, qs, bs, left=bs[0], right=bs[-1])
        stacked.append(curve)
    if not stacked:
        return []
    arr = np.vstack(stacked)
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    ci95 = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return [
        [int(q), round(float(m), 6), round(float(c), 6)]
        for q, m, c in zip(grid, mean, ci95)
    ]


def _paired_advantage(a: List[float], b: List[float]) -> Dict[str, float]:
    """Paired mean difference ``a - b`` with a 95% CI and significance flag."""
    da = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = da.size
    mean = float(da.mean()) if n else 0.0
    std = float(da.std(ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "delta_mean": round(mean, 6),
        "delta_ci95": round(ci95, 6),
        "frac_test_graphs_improved": round(float(np.mean(da > 1e-9)), 4) if n else 0.0,
        "significant_positive": bool(mean - ci95 > 0),
        "n": int(n),
    }


def _run_depth(cfg: Config, data, p: int) -> Dict:
    """Label, evaluate and aggregate the full benchmark at one QAOA depth."""
    # 1) Near-optimal schedule labels for every graph (the transfer targets).
    label_rng = np.random.default_rng(cfg.seed + 7919 * p)
    labels: List[np.ndarray] = []
    opt_ratios: List[float] = []
    for inst in data:
        env = BudgetEnv.from_graph(inst.graph, p)  # reuse for max_cut
        val, theta = qaoa.optimize_schedule(
            inst.graph, p, label_rng, restarts=cfg.label_restarts
        )
        labels.append(theta)
        opt_ratios.append(val / env.max_cut_value)

    eval_rng = np.random.default_rng(cfg.seed + 31 * p + 1)

    ratios: Dict[str, List[float]] = {pol: [] for pol in POLICIES}
    first_q: Dict[str, List[float]] = {pol: [] for pol in POLICIES}
    qtt: Dict[str, List[int]] = {pol: [] for pol in POLICIES}
    histories: Dict[str, List[List[tuple]]] = {pol: [] for pol in POLICIES}
    by_family: Dict[str, Dict[str, List[float]]] = {}
    leakage_clean = True

    index = {id(inst): i for i, inst in enumerate(data)}
    for held in cfg.families:
        split = family_holdout(data, held)
        if not split["test"] or not split["train"]:
            continue
        if has_leakage(split["train"], split["test"]):
            leakage_clean = False
        train_graphs = [d.graph for d in split["train"]]
        train_labels = [labels[index[id(d)]] for d in split["train"]]
        gc = baselines.GraphConditionedPolicy().fit(train_graphs, train_labels)
        stk = baselines.STKPolicy(r=cfg.stk_r, ridge=cfg.stk_ridge).fit(
            train_graphs, train_labels
        )
        by_family.setdefault(held, {})
        for inst in split["test"]:
            for pol in POLICIES:
                policy = _make_policy(pol, eval_rng, gc, stk)
                env = _eval_policy(policy, inst.graph, p, cfg.budget, eval_rng)
                ratios[pol].append(env.best_ratio)
                first_q[pol].append(env.history[0][1] if env.history else 0.0)
                histories[pol].append(env.history)
                qtt[pol].append(metrics.queries_to_target(env.history, cfg.target_ratio))
                by_family[held].setdefault(pol, []).append(env.best_ratio)

    out: Dict = {"policies": {}, "frontier": {}, "by_family": {}}
    for pol in POLICIES:
        frontier = _frontier_curve(histories[pol], cfg.budget)
        out["policies"][pol] = {
            "approx_ratio": metrics.summarize(ratios[pol]),
            "first_query_ratio": round(frontier[0][1], 6) if frontier else 0.0,
            "queries_to_target_mean": round(
                float(
                    np.mean([q for q in qtt[pol] if q > 0])
                    if any(q > 0 for q in qtt[pol])
                    else cfg.budget
                ),
                3,
            ),
            "target_hit_rate": round(float(np.mean([q > 0 for q in qtt[pol]])), 4),
        }
        out["frontier"][pol] = frontier
    for fam, pols in by_family.items():
        out["by_family"][fam] = {pol: metrics.summarize(v) for pol, v in pols.items()}

    # Best non-learned ramp per graph (max of spectral/topology) is the toughest
    # baseline; we report the learned advantage against it as well as spectral.
    best_ramp = [max(s, t) for s, t in zip(ratios["spectral"], ratios["topology"])]
    best_ramp_fq = [max(s, t) for s, t in zip(first_q["spectral"], first_q["topology"])]
    out["advantage"] = {
        "stk_vs_spectral": _paired_advantage(ratios["stk"], ratios["spectral"]),
        "stk_vs_best_ramp": _paired_advantage(ratios["stk"], best_ramp),
        "stk_vs_best_ramp_first_query": _paired_advantage(first_q["stk"], best_ramp_fq),
        "graph_conditioned_vs_spectral": _paired_advantage(
            ratios["graph_conditioned"], ratios["spectral"]
        ),
        "stk_vs_graph_conditioned": _paired_advantage(
            ratios["stk"], ratios["graph_conditioned"]
        ),
        "stk_vs_random": _paired_advantage(ratios["stk"], ratios["random"]),
    }
    out["oracle_ratio_mean"] = round(float(np.mean(opt_ratios)), 6)
    out["leakage_clean"] = leakage_clean
    return out


def run(cfg: Config, out_dir: Path) -> Dict:
    prov = RunProvenance(seed=cfg.seed)
    rng = set_seed(cfg.seed)
    data = build_dataset(cfg.families, cfg.n_per_family, cfg.sizes, rng)

    by_depth: Dict[str, Dict] = {}
    leakage_clean = True
    for p in cfg.depths:
        res = _run_depth(cfg, data, p)
        by_depth[str(p)] = res
        leakage_clean = leakage_clean and res["leakage_clean"]

    advantage_vs_depth = []
    for p in cfg.depths:
        d = by_depth[str(p)]
        advantage_vs_depth.append(
            {
                "p": p,
                "stk_mean": d["policies"]["stk"]["approx_ratio"]["mean"],
                "stk_ci95": d["policies"]["stk"]["approx_ratio"]["ci95"],
                "stk_first_query": d["policies"]["stk"]["first_query_ratio"],
                "graph_conditioned_mean": d["policies"]["graph_conditioned"][
                    "approx_ratio"
                ]["mean"],
                "spectral_mean": d["policies"]["spectral"]["approx_ratio"]["mean"],
                "spectral_ci95": d["policies"]["spectral"]["approx_ratio"]["ci95"],
                "topology_mean": d["policies"]["topology"]["approx_ratio"]["mean"],
                "random_mean": d["policies"]["random"]["approx_ratio"]["mean"],
                "oracle_mean": d["oracle_ratio_mean"],
                "delta_stk_vs_spectral": d["advantage"]["stk_vs_spectral"]["delta_mean"],
                "delta_ci95": d["advantage"]["stk_vs_spectral"]["delta_ci95"],
                "delta_stk_vs_best_ramp": d["advantage"]["stk_vs_best_ramp"]["delta_mean"],
                "delta_best_ramp_ci95": d["advantage"]["stk_vs_best_ramp"]["delta_ci95"],
                "delta_stk_vs_best_ramp_first_query": d["advantage"][
                    "stk_vs_best_ramp_first_query"
                ]["delta_mean"],
                "delta_first_query_ci95": d["advantage"][
                    "stk_vs_best_ramp_first_query"
                ]["delta_ci95"],
                "significant_positive": d["advantage"]["stk_vs_best_ramp"][
                    "significant_positive"
                ],
            }
        )

    hd = str(cfg.headline_depth)
    head = by_depth[hd]
    summary: Dict = {
        "config": cfg.__dict__,
        "provenance": prov.finalize().to_dict(),
        "leakage_clean": leakage_clean,
        "depths": list(cfg.depths),
        "headline_depth": cfg.headline_depth,
        "by_depth": by_depth,
        "advantage_vs_depth": advantage_vs_depth,
        "headline": {
            "headline_depth": cfg.headline_depth,
            "stk_approx_ratio_mean": head["policies"]["stk"]["approx_ratio"]["mean"],
            "spectral_approx_ratio_mean": head["policies"]["spectral"]["approx_ratio"][
                "mean"
            ],
            "topology_approx_ratio_mean": head["policies"]["topology"]["approx_ratio"][
                "mean"
            ],
            "random_approx_ratio_mean": head["policies"]["random"]["approx_ratio"][
                "mean"
            ],
            "delta_stk_vs_spectral": head["advantage"]["stk_vs_spectral"]["delta_mean"],
            "delta_stk_vs_best_ramp": head["advantage"]["stk_vs_best_ramp"]["delta_mean"],
            "delta_best_ramp_ci95": head["advantage"]["stk_vs_best_ramp"]["delta_ci95"],
            "delta_stk_vs_best_ramp_first_query": head["advantage"][
                "stk_vs_best_ramp_first_query"
            ]["delta_mean"],
            "significant_positive": head["advantage"]["stk_vs_best_ramp"][
                "significant_positive"
            ],
            "n_graphs": len(data),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
