"""End-to-end experiment driver.

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. The protocol:

  for each held-out family F:
      fit GraphConditioned on graphs NOT in F  (no leakage)
      for each test graph in F:
          for each policy: warm-start -> budgeted refine -> record exact ratio

Aggregates: mean approximation ratio per policy, mean queries-to-target, the
query-budget frontier curve, and per-family held-out ratios. Headline "query
efficiency" = (mean queries-to-target of random) / (that of graph_conditioned).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from . import baselines, metrics
from .config import Config
from .env import BudgetEnv, refine
from .seed import RunProvenance, set_seed
from .splits import build_dataset, family_holdout, has_leakage

POLICIES = ["random", "spectral", "topology", "graph_conditioned", "rl"]


def _eval_policy(name, graph, gcp, budget, rng) -> BudgetEnv:
    env = BudgetEnv.from_graph(graph)
    env._budget = budget
    if name == "random":
        g0, b0 = baselines.RandomPolicy(rng).propose(graph)
        refine(env, g0, b0, budget, rng)
    elif name == "spectral":
        g0, b0 = baselines.SpectralPolicy().propose(graph)
        refine(env, g0, b0, budget, rng)
    elif name == "topology":
        g0, b0 = baselines.TopologyPolicy().propose(graph)
        refine(env, g0, b0, budget, rng)
    elif name == "graph_conditioned":
        g0, b0 = gcp.propose(graph)
        refine(env, g0, b0, budget, rng)
    elif name == "rl":
        baselines.RLPolicy(gcp, rng).optimize(graph, env)
    else:
        raise ValueError(name)
    return env


def _frontier_curve(histories: List[List[tuple]], budget: int) -> List[list]:
    """Average running-best ratio across graphs onto a 1..budget grid.

    Each grid point is returned as ``[q, mean, ci95]`` where ``ci95`` is the
    95% confidence interval of the mean (1.96 * s / sqrt(n)) across graphs, so
    line plots can carry an explicit uncertainty band (Nature requirement).
    """
    grid = np.arange(1, budget + 1)
    stacked = []
    for hist in histories:
        if not hist:
            continue
        qs = np.array([q for q, _ in hist])
        bs = np.array([b for _, b in hist])
        # carry-forward best at each grid query
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


def run(cfg: Config, out_dir: Path) -> Dict:
    prov = RunProvenance(seed=cfg.seed)
    rng = set_seed(cfg.seed)
    data = build_dataset(cfg.families, cfg.n_per_family, cfg.sizes, rng)

    ratios: Dict[str, List[float]] = {p: [] for p in POLICIES}
    qtt: Dict[str, List[int]] = {p: [] for p in POLICIES}
    histories: Dict[str, List[List[tuple]]] = {p: [] for p in POLICIES}
    by_family: Dict[str, Dict[str, List[float]]] = {}
    leakage_clean = True

    for held in cfg.families:
        split = family_holdout(data, held)
        if not split["test"] or not split["train"]:
            continue
        if has_leakage(split["train"], split["test"]):
            leakage_clean = False
        gcp = baselines.GraphConditionedPolicy().fit([d.graph for d in split["train"]])
        by_family.setdefault(held, {})
        for inst in split["test"]:
            for pol in POLICIES:
                env = _eval_policy(pol, inst.graph, gcp, cfg.budget, rng)
                ratios[pol].append(env.best_ratio)
                histories[pol].append(env.history)
                qtt[pol].append(metrics.queries_to_target(env.history, cfg.target_ratio))
                by_family[held].setdefault(pol, []).append(env.best_ratio)

    # Aggregate
    summary: Dict = {
        "config": cfg.__dict__,
        "provenance": prov.finalize().to_dict(),
        "leakage_clean": leakage_clean,
        "policies": {},
        "frontier": {},
        "by_family": {},
        "headline": {},
    }
    for pol in POLICIES:
        summary["policies"][pol] = {
            "approx_ratio": metrics.summarize(ratios[pol]),
            "queries_to_target_mean": round(
                float(np.mean([q for q in qtt[pol] if q > 0]) if any(q > 0 for q in qtt[pol]) else cfg.budget), 3
            ),
            "target_hit_rate": round(float(np.mean([q > 0 for q in qtt[pol]])), 4),
        }
        summary["frontier"][pol] = _frontier_curve(histories[pol], cfg.budget)
    for fam, pols in by_family.items():
        summary["by_family"][fam] = {p: metrics.summarize(v) for p, v in pols.items()}

    gc_q = summary["policies"]["graph_conditioned"]["queries_to_target_mean"]
    rnd_q = summary["policies"]["random"]["queries_to_target_mean"]
    summary["headline"] = {
        "gc_approx_ratio_mean": summary["policies"]["graph_conditioned"]["approx_ratio"]["mean"],
        "random_approx_ratio_mean": summary["policies"]["random"]["approx_ratio"]["mean"],
        "query_efficiency_vs_random": round(rnd_q / max(1e-9, gc_q), 3),
        "n_graphs": len(data),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
