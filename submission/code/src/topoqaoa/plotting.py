"""Figure generation from results artifacts (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def fig_frontier(summary: Dict, out: Path) -> Path:
    """Approximation ratio vs query budget, per policy (the headline figure)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    frontier = summary.get("frontier", {})
    for policy, curve in frontier.items():
        xs = [pt[0] for pt in curve]
        ys = [pt[1] for pt in curve]
        ax.plot(xs, ys, marker="o", ms=3, label=policy)
    ax.set_xlabel("query budget (objective evaluations)")
    ax.set_ylabel("approximation ratio")
    ax.set_title("Budgeted QAOA: warm-start policies")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_family_bars(summary: Dict, out: Path) -> Path:
    """Held-out approximation ratio by graph family for each policy."""
    by_family = summary.get("by_family", {})
    families = sorted(by_family.keys())
    policies = sorted({p for fam in by_family.values() for p in fam})
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    import numpy as np

    width = 0.8 / max(1, len(policies))
    x = np.arange(len(families))
    for i, pol in enumerate(policies):
        vals = [by_family[f].get(pol, {}).get("mean", 0.0) for f in families]
        ax.bar(x + i * width, vals, width, label=pol)
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(families, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("held-out approx. ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
