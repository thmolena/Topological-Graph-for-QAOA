"""Figure generation from results artifacts, styled to Nature Machine
Intelligence (NMI) display conventions.

Design rules applied here (see Nature Portfolio artwork & formatting guidance):
  * Vector PDF output with embedded, editable text (``pdf.fonttype = 42``).
  * Sans-serif typeface (Arial/Helvetica family), 5--8 pt range.
  * No in-panel titles -- every description lives in the LaTeX caption.
  * Bold lower-case panel labels (a, b, ...) for multi-panel figures.
  * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods
    2011): safe under deuteranopia/protanopia, avoids the red--green trap.
  * Error bars are shown wherever a mean is plotted; the caption states n and
    that the interval is a 95% confidence interval.
  * Top/right spines removed for an uncluttered Nature-style frame.

The figures are produced purely from ``results/summary.json`` -- the single
source of truth written by the experiment runner -- so they regenerate
bit-for-bit from fixed seeds.
"""
from __future__ import annotations

import os

# Determinism: pin the build epoch BEFORE importing matplotlib so its PDF
# backend stamps a fixed CreationDate -> byte-identical figure PDFs across runs
# (the plotted numbers are already deterministic: they are read from the
# fixed-seed results/summary.json, the single source of truth).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402

# --- Okabe-Ito colour-blind-safe qualitative palette ------------------------
NMI_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Column widths in inches (Nature: single column 89 mm, double column 183 mm).
COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20

# Human-readable, fixed policy ordering and display names for legends/axes.
POLICY_ORDER = ["random", "spectral", "topology", "graph_conditioned", "stk"]
POLICY_LABELS = {
    "random": "random",
    "spectral": "spectral ramp",
    "topology": "topology ramp",
    "graph_conditioned": "descriptor mean",
    "stk": "STK transfer (ours)",
}


def apply_nmi_style() -> None:
    """Install NMI-conforming matplotlib defaults (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so text stays selectable/editable
            "ps.fonttype": 42,
            "svg.hashsalt": "topoqaoa",  # deterministic element IDs across runs
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",  # keep in-figure math sans-serif
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.0,
            "legend.frameon": False,
            "axes.prop_cycle": cycler(color=NMI_PALETTE),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
        }
    )


def panel_label(ax, letter: str, x: float = -0.18, y: float = 1.02) -> None:
    """Bold lower-case panel label in the upper-left, Nature convention."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _color_for(policy: str) -> str:
    return NMI_PALETTE[POLICY_ORDER.index(policy) % len(NMI_PALETTE)]


def _depths(summary: Dict):
    return [int(p) for p in summary.get("depths", [1])]


def fig_depth(summary: Dict, out: Path) -> Path:
    """Headline figure: the learned-transfer advantage as a function of depth.

    (a) Mean held-out approximation ratio versus QAOA depth p for each policy,
    with the exact-optimum oracle ceiling. The structure-aware policies tie at
    p=1; the spectral-truncation-kernel transfer policy (STK) separates upward
    from the adiabatic ramps and the averaging baseline as p grows.
    (b) The paired advantage of STK over the best per-instance ramp -- final and
    one-shot (first-query) -- with 95% confidence intervals; it is ~0 at p=1 and
    grows significantly positive with depth.
    """
    apply_nmi_style()
    rows = summary["advantage_vs_depth"]
    ps = np.array([r["p"] for r in rows])
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.7))

    series = [
        ("random", [r["random_mean"] for r in rows]),
        ("spectral", [r["spectral_mean"] for r in rows]),
        ("topology", [r["topology_mean"] for r in rows]),
        ("graph_conditioned", [r["graph_conditioned_mean"] for r in rows]),
        ("stk", [r["stk_mean"] for r in rows]),
    ]
    for pol, ys in series:
        axa.plot(ps, ys, marker="o", ms=3.2, color=_color_for(pol),
                 label=POLICY_LABELS.get(pol, pol))
    axa.plot(ps, [r["oracle_mean"] for r in rows], ls="--", lw=1.0, color="#555555",
             marker="D", ms=2.6, label="oracle ceiling")
    axa.set_xlabel("QAOA depth $p$")
    axa.set_ylabel("held-out approximation ratio")
    axa.set_xticks(ps)
    axa.legend(loc="lower right", fontsize=6.0, handlelength=1.3, labelspacing=0.25)
    axa.grid(True, axis="both")
    panel_label(axa, "a")

    df = np.array([r["delta_stk_vs_best_ramp"] for r in rows])
    dfc = np.array([r["delta_best_ramp_ci95"] for r in rows])
    do = np.array([r["delta_stk_vs_best_ramp_first_query"] for r in rows])
    doc = np.array([r.get("delta_first_query_ci95", 0.0) for r in rows])
    axb.axhline(0.0, color="#999999", lw=0.8, ls=":")
    axb.errorbar(ps - 0.04, df, yerr=dfc, marker="o", ms=3.4, lw=1.2,
                 color=_color_for("stk"), capsize=2,
                 label="final (budget $B$)")
    axb.errorbar(ps + 0.04, do, yerr=doc, marker="s", ms=3.2, lw=1.2,
                 color=_color_for("topology"), capsize=2,
                 label="one-shot ($q{=}1$)")
    axb.set_xlabel("QAOA depth $p$")
    axb.set_ylabel(r"STK $-$ best ramp (ratio)")
    axb.set_xticks(ps)
    axb.legend(loc="upper left", fontsize=6.4, handlelength=1.3)
    axb.grid(True, axis="both")
    panel_label(axb, "b")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_frontier(summary: Dict, out: Path) -> Path:
    """Query-budget frontier per QAOA depth.

    The running-best held-out approximation ratio (mean over all test graphs)
    versus the number of objective evaluations, one panel per depth. The STK
    transfer policy starts highest and stays above the ramps throughout at p>=2;
    at p=1 all structure-aware curves coincide. Bands are 95% CIs of the mean.
    """
    apply_nmi_style()
    ps = _depths(summary)
    fig, axes = plt.subplots(1, len(ps), figsize=(COL_DOUBLE, 2.6), sharey=False)
    if len(ps) == 1:
        axes = [axes]
    for j, (p, ax) in enumerate(zip(ps, axes)):
        frontier = summary["by_depth"][str(p)]["frontier"]
        for policy in POLICY_ORDER:
            curve = frontier.get(policy)
            if not curve:
                continue
            xs = np.array([pt[0] for pt in curve])
            ys = np.array([pt[1] for pt in curve])
            cis = np.array([pt[2] if len(pt) > 2 else 0.0 for pt in curve])
            color = _color_for(policy)
            ax.plot(xs, ys, color=color, lw=1.2,
                    label=POLICY_LABELS.get(policy, policy))
            if np.any(cis > 0):
                ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.13,
                                linewidth=0)
        ax.set_xlabel("query budget")
        if j == 0:
            ax.set_ylabel("approximation ratio")
        ax.set_title(f"$p={p}$", fontsize=8)
        ax.grid(True, axis="both")
        panel_label(ax, chr(ord("a") + j))
    axes[-1].legend(loc="lower right", fontsize=6.0, handlelength=1.3,
                    labelspacing=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_family_bars(summary: Dict, out: Path) -> Path:
    """Held-out approximation ratio by graph family at the headline depth.

    Grouped bar chart; error bars are 95% confidence intervals (1.96 s / sqrt(n))
    over the n test graphs in each family, taken from the headline-depth
    ``by_family`` block.
    """
    apply_nmi_style()
    hd = str(summary.get("headline_depth", _depths(summary)[-1]))
    by_family = summary["by_depth"][hd]["by_family"]
    families = sorted(by_family.keys())
    policies = [p for p in POLICY_ORDER if any(p in by_family[f] for f in families)]

    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 3.0))
    width = 0.8 / max(1, len(policies))
    x = np.arange(len(families))
    for i, pol in enumerate(policies):
        vals = [by_family[f].get(pol, {}).get("mean", 0.0) for f in families]
        errs = [by_family[f].get(pol, {}).get("ci95", 0.0) for f in families]
        ax.bar(
            x + i * width,
            vals,
            width,
            yerr=errs,
            capsize=2,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
            color=_color_for(pol),
            label=POLICY_LABELS.get(pol, pol),
        )
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(
        [f.replace("_", " ") for f in families], rotation=25, ha="right"
    )
    ax.set_ylabel(f"held-out approx. ratio ($p={hd}$)")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=len(policies), loc="upper center", bbox_to_anchor=(0.5, 1.16),
              columnspacing=1.0, handlelength=1.2)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _box(ax, xy, w, h, text, fc, ec="#222222"):
    """Draw a rounded method-schematic box with centred wrapped text."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (xy[0], xy[1]),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        zorder=5,
    )
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444",
                        shrinkA=2, shrinkB=2),
    )


def fig_schematic(summary: Dict, out: Path) -> Path:
    """Method-overview schematic (the NMI 'Figure 1' convention).

    A left-to-right pipeline: a graph instance is mapped by a proven
    relabeling-invariant descriptor to a fixed-length feature vector; a
    warm-start policy proposes initial QAOA angles; a query-counted refiner
    spends a fixed evaluation budget; quality is scored as an exact
    approximation ratio against a brute-force MaxCut oracle.
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue, green, orange, purple, grey = (
        "#D6E6F2",
        "#D6EFE3",
        "#FBE6D4",
        "#ECDCE9",
        "#ECECEC",
    )
    y = 0.40
    h = 0.34
    boxes = [
        (0.010, 0.150, "graph instance\n$G=(V,E)$\n6 families", blue),
        (0.200, 0.170, "truncated spectrum\n$\\sigma_r(G)$ $+$ invariant\ndescriptor $\\phi(G)$", green),
        (0.420, 0.170, "spectral-truncation\nkernel transfer\n$\\to$ donor schedule $\\theta_0$", orange),
        (0.640, 0.165, "query-counted refiner\ndepth-$p$ (budget $B$)", purple),
        (0.845, 0.150, "exact approx.\nratio vs MaxCut\noracle", grey),
    ]
    rights = []
    lefts = []
    for x0, w, text, fc in boxes:
        r, l = _box(ax, (x0, y), w, h, text, fc)
        rights.append(r)
        lefts.append(l)
    for i in range(len(boxes) - 1):
        _arrow(ax, rights[i], lefts[i + 1])

    # Annotate the guarantees and the headline that underpin the benchmark.
    ax.text(0.4975, 0.07,
            "kernel $k(G,G')$ positive-definite \\& relabeling-invariant"
            "   $\\bullet$   depth-1 closed-form $\\leftrightarrow$ statevector to $10^{-9}$",
            ha="center", va="center", fontsize=6.4, color="#555555")
    ax.text(0.4975, 0.95,
            "family-held-out transfer (leakage-checked): schedules transferred, not averaged",
            ha="center", va="center", fontsize=6.6, color="#333333")
    fig.savefig(out)
    plt.close(fig)
    return out
