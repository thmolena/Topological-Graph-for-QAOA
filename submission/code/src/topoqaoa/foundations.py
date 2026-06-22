"""From-scratch foundations for topology-conditioned QAOA warm starts."""

from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "MaxCut and QAOA",
        "A graph instance has a brute-force MaxCut optimum at the small sizes "
        "used here.  qaoa.py evaluates depth-one QAOA both by an analytic "
        "per-edge expression and by an exact statevector cross-check, so the "
        "objective implementation is auditable.",
    ),
    (
        "Warm-start problem",
        "The expensive part of variational QAOA is searching angles.  A warm "
        "start maps graph features to initial angles before a query-counted "
        "refiner spends the fixed evaluation budget.",
    ),
    (
        "Invariant graph descriptors",
        "descriptors.py builds fixed-length features from degree statistics, "
        "motifs, Weisfeiler-Lehman hashes, cycles, Laplacian spectrum and "
        "connectivity.  Tests verify invariance under node relabeling so the "
        "learner cannot exploit graph-label leakage.",
    ),
    (
        "Leak-free transfer",
        "splits.py holds out whole graph families.  A method trained on five "
        "families must transfer to a sixth unseen family, making family identity "
        "rather than graph relabeling the real generalization challenge.",
    ),
    (
        "Baselines",
        "baselines.py compares random, spectral, topology heuristic, learned "
        "descriptor-conditioned and cross-entropy policies.  The reported "
        "finding is conservative: structure helps versus random, but learned "
        "conditioning matches rather than beats the spectral prior at tested "
        "scale.",
    ),
    (
        "Metrics and figures",
        "metrics.py reports approximation ratio, query-to-target and hit rate.  "
        "The main figures are the method schematic, budget frontier and family "
        "bar comparison, all generated from summary.json.",
    ),
    (
        "Reproduction path",
        "Run scripts/reproduce_all.sh full or the topoqaoa-reproduce entry "
        "point.  The scripts regenerate summary.json, main_results.tex and the "
        "vector PDF figures used by main.tex.",
    ),
)


def iter_foundations() -> tuple[tuple[str, str], ...]:
    return FOUNDATION_SECTIONS


def print_foundations() -> None:
    for index, (heading, body) in enumerate(FOUNDATION_SECTIONS, start=1):
        print(f"{index}. {heading}\n{body}\n")


if __name__ == "__main__":
    print_foundations()
