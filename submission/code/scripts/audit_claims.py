#!/usr/bin/env python
"""Readiness-gate audit: traceable numbers, required phrase, no forbidden claims.

Exits non-zero if any check fails, so it can gate CI / `make audit`.
"""
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from topoqaoa import audit


def main() -> int:
    problems = []

    summary_path = Path("results/summary.json")
    if not summary_path.exists():
        print("FAIL: results/summary.json missing -- run `make demo` first")
        return 1
    summary = json.loads(summary_path.read_text())
    flat = {**summary["headline"]}

    # README must carry the mandated phrase and avoid forbidden claims.
    readme = Path("../README.md")
    if readme.exists():
        text = readme.read_text()
        problems += audit.audit_reproducibility_phrase(text)
        problems += audit.audit_forbidden_claims(text)

    # Every headline macro must be present (traceable to this run).
    for key in ("stk_approx_ratio_mean", "delta_stk_vs_best_ramp",
                "delta_stk_vs_best_ramp_first_query", "n_graphs"):
        if key not in flat:
            problems.append(f"headline macro {key!r} not generated")

    # The reported depth-dependent finding must hold in the artifact: STK ties
    # the ramp at p=1 (no significant advantage) and significantly beats it at
    # the headline depth. This guards against silently shipping a run in which
    # the claim does not actually hold.
    adv = {row["p"]: row for row in summary.get("advantage_vs_depth", [])}
    if 1 in adv and adv[1]["significant_positive"]:
        problems.append("claim mismatch: STK significantly beats ramp at p=1 (should tie)")
    hd = summary.get("headline_depth")
    if hd in adv and not adv[hd]["significant_positive"]:
        problems.append(f"claim mismatch: STK does not significantly beat ramp at p={hd}")

    if not summary.get("leakage_clean", False):
        problems.append("family-holdout leakage detected")

    if problems:
        print("AUDIT FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("AUDIT PASSED: all numbers traceable, phrase present, no forbidden claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
