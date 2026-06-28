#!/usr/bin/env python
"""Run the QAOA warm-start experiment and write results/summary.json."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from topoqaoa.config import Config
from topoqaoa.runner import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    summary = run(cfg, Path(args.out))
    h = summary["headline"]
    print(f"[{cfg.name}] graphs={h['n_graphs']}  depths={summary['depths']}  "
          f"runtime={summary['provenance']['runtime_sec']}s")
    print(f"  headline depth p={h['headline_depth']}: STK={h['stk_approx_ratio_mean']:.4f}  "
          f"best-ramp delta(final)={h['delta_stk_vs_best_ramp']:+.4f}+-{h['delta_best_ramp_ci95']:.4f}  "
          f"delta(one-shot)={h['delta_stk_vs_best_ramp_first_query']:+.4f}  "
          f"significant={h['significant_positive']}")
    for row in summary["advantage_vs_depth"]:
        print(f"  p={row['p']}: STK={row['stk_mean']:.4f}(1q {row['stk_first_query']:.4f})  "
              f"spectral={row['spectral_mean']:.4f}  topo={row['topology_mean']:.4f}  "
              f"gc(avg)={row['graph_conditioned_mean']:.4f}  oracle={row['oracle_mean']:.4f}  "
              f"| STK-bestramp final={row['delta_stk_vs_best_ramp']:+.4f}+-{row['delta_best_ramp_ci95']:.4f} "
              f"1q={row['delta_stk_vs_best_ramp_first_query']:+.4f}  sig={row['significant_positive']}")


if __name__ == "__main__":
    main()
