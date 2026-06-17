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
    print(f"[{cfg.name}] graphs={h['n_graphs']}  "
          f"graph_conditioned ratio={h['gc_approx_ratio_mean']:.4f}  "
          f"random ratio={h['random_approx_ratio_mean']:.4f}  "
          f"query-efficiency vs random={h['query_efficiency_vs_random']}x  "
          f"runtime={summary['provenance']['runtime_sec']}s")


if __name__ == "__main__":
    main()
