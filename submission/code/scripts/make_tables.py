#!/usr/bin/env python
"""Generate LaTeX tables from results/summary.json."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    tables = Path("results")
    tables.mkdir(exist_ok=True)

    rows = []
    for pol, m in summary["policies"].items():
        ar = m["approx_ratio"]
        rows.append((pol, ar["mean"], ar["ci95"], m["queries_to_target_mean"],
                     m["target_hit_rate"]))

    # LaTeX
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"policy & approx.\ ratio & $\pm$95\% & queries$\to$target & hit rate \\",
             r"\midrule"]
    for pol, mean, ci, q, hr in rows:
        lines.append(f"{pol.replace('_',' ')} & {mean:.4f} & {ci:.4f} & {q:.2f} & {hr:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "main_results.tex").write_text("\n".join(lines) + "\n")

    print("wrote results/main_results.tex")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
