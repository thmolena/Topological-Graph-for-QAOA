#!/usr/bin/env python
"""Generate all figures from results/summary.json."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from topoqaoa import plotting


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    out = Path("figures")
    out.mkdir(exist_ok=True)
    plotting.fig_schematic(summary, out / "fig_schematic.pdf")
    plotting.fig_depth(summary, out / "fig_depth.pdf")
    plotting.fig_frontier(summary, out / "fig_frontier.pdf")
    plotting.fig_family_bars(summary, out / "fig_family.pdf")
    # PNG copies of the two headline figures for the web landing page (index.html).
    plotting.fig_depth(summary, out / "fig_depth.png")
    plotting.fig_frontier(summary, out / "fig_frontier.png")
    print(f"wrote {out}/fig_schematic.pdf, {out}/fig_depth.pdf, "
          f"{out}/fig_frontier.pdf, {out}/fig_family.pdf (+ headline PNGs)")


if __name__ == "__main__":
    main()
