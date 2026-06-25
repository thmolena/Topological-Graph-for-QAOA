"""Installed reproduction command for the topology-conditioned QAOA artifact.

The ``topoqaoa-reproduce`` console entry point regenerates the experiment
summary, LaTeX/Markdown tables, and figure PDFs from a fixed configuration and
seed. The driver invokes the source-tree scripts (``scripts/run.py``,
``scripts/make_tables.py``, ``scripts/make_figures.py``) so that an installed
copy of the package reproduces the same artifacts as the repository Makefile.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _code_dir() -> Path:
    """Return the artifact root (the ``code/`` tree two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def _run(*args: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, *args], cwd=_code_dir(), env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/full.yaml",
        help="experiment configuration (default: configs/full.yaml; "
        "use configs/smoke.yaml for a fast laptop-scale check)",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="output directory for the experiment summary (default: results)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="regenerate tables and figures from an existing summary "
        "without rerunning the experiment",
    )
    args = parser.parse_args(argv)
    if not args.skip_run:
        _run("scripts/run.py", "--config", args.config, "--out", args.out)
    _run("scripts/make_tables.py")
    _run("scripts/make_figures.py")


if __name__ == "__main__":
    main()
