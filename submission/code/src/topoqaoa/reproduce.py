"""Installed reproduction command for the topology-conditioned QAOA artifact."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _code_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(*args: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, *args], cwd=_code_dir(), env=env, check=True)


def _sync_submission() -> None:
    code = _code_dir()
    submission = code.parent
    (submission / "figures").mkdir(exist_ok=True)
    (submission / "tables").mkdir(exist_ok=True)
    # The manuscript reads figures/*.pdf and tables/*.tex at the submission top
    # level; mirror exactly what it references (PDF figures, and the LaTeX
    # tables/macros generated from results/summary.json).
    for path in (code / "figures").glob("*.pdf"):
        shutil.copy2(path, submission / "figures" / path.name)
    for src in (code / "results").glob("*.tex"):
        shutil.copy2(src, submission / "tables" / src.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the smoke-scale configuration (configs/smoke.yaml): seconds instead of minutes",
    )
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args(argv)
    config = "configs/smoke.yaml" if args.quick else args.config
    if not args.skip_run:
        _run("scripts/run.py", "--config", config, "--out", "results")
    _run("scripts/make_tables.py")
    _run("scripts/make_figures.py")
    _sync_submission()


if __name__ == "__main__":
    main()
