# topoqaoa — code artifact

CPU reference implementation for **Topology-Conditioned QAOA Warm Starts: a query-counted,
relabeling-invariant MaxCut benchmark**. QAOA is evaluated with an exact numpy statevector
simulator and the analytic depth-1 MaxCut expression, cross-checked to a tolerance of 1e-9. The
topology-conditioned policy uses scikit-learn; an optional PyTorch backend may be substituted.
The artifact depends on no heavyweight quantum framework.

Under the matched-budget protocol the topology-conditioned warm start matches spectral
initialization and does not surpass it; see the headline table in the
[project README](../README.md).

## Install

From the Python Package Index:

```bash
pip install topoqaoa
```

For source-tree development with the test suite and packaging tools:

```bash
pip install -e ".[dev]"
```

A Conda environment is also provided:

```bash
conda env create -f environment.yml && conda activate topoqaoa
```

## Reproduce

The package installs the `topoqaoa-reproduce` console entry point, which deterministically
regenerates the tables and figures from a fixed configuration and seed:

```bash
topoqaoa-reproduce                              # reported scale (configs/full.yaml, seed 0)
topoqaoa-reproduce --config configs/smoke.yaml  # laptop-scale check (a few seconds)
topoqaoa-reproduce --skip-run                    # tables and figures from an existing summary
```

The equivalent Makefile targets operate on the source tree:

```bash
make test        # closed-form vs statevector to 1e-9, invariance, leakage, metrics
make demo        # smoke config (a few seconds) -> results/summary.json
make full-run    # reported-scale config (configs/full.yaml)
make tables      # results/main_results.{md,tex}
make figures     # figures/fig_frontier.pdf, figures/fig_family.pdf
make audit       # readiness gate: traceable numbers, required phrase, no forbidden claims
# or, one command:
bash scripts/reproduce_all.sh         # smoke
bash scripts/reproduce_all.sh full    # reported scale
```

> macOS: the Makefile and scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because a Conda runtime and a
> pip-installed PyTorch can each ship an OpenMP library. The setting affects loader behavior
> alone and leaves results unchanged.

## Artifacts regenerated

`topoqaoa-reproduce` and the Makefile targets write the following:

| Artifact | Produced by | Contents |
| --- | --- | --- |
| `results/summary.json` | `scripts/run.py` | Per-policy approximation ratio with 95% CIs, mean queries-to-target, target hit rate, the query-budget frontier curve, the per-family breakdown, and run provenance. The single source of truth for every table and figure. |
| `results/main_results.md` | `scripts/make_tables.py` | Markdown headline table. |
| `results/main_results.tex` | `scripts/make_tables.py` | LaTeX headline table. |
| `figures/fig_frontier.pdf` | `scripts/make_figures.py` | Approximation ratio against query budget, per policy. |
| `figures/fig_family.pdf` | `scripts/make_figures.py` | Per-family approximation ratio bars, per policy. |

## Determinism

Reproduction is deterministic given the configuration. The seed is fixed at `0` in both
`configs/smoke.yaml` and `configs/full.yaml` and is threaded through graph generation, the
descriptor policy, and the refiner via `seed.py`. The reported scale corresponds to
`configs/full.yaml`: six families, 20 instances per family, sizes n in {10, 12, 14, 16}, depth
1, query budget 40. The entry point sets `OMP_NUM_THREADS=1` to suppress thread-scheduling
nondeterminism in the linear-algebra backends. The committed `results/summary.json` records the
seed, platform, library versions, timestamp, runtime, and peak memory under its `provenance`
key.

## Pinned dependencies

Runtime dependencies are declared in `pyproject.toml` with bounded version ranges
(numpy>=1.24,<3; scipy>=1.10,<2; networkx>=3.0,<4; scikit-learn>=1.2,<2; matplotlib>=3.6,<4;
pyyaml>=6.0,<7). The pip-style pins are mirrored in `requirements.txt`, and a CPU-only Conda
specification is provided in `environment.yml`. The exact library versions used to produce the
committed artifacts are recorded under `provenance.versions` in `results/summary.json` (numpy
2.4.3, scipy 1.17.1, networkx 3.6.1, scikit-learn 1.8.0, matplotlib 3.10.8, Python 3.13.12).

## Layout

```text
src/topoqaoa/
  graph_generators.py  ER / 3-regular / BA / WS / grid / SBM families + relabeling utility
  maxcut_exact.py      brute-force MaxCut oracle (exact, n<=20)
  descriptors.py       relabeling-invariant topology descriptors
  qaoa.py              exact statevector + analytic depth-1 closed form
  env.py               query-budgeted refiner (counts every evaluation)
  baselines.py         random / spectral / topology / graph-conditioned / RL policies
  splits.py            family-held-out splits + leakage check
  metrics.py           approx-ratio / regret / queries-to-target / summaries
  runner.py            end-to-end protocol -> results/summary.json
  audit.py             forbidden-claims + traceable-number checks
  plotting.py          frontier + per-family figures
  reproduce.py         topoqaoa-reproduce console entry point
scripts/   run.py · make_tables.py · make_figures.py · audit_claims.py · reproduce_all.sh
configs/   smoke.yaml (demo) · full.yaml (reported)
tests/     exact-MaxCut · closed-form vs statevector · descriptor invariance · no-leakage · metrics
```

## Distribution name

This package is distributed as `topoqaoa`. An identically named copy of the package exists
elsewhere in the broader repository tree (under `papers/github/P1`). A single canonical source
should be published to PyPI under the name `topoqaoa`; the duplicate is not to be uploaded
separately.

## License

MIT. See [`LICENSE`](LICENSE).

---

All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.
