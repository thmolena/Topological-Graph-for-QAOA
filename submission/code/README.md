# topoqaoa — code artifact

CPU reference implementation for **Topology-Guided Graph RL for QAOA Warm Starts**.
QAOA is evaluated with an exact numpy statevector simulator and the analytic depth-1
MaxCut formula; the topology-conditioned policy uses scikit-learn (an optional PyTorch
GNN backend can be dropped in). No heavyweight quantum dependency.

## Install
```bash
conda env create -f environment.yml && conda activate topoqaoa
export PYTHONPATH=src
```

## Reproduce
```bash
make test        # 15 tests, incl. closed-form vs statevector to 1e-9
make demo        # smoke config (~1 s) -> results/summary.json
make tables      # results/main_results.tex
make figures     # figures/fig_frontier.pdf, figures/fig_family.pdf
make audit       # readiness gate
make full-run    # reported-scale config (minutes)
# or, one command:
bash scripts/reproduce_all.sh         # smoke
bash scripts/reproduce_all.sh full    # reported scale
```
> macOS: the Makefile/scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because conda and
> pip-PyTorch can both ship an OpenMP runtime. No effect on results.

## Layout
```
src/topoqaoa/
  graph_generators.py  ER / 3-regular / BA / WS / grid / SBM families + relabeling
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
scripts/   run.py · make_tables.py · make_figures.py · audit_claims.py · reproduce_all.sh
configs/   smoke.yaml (demo) · full.yaml (reported)
tests/     exact-MaxCut · closed-form↔statevector · descriptor invariance · no-leakage · metrics
```

## What is computed
`results/summary.json` holds, per warm-start policy: the held-out approximation ratio with
95% CIs, mean queries-to-target, target hit rate, the query-budget frontier curve, and the
per-family breakdown. It is the single source of truth for every table, figure and macro.

All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.
A from-basics explanation of the graph, QAOA, and learning objects is in `src/topoqaoa/foundations.py`.
