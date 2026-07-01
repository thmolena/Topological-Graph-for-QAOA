# specops-stk

Reference implementation for **"Spectral-truncation graph kernels for QAOA warm
starts: topology-conditioned schedule transfer beyond depth one"** (Molena
Huynh, 2026), part of the *spectral-truncation operators* program.

The package introduces the **spectral-truncation kernel (STK)**: a
relabeling-invariant, positive-definite graph kernel that represents each graph
by a finite low-frequency window of its normalized-Laplacian spectrum and
transfers the optimized QAOA angle schedule of the kernel-nearest training
graph. On a query-counted MaxCut benchmark over connected graphs from six
families under family-held-out splits, STK ties an adiabatic ramp at circuit
depth `p=1` and beats the strongest ramp by a margin that grows with depth
(`p=2` and `p=3`). QAOA is evaluated with an exact numpy statevector simulator
and the analytic depth-1 closed form, cross-checked to `1e-9`, against a
brute-force MaxCut oracle. No heavyweight quantum framework is required.

## Install

From the source tree (this directory):

```bash
pip install .
# or, for development with the test suite:
pip install -e ".[dev]"
```

A Conda environment is also provided:

```bash
conda env create -f environment.yml && conda activate topoqaoa
```

The package targets Python `>=3.9` and depends only on numpy, scipy, networkx,
scikit-learn, matplotlib, and pyyaml (see `pyproject.toml` / `requirements.txt`).

## Reproduce

Installing the package provides the `stk-reproduce` console entry point, which
deterministically regenerates every table, figure, and macro from a fixed
configuration and seed:

```bash
stk-reproduce                                 # reported scale (configs/full.yaml, seed 0)
stk-reproduce --config configs/smoke.yaml     # laptop-scale check (a few seconds)
stk-reproduce --skip-run                       # rebuild tables/figures from an existing summary
```

The reported-scale run (`configs/full.yaml`) takes a few minutes on a laptop CPU
and writes:

| Artifact | Produced by | Read by the manuscript as |
| --- | --- | --- |
| `results/summary.json` | `scripts/run.py` | source of truth (per-policy approx. ratios with 95% CIs, one-shot ratios, queries-to-target, hit rate, budget frontier, per-family breakdown, provenance) |
| `results/macros.tex` | `scripts/make_tables.py` | `\input{code/results/macros.tex}` (every scalar the prose cites) |
| `results/tab_main.tex`, `tab_depth.tex`, `tab_family.tex`, `tab_frontier.tex` | `scripts/make_tables.py` | `\input{...}` table bodies |
| `figures/fig_schematic.pdf`, `fig_depth.pdf`, `fig_frontier.pdf`, `fig_family.pdf` | `scripts/make_figures.py` | `\includegraphics{...}` |

The equivalent Makefile targets operate on the source tree: `make test`,
`make demo`, `make full-run`, `make tables`, `make figures`, `make audit`.

> macOS: the Makefile, scripts, and the `stk-reproduce` entry point set
> `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`, because a Conda runtime
> and a pip-installed PyTorch can each bundle an OpenMP library. These settings
> affect loader behavior and thread scheduling only and leave results unchanged.

## Extend / tweak

Every experiment parameter lives in a YAML config (`configs/full.yaml`,
`configs/smoke.yaml`) and is parsed into `topoqaoa.config.Config`. Copy a config,
edit the fields, and pass it with `--config`:

```bash
cp configs/full.yaml configs/mine.yaml
# edit configs/mine.yaml
stk-reproduce --config configs/mine.yaml
```

Configurable fields (defaults from `configs/full.yaml`):

| Field | Meaning | Default |
| --- | --- | --- |
| `name` | run label recorded in provenance | `full` |
| `seed` | master seed threaded through graph generation, splits, and the refiner | `0` |
| `families` | graph families sampled (`erdos_renyi`, `regular`, `barabasi_albert`, `watts_strogatz`, `grid`, `stochastic_block`) | all six |
| `n_per_family` | instances generated per family | `14` |
| `sizes` | node counts (kept `<=14` so exact MaxCut is tractable) | `[8,10,12,14]` |
| `budget` | query budget: max circuit evaluations a policy may spend per instance | `28` |
| `target_ratio` | approximation ratio that defines "queries-to-target" | `0.95` |
| `depths` | QAOA circuit depths `p` benchmarked | `[1,2,3]` |
| `headline_depth` | depth used for the main/family/frontier tables | `2` |
| `stk_r` | STK spectral truncation window: number of low-frequency Laplacian eigenvalues | `6` |
| `stk_ridge` | ridge added to the kernel Gram diagonal | `0.01` |
| `label_restarts` | random restarts for the schedule-labelling oracle | `1` |

To **add a new config field**, add it as a dataclass field (with a default) in
`src/topoqaoa/config.py`; it is then available on the `Config` object passed to
`topoqaoa.runner.run` and can be consumed anywhere in the pipeline.

To **add a new graph family**, add a generator to
`src/topoqaoa/graph_generators.py` and reference its name in the `families` list.
To **add a new warm-start policy**, implement a class with `fit(...)` and
`propose(graph, p)` in `src/topoqaoa/baselines.py` (mirroring
`GraphConditionedPolicy` / the STK policy) and register it in `build_policies`;
the runner, tables, and figures pick it up automatically.

To **tune STK itself**, the kernel is `topoqaoa.kernels` (see
`truncated_spectrum(graph, r)` and the `STKKernel` class with `r`, `ridge`,
`use_descriptor`); the STK warm-start policy in `baselines.py` exposes `r`,
`ridge`, and `use_descriptor` constructor arguments.

### Use the package in your own project

After `pip install .` the modules are importable directly:

```python
import networkx as nx
from topoqaoa.kernels import STKKernel, truncated_spectrum
from topoqaoa.qaoa import ...        # exact statevector + depth-1 closed form
from topoqaoa.config import Config
from topoqaoa.runner import run

# 1) Compute the relabeling-invariant STK descriptor of any graph:
g = nx.erdos_renyi_graph(12, 0.5, seed=0)
feat = truncated_spectrum(g, r=6)

# 2) Or drive the full benchmark programmatically:
cfg = Config.load("configs/smoke.yaml")
summary = run(cfg, out="results")   # returns the dict written to summary.json
```

`truncated_spectrum` and the STK warm start are deterministic functions of a
graph's isomorphism class, so you can drop STK into an existing QAOA optimizer as
a seed generator: fit `STKKernel` on your training graphs and call its
`predict`/nearest-neighbor schedule for each new instance.

## Determinism and provenance

Reproduction is deterministic given the config; the seed is fixed at `0` and
threaded through every stochastic step. `results/summary.json` records seed,
platform, library versions, timestamp, runtime, and peak memory under
`provenance`. All manuscript numbers are `\input` from `results/macros.tex` and
the `tab_*.tex` files, so nothing is transcribed by hand.

## Cite this work

If you use this package or the results, please cite the paper:

```bibtex
@article{huynh2026topoqaoa,
  author  = {Huynh, Molena},
  title   = {Spectral-truncation graph kernels for {QAOA} warm starts:
             topology-conditioned schedule transfer beyond depth one},
  year    = {2026},
  note    = {Part of the spectral-truncation operators program},
}
```

Software metadata is also provided in `CITATION.cff`.

## License

MIT. See [`LICENSE`](LICENSE).
