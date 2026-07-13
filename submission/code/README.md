# specops-stk

Reference implementation for **"Spectral-truncation graph kernels for QAOA warm
starts: topology-conditioned schedule transfer beyond depth one"** (Molena
Huynh, North Carolina State University, 2026), part of the *spectral-truncation
operators* program. The distribution name is `specops-stk`; the importable
module is `topoqaoa`.

## Summary

This package introduces and evaluates the **spectral-truncation kernel (STK)**:
a relabeling-invariant, positive-definite kernel over graphs that represents each
graph by a finite low-frequency window of its normalized-Laplacian spectrum and,
for an unseen test graph, transfers the optimized QAOA angle schedule of the
single most similar training graph. On a query-counted MaxCut benchmark over 84
connected graphs drawn from six structural families under family-held-out splits,
STK reproduces the established depth-one parity with a strong adiabatic ramp
(paired advantage `+0.0000 ± 0.0001`) and then surpasses the strongest ramp by a
margin that grows monotonically with circuit depth: `+0.0103` at `p=2` and
`+0.0262` at `p=3` on the final approximation ratio, and by a wider one-shot
margin still (`+0.0416` at `p=2`, `+0.0454` at `p=3`), because the transferred
schedule is near-optimal on the very first objective query. QAOA expectations are
computed exactly by a numpy statevector simulator and cross-verified against the
analytic depth-one closed form to `1e-9`, and approximation ratios are measured
against a brute-force MaxCut oracle. No heavyweight quantum framework is required.

## Background and problem setting (from first principles)

**QAOA and MaxCut.** The quantum approximate optimization algorithm (QAOA) is a
variational quantum method for combinatorial optimization. For the MaxCut
problem on a graph `G = (V, E)` — partition the vertices into two sets so as to
maximize the number of edges crossing the partition — QAOA encodes the objective
in a cost operator `C = Σ_{(u,v)∈E} ½(1 − Z_u Z_v)` and prepares, at circuit
depth `p`, the trial state

```text
|γ, β⟩ = Π_{ℓ=1..p} e^{−iβ_ℓ B} e^{−iγ_ℓ C} |+⟩^{⊗n},   B = Σ_v X_v,
```

then tunes the `2p` real angles `θ = (γ_1..γ_p, β_1..β_p)` to maximize the
expected cut `⟨C(γ, β)⟩`. The quality of a run is the *approximation ratio*
`⟨C⟩ / C*`, where `C*` is the true MaxCut value.

**Why queries are the scarce resource.** A quantum device only ever *samples*
the objective: each estimate of `⟨C⟩` is assembled from many circuit executions.
The dominant cost of QAOA in practice is therefore the number of objective
evaluations spent by a classical optimizer searching for good angles, a search
made harder by barren plateaus and cost concentration. The operationally relevant
figure of merit is not the best achievable cut but the cut a policy reaches
*within a fixed query budget*, and in particular the **one-shot ratio** at query
`q=1`, which measures pure warm-start quality with zero optimization.

**Warm starts and the depth-one trap.** A *warm start* supplies good initial
angles so the optimizer begins inside a high-quality basin. The premise that one
can be *learned* across instances rests on documented QAOA regularities: the
objective concentrates across typical instances of a given structure, and optimal
parameters transfer between graphs and across sizes. Yet under controlled,
query-counted comparison the depth-one verdict has been negative: at `p=1` a
warm start is a single pair `(γ_0, β_0)`, and a one-line spectral or adiabatic
rule already fixes the only relevant angle scale, so every structure-aware policy
merely ties the ramp. This work shows that verdict is an artifact of `p=1`.
Beyond depth one a warm start must specify an entire `2p`-angle schedule that a
single physical scale can only crudely approximate, and there topology
conditioning begins to pay.

**Why transfer, not averaging.** Optimal QAOA schedules are not unique:
time-reversal and mixer/cost periodicities place equally good schedules in
distinct, symmetry-related basins separated by inferior regions. Copying one
optimized schedule from a sufficiently similar graph lands the optimizer inside a
genuine basin (near-optimal on the first query); regressing several schedules
toward their mean places the seed *between* basins, which the refiner must then
climb out of. The remaining question is which invariant notion of graph
similarity governs schedule transfer — answered here by the spectral-truncation
kernel.

## Contributions

1. **A depth-resolved finding.** Within a controlled, query-counted,
   leakage-checked benchmark, the depth-one "learned warm starts only match"
   verdict is shown to be an artifact of `p=1`: topology-conditioned warm starts
   tie an adiabatic ramp at `p=1` and surpass it for `p≥2` by a margin that grows
   monotonically with depth (`+0.0103` at `p=2`, `+0.0262` at `p=3`).

2. **A transferable warm-start operator (STK).** A positive-definite,
   relabeling-invariant kernel on the truncated normalized-Laplacian spectrum
   whose nearest neighbour transfers a single optimized depth-`p` schedule. It is
   proven invariant and positive definite, making the warm start a deterministic
   function of the graph's isomorphism class. It attains the best one-shot
   ratio of any policy at `p=1` and `p=2` and the best final ratio at `p=2`;
   at `p=3` the averaging descriptor-mean learner draws statistically level
   (final `0.8925` vs `0.8902`, one-shot `0.8853` vs `0.8845`, overlapping
   95% CIs), while STK's margin over the ramps keeps growing with depth.

3. **A cross-verified, reproducible benchmark.** A proven relabeling-invariant
   descriptor and kernel, a depth-one objective cross-verified to `1e-9` by two
   independent evaluators (analytic closed form and exact statevector), exact
   approximation ratios against a brute-force MaxCut oracle, family-held-out
   transfer with a programmatic leakage check, and a `pip`-installable package
   that regenerates every figure, table, and number from fixed seeds.

## Method

Each graph `G` is mapped to two relabeling-invariant objects. The
**spectral-truncation feature** `σ_r(G)` retains the `r=6` smallest non-zero
eigenvalues of the normalized Laplacian `L = D^{-1/2} L D^{-1/2}`, in increasing
order and right-padded to length `r` — a finite low-frequency window of the graph
operator, adapting the C*-algebraic spectral-truncation kernels of Hashimoto et
al. (2024) from operators to graphs. These low-lying modes encode community
structure, regularity, and connectivity bottlenecks. A **topology descriptor**
`φ(G)` concatenates degree statistics, motif densities from traces of adjacency
powers, a hashed Weisfeiler–Lehman colour-refinement histogram, cycle features,
and Laplacian-spectral and connectivity features. Both are invariant under node
relabeling.

On standardized features the kernel is a Hadamard product of two Gaussian factors,

```text
k(G, G') = exp(−‖σ_r(G) − σ_r(G')‖² / 2ℓ_s²) · exp(−‖φ(G) − φ(G')‖² / 2ℓ_d²),
```

with bandwidths set by the median heuristic. It is positive definite by the
Schur product theorem and invariant by construction. The STK policy outputs the
optimized schedule of the kernel-nearest training graph, `Θ(G) = θ_{j*}` with
`j* = argmax_i k(G, G_i)`. That schedule seeds a coordinate-ascent refiner that
counts **every** objective evaluation against a fixed budget; the running-best
ratio versus query count (the query-budget frontier) and its first point (the
one-shot ratio) are the hardware-relevant comparisons.

## Main results

Reported-scale configuration (`configs/full.yaml`): six families (Erdős–Rényi,
random 3-regular, Barabási–Albert, Watts–Strogatz, 2-D grid, stochastic block
model), 84 connected graphs, sizes `n ∈ {8, 10, 12, 14}`, query budget 28, master
seed 0, depths `p ∈ {1, 2, 3}`, family-held-out with a leakage-clean check.

Held-out approximation ratio versus depth (best per-instance ramp / descriptor
mean / STK / oracle, and the paired STK − ramp advantage):

| `p` | best ramp | descriptor mean | STK (ours) | oracle | Δ final | Δ one-shot |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7825 | 0.7824 | 0.7825 | 0.7827 | +0.0000 ± 0.0001 | +0.0030 |
| 2 | 0.8394 | 0.8470 | **0.8503** | 0.8536 | +0.0103 ± 0.0019 | +0.0416 |
| 3 | 0.8622 | 0.8925 | **0.8902** | 0.8999 | +0.0262 ± 0.0026 | +0.0454 |

Bold marks STK where it meets or exceeds the best per-instance ramp (as in the
manuscript's Table); note that at `p=3` the descriptor mean's `0.8925` is
nominally the highest final ratio, statistically indistinguishable from STK's
`0.8902`.

At the primary depth `p=2`, STK attains the best one-shot ratio (`0.8449`) by a
decisive margin over the ramps (`0.8032` topology, `0.7483` spectral) and the
descriptor mean (`0.8050`), and the highest final ratio (`0.8503`). The
averaging-based descriptor mean reaches a competitive final ratio only after the
refiner repairs its between-basins seed, confirming that transfer is the more
query-efficient form of the same topology conditioning. Both `p≥2` final
advantages lie outside their paired 95% confidence intervals. No policy reached
the stringent `0.95` target within 28 queries at this depth; this is reported
transparently rather than by relaxing the target.

## Significance

The work supplies a transferable, theoretically grounded warm-start operator and
a reusable benchmark that resolves *where* learned QAOA warm starts help. It
separates two questions the literature often conflates: whether learned warm
starts help at all (they do, beyond depth one) and whether the *form* of learning
matters for query efficiency (single-schedule transfer is near-optimal per query
while averaging needs refinement). Because STK is near-optimal on the first query
— exactly the regime in which each query is an expensive circuit execution — it
is directly relevant to hardware practice, and its relabeling-invariant kernel
makes "most similar graph" a principled, deterministic choice.

## Installation and reproduction

Install from PyPI (distribution name `specops-stk`, imports as `topoqaoa`):

```bash
pip install specops-stk
```

Or install from this source tree:

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

Installing the package provides the `stk-reproduce` console entry point, which
deterministically regenerates every table, figure, and macro from a fixed
configuration and seed:

```bash
stk-reproduce                                 # reported scale (configs/full.yaml, seed 0), ~7 min
stk-reproduce --quick                         # smoke-scale check (configs/smoke.yaml), a few seconds
stk-reproduce --skip-run                      # rebuild tables/figures from an existing summary
```

The reported-scale run (`configs/full.yaml`) took 423 s (about seven minutes) at
163 MB peak memory on a laptop CPU (recorded in `results/summary.json`
provenance) and writes, mirroring into the manuscript's `figures/` and `tables/`:

| Artifact | Produced by | Read by the manuscript as |
| --- | --- | --- |
| `results/summary.json` | `scripts/run.py` | source of truth (per-policy approx. ratios with 95% CIs, one-shot ratios, queries-to-target, hit rate, budget frontier, per-family breakdown, provenance) |
| `results/macros.tex` | `scripts/make_tables.py` | `\input{tables/macros.tex}` (every scalar the prose cites) |
| `results/tab_main.tex` | `scripts/make_tables.py` | `\input{tables/tab_main.tex}` — Table II (per-policy, `p=2`) |
| `results/tab_depth.tex` | `scripts/make_tables.py` | `\input{tables/tab_depth.tex}` — Table I (ratio vs depth) |
| `results/tab_family.tex` | `scripts/make_tables.py` | `\input{tables/tab_family.tex}` — Appendix Table III (per family) |
| `results/tab_frontier.tex` | `scripts/make_tables.py` | `\input{tables/tab_frontier.tex}` — Appendix Table IV (frontier) |
| `figures/fig_schematic.pdf` | `topoqaoa.plotting.fig_schematic` via `scripts/make_figures.py` | Fig. 1 (pipeline schematic) |
| `figures/fig_depth.pdf` | `topoqaoa.plotting.fig_depth` | Fig. 2 (ratio and paired advantage vs depth) |
| `figures/fig_frontier.pdf` | `topoqaoa.plotting.fig_frontier` | Fig. 3 (query-budget frontier per depth) |
| `figures/fig_family.pdf` | `topoqaoa.plotting.fig_family_bars` | Fig. 4 (per-family bars) |

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

After installation the modules are importable directly:

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
graph's isomorphism class, so STK drops into an existing QAOA optimizer as a seed
generator: fit `STKKernel` on your training graphs and call its
`predict`/nearest-neighbor schedule for each new instance.

## Determinism and provenance

Reproduction is deterministic given the config; the seed is fixed at `0` and
threaded through every stochastic step. `results/summary.json` records seed,
platform, library versions, timestamp, runtime, and peak memory under
`provenance`. All manuscript numbers are `\input` from `results/macros.tex` and
the `tab_*.tex` files, so nothing is transcribed by hand.

## Cite this work

If this package or its results are used, please cite the paper:

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
</content>
</invoke>
