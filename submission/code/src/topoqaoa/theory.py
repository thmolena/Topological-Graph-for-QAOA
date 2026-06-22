"""From-principles guide for topology-conditioned QAOA warm starts.

MaxCut and QAOA
---------------
For a graph G = (V, E), MaxCut asks for binary vertex labels that maximize the
number of edges crossing the split. qaoa.py implements depth-one QAOA, where a
cost phase and a mixer rotation prepare a parameterized quantum state. The reward
is the expected cut value divided by the exact brute-force MaxCut value from
maxcut_exact.py.

Closed-form objective
---------------------
At depth one, the expected cut has an analytic per-edge formula. qaoa.py
implements both this formula and a statevector evaluator, and the tests require
agreement to numerical precision. The closed form makes the benchmark fast while
keeping an exact reference.

Graph descriptors
-----------------
descriptors.py builds fixed-length, relabeling-invariant vectors from degree
statistics, motifs, Weisfeiler-Lehman color histograms, cycle information,
Laplacian spectra and connectivity. If vertex names are permuted, the descriptor
does not change. This is the representation used by learned warm starts.

Machine learning
----------------
baselines.py compares random, spectral, topology, graph-conditioned and
cross-entropy policies. The graph-conditioned policy is a deterministic
k-nearest-neighbour regressor in descriptor space: graphs with nearby invariant
features share warm-start angles. splits.py enforces family-held-out evaluation
so the learned policy cannot see the test graph family during fitting.

Query-counted evaluation
------------------------
env.py refines the warm start with coordinate ascent while counting every
objective query. This turns the hardware-relevant cost into a measured metric
instead of an implicit optimizer detail. runner.py writes the single source of
truth, results/summary.json; scripts derive tables and figures from it.

Reproduction
------------
From code/:

    export PYTHONPATH=src
    make test
    bash scripts/reproduce_all.sh full
"""


GUIDE = __doc__


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
