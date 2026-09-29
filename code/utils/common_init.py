"""
Shared initial-population sampling.

FIX (common LHS initialisation across algorithms): the manuscript's
fairness protocol pairs runs by random seed, but pairing by seed alone
does not guarantee every algorithm starts from the SAME initial
population -- each algorithm's own `sampling` object draws its own LHS
sample from its own random-state consumption pattern, which differs
across algorithm implementations even under the same seed.

`shared_initial_population` generates exactly one (pop_size, n_var) LHS
sample per (problem, seed) pair, deterministically from the seed alone,
and every algorithm in that comparison is built with this SAME array
passed as `init_X` (KOA) / `sampling` (every pymoo baseline, which
accepts a raw ndarray for `sampling` and uses it directly as the
initial decision vectors -- see pymoo.core.initialization.Initialization.do).
"""
import numpy as np
from pymoo.operators.sampling.lhs import LHS


def shared_initial_population(problem, pop_size, seed):
    """Deterministic LHS sample of the decision space, shared by every
    algorithm compared under this (problem, seed) pair."""
    rng = np.random.default_rng(seed)
    X = LHS().do(problem, pop_size, random_state=rng).get("X")
    return np.asarray(X, dtype=float)
