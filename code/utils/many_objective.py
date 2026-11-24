"""
utils/many_objective.py

Builds DTLZ2, DTLZ7, WFG4, and MaF4 at M in {3, 5, 8, 10}, following the
decision-variable scaling conventions used by their original test-suite
papers (n_var = M - 1 + k for DTLZ, n_var = 2*(M-1) + l for WFG), and
returns a Pareto front reference set for each -- handling a pymoo quirk
discovered while building this: DTLZ2/DTLZ4 and every WFG problem raise
"Please provide reference directions for more than 3 objectives!" from
`pareto_front()` when M > 3, because pymoo does not auto-generate
reference directions above 3 objectives. This module supplies dense
reference directions explicitly (reusing problems/ref_dirs_dense.py) so
`pareto_front()` succeeds at every M without silently falling back to a
sparse or wrong front.

Verified directly (see chat) for M in {3,5,8,10}: DTLZ2, DTLZ7, WFG4,
and MaF4 all construct and produce a non-empty pareto_front().
"""
from pymoo.problems import get_problem
from problems.maf import MaF4
from problems.ref_dirs_dense import dense_simplex_points

# k offsets per the original DTLZ paper (Deb et al. 2005): DTLZ1 k=5,
# DTLZ2-6 k=10, DTLZ7 k=20. n_var = n_obj - 1 + k.
_DTLZ_K = {"dtlz1": 5, "dtlz2": 10, "dtlz3": 10, "dtlz4": 10,
           "dtlz5": 10, "dtlz6": 10, "dtlz7": 20}

# WFG position-parameter count k = 2*(M-1) (must be a multiple of M-1,
# per pymoo's WFG constructor), distance-parameter count l fixed at 6 --
# this matches n_var=10 at M=3 already used for WFG1/4/6/9 elsewhere in
# this repository (k=4, l=6), generalised to any M.
_WFG_L = 6

MANY_OBJECTIVE_PROBLEMS = ["DTLZ2", "DTLZ7", "WFG4", "MaF4"]
M_VALUES = [3, 5, 8, 10]


def build_many_objective_problem(name, M):
    """Returns (problem, kind_needs_explicit_ref_dirs: bool)."""
    if name == "DTLZ2":
        n_var = M - 1 + _DTLZ_K["dtlz2"]
        return get_problem("dtlz2", n_var=n_var, n_obj=M), True
    elif name == "DTLZ7":
        n_var = M - 1 + _DTLZ_K["dtlz7"]
        return get_problem("dtlz7", n_var=n_var, n_obj=M), False
    elif name == "WFG4":
        k = 2 * (M - 1)
        n_var = k + _WFG_L
        return get_problem("wfg4", n_var=n_var, n_obj=M), True
    elif name == "MaF4":
        return MaF4(n_obj=M, n_var=None), False
    raise ValueError(f"Unknown many-objective problem: {name}")


def many_objective_pareto_front(problem, name, M, n_points=1000):
    """pareto_front() with the explicit ref_dirs pymoo requires for
    DTLZ2/WFG4 at M>3 (see module docstring); a no-op passthrough for
    DTLZ7/MaF4, which build their own dense reference sets internally."""
    if name in ("DTLZ2", "WFG4") and M > 3:
        rd = dense_simplex_points(M, n_points=n_points)
        return problem.pareto_front(ref_dirs=rd)
    if name == "MaF4":
        return problem.pareto_front(n_points=n_points)
    if str(name).upper() == "DTLZ7" and M > 3:
        return dtlz7_reference_front(M, n_points)
    return problem.pareto_front()


def dtlz7_reference_front(M, n_points=1000, n_samples=None, seed=12345):
    """Reference Pareto front of DTLZ7 for any number of objectives M.

    The first M-1 objectives of the Pareto-optimal set lie in the intervals
    [0, 0.251412] and [0.631627, 0.859401]; the last objective follows from
    f_M = 2 * (M - sum_i f_i / 2 * (1 + sin(3*pi*f_i))) with g = 1.
    Points are sampled in these intervals with a fixed seed, mapped as in
    PlatEMO, filtered for non-dominance, and reduced to n_points.
    """
    import numpy as np
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    rng = np.random.default_rng(seed)
    n_samples = n_samples or max(20 * n_points, 20000)
    a, b, c, d = 0.0, 0.251412, 0.631627, 0.859401
    med = (b - a) / ((d - c) + (b - a))
    U = rng.random((n_samples, M - 1))
    X = np.where(U <= med, U * (b - a) / med + a, (U - med) * (d - c) / (1 - med) + c)
    fM = 2.0 * (M - np.sum(X / 2.0 * (1.0 + np.sin(3.0 * np.pi * X)), axis=1))
    F = np.column_stack([X, fM])
    F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    if len(F) > n_points:
        F = F[rng.choice(len(F), n_points, replace=False)]
    return F
