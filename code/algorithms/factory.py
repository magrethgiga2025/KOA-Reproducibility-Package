"""
Algorithm factory -- KOA Final V3, plus 6 baselines.

FIXES APPLIED (see README_FIXES.md):

FIX 9 -- Baseline population-size fairness. For M=3, das-dennis with
  n_partitions=12 produces C(14,2)=91 reference directions, not 100.
  NSGA-III and RVEA were built WITHOUT an explicit `pop_size`, so pymoo
  silently defaulted their population to 91 (=len(ref_dirs)), while KOA
  and NSGA-II/CMOPSO/AGE-MOEA2 ran at 100. MOEA/D ties its population
  size to the reference-direction count by design (one subproblem per
  direction) and has no independent `pop_size` knob at all.

  Fixed by:
    (a) generating reference directions with the Riesz s-Energy method
        (`get_reference_directions("energy", n_obj, pop_size, seed=1)`),
        which accepts an exact point count for any n_obj -- so the
        reference-direction set itself has exactly `pop_size` members
        for every M, instead of whatever a das-dennis partitioning
        happens to produce; and
    (b) passing `pop_size=pop_size` explicitly to NSGA-III and RVEA as
        a second, redundant safeguard in case (a) is ever bypassed.

  With this fix, every algorithm -- KOA, NSGA-II, NSGA-III, MOEA/D, RVEA,
  AGE-MOEA2, CMOPSO -- runs with exactly `pop_size` individuals per
  generation for every M, so every algorithm consumes exactly
  pop_size * (n_gen + 1) function evaluations for identical (n_gen,
  pop_size) settings (see config.FE_BUDGET).

FIX (common initialisation) -- `init_X`, when supplied, is threaded
  through to every algorithm's `sampling` parameter (pymoo accepts a
  plain ndarray there) so every algorithm in a given (problem, seed)
  comparison starts from the IDENTICAL initial population, strengthening
  the paired-seed comparison beyond "same seed" to "same seed AND same
  initial population".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from functools import lru_cache
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


@lru_cache(maxsize=None)
def _exact_ref_dirs_cached(n_obj, n_points, seed):
    """Cached core of _exact_ref_dirs. The Riesz s-Energy method used
    for n_obj>=3 takes several seconds per call and is deterministic
    given (n_obj, n_points, seed), so every run at a given (M, pop_size)
    reuses the same computed directions instead of repeating an
    expensive optimisation on every one of the 30 seeds x 7 algorithms."""
    if n_obj == 2:
        return get_reference_directions("das-dennis", n_obj,
                                         n_partitions=n_points - 1)
    return get_reference_directions("energy", n_obj, n_points, seed=seed)


def _exact_ref_dirs(n_obj, n_points, seed=1):
    """Reference directions with EXACTLY n_points members, for any n_obj.

    For n_obj == 2, das-dennis with n_partitions = n_points-1 already
    gives exactly n_points points and is deterministic / cheap, so we
    keep using it there. For n_obj >= 3 we use the Riesz s-Energy method,
    which accepts an arbitrary point count directly (Fix 9), instead of
    das-dennis's fixed-partition combinatorics that generally cannot hit
    an arbitrary target count exactly. Cached (see _exact_ref_dirs_cached)
    since this is otherwise a repeated multi-second cost per run.
    """
    return _exact_ref_dirs_cached(n_obj, n_points, seed).copy()


def build_algorithm(name, n_obj=3, pop_size=100, n_archive=120, seed=None,
                     init_X=None):

    if name == "KOA":
        from algorithms.koa_final_v3 import KOAFinalV3
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X)

    elif name == "NSGA-II":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.lhs import LHS
        sampling = init_X if init_X is not None else LHS()
        return NSGA2(pop_size=pop_size, sampling=sampling,
                     crossover=SBX(prob=0.9, eta=15),
                     mutation=PM(eta=20), eliminate_duplicates=True)

    elif name == "NSGA-III":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.lhs import LHS
        ref_dirs = _exact_ref_dirs(n_obj, pop_size)          # Fix 9(a)
        sampling = init_X if init_X is not None else LHS()
        return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size,   # Fix 9(b)
                     sampling=sampling,
                     crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20))

    elif name == "MOEA/D":
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.operators.sampling.lhs import LHS
        ref_dirs = _exact_ref_dirs(n_obj, pop_size)          # Fix 9(a)
        sampling = init_X if init_X is not None else LHS()
        return MOEAD(ref_dirs=ref_dirs, n_neighbors=15,
                     decomposition=None, prob_neighbor_mating=0.9,
                     sampling=sampling)

    elif name == "RVEA":
        from pymoo.algorithms.moo.rvea import RVEA
        from pymoo.operators.sampling.lhs import LHS
        ref_dirs = _exact_ref_dirs(n_obj, pop_size)          # Fix 9(a)
        sampling = init_X if init_X is not None else LHS()
        return RVEA(ref_dirs=ref_dirs, pop_size=pop_size,    # Fix 9(b)
                    sampling=sampling)

    elif name == "AGE-MOEA2":
        sampling = init_X
        try:
            from pymoo.algorithms.moo.age2 import AGEMOEA2
            return (AGEMOEA2(pop_size=pop_size, sampling=sampling)
                    if sampling is not None else AGEMOEA2(pop_size=pop_size))
        except Exception:
            from pymoo.algorithms.moo.age import AGEMOEA
            return (AGEMOEA(pop_size=pop_size, sampling=sampling)
                    if sampling is not None else AGEMOEA(pop_size=pop_size))

    elif name == "CMOPSO":
        from pymoo.algorithms.moo.cmopso import CMOPSO
        if init_X is not None:
            return CMOPSO(pop_size=pop_size, sampling=init_X)
        return CMOPSO(pop_size=pop_size)

    raise ValueError(f"Unknown algorithm: {name}")
