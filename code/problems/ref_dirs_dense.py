"""
Dense reference-direction generation for IGD reference (true-PF) sets.

FIX (sparse IGD reference sets on disconnected fronts): MaF7's and
LSMOP9's M=3 Pareto fronts were built from
    get_reference_directions("das-dennis", M - 1, n_partitions=12)
which, for M-1=2, is a 1-simplex (a line segment in 2 coordinates)
divided into 12 parts -- i.e. only 13 reference points. IGD computed
against 13 reference points on a genuinely disconnected 3-objective
front is far too sparse to be a reliable indicator (this was the
review's concern, and our own diagnosis agreed).

`dense_simplex_points(dim, n_points)` replaces every das-dennis call
used to build a true-PF reference set from a `dim`-dimensional simplex,
and always returns AT LEAST `n_points` points (default 2000, i.e.
roughly two orders of magnitude denser than the previous 13-91 point
sets used for MaF4/MaF7/LSMOP9's M=3 fronts):

  * dim == 1: exact closed-form linspace (only used directly by the
    M==2 special cases in maf.py/lsmop.py, not through this function).
  * dim >= 2: das-dennis, but with the SMALLEST number of partitions p
    such that C(p+dim-1, dim-1) >= n_points, found by direct search
    rather than a fixed p=12. This keeps the fast, deterministic,
    combinatorial das-dennis construction (appropriate for a one-off
    reference-set build that may be called many times across runs)
    while guaranteeing density regardless of dim.
"""
import numpy as np
from math import comb
from pymoo.util.ref_dirs import get_reference_directions


def _min_partitions_for(dim, n_points):
    """Smallest n_partitions p with C(p+dim-1, dim-1) >= n_points."""
    p = 1
    while comb(p + dim - 1, dim - 1) < n_points:
        p += 1
    return p


def dense_simplex_points(dim, n_points=2000):
    """>= n_points points on the (dim-1)-dimensional simplex, embedded
    in `dim` coordinates (i.e. the usual das-dennis convention where
    `dim` equals the number of coordinates / objectives being spanned).
    """
    if dim <= 1:
        return np.ones((1, max(dim, 1)))
    p = _min_partitions_for(dim, n_points)
    return get_reference_directions("das-dennis", dim, n_partitions=p)
