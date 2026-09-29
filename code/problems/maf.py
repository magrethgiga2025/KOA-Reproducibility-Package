"""
MaF4 and MaF7 from the MaF many-objective benchmark suite.

Faithful NumPy port of the reference MATLAB implementation in PlatEMO
(BIMK Group, https://github.com/BIMK/PlatEMO), reference implementation
for:

    R. Cheng, M. Li, Y. Tian, X. Zhang, S. Yang, Y. Jin, X. Yao.
    "A benchmark test suite for evolutionary many-objective optimization."
    Complex & Intelligent Systems, 3(1):67-81, 2017.

MaF4: inverted and scaled DTLZ3 (multimodal, badly-scaled, concave-inverted
front -- complements WFG1's deceptive/separable character).
MaF7: equivalent to DTLZ7 (mixed continuous/disconnected Pareto front --
many-objective analogue of ZDT3's disconnection theme).
"""
import numpy as np
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from problems.ref_dirs_dense import dense_simplex_points


class MaF4(Problem):
    """Inverted, exponentially-scaled DTLZ3."""

    def __init__(self, n_obj=3, n_var=None, **kwargs):
        if n_var is None:
            n_var = n_obj + 9
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        N, D = X.shape
        M = self.n_obj
        tail = X[:, M - 1:] - 0.5
        g = 100.0 * (D - M + 1 + np.sum(tail ** 2 - np.cos(20 * np.pi * tail), axis=1))

        ones_col = np.ones((N, 1))
        cum = np.cumprod(np.concatenate([ones_col, np.cos(X[:, :M - 1] * np.pi / 2.0)], axis=1), axis=1)
        cum = np.fliplr(cum)
        sin_tail = np.concatenate([ones_col, np.sin(X[:, M - 2::-1] * np.pi / 2.0)], axis=1)

        base = (1.0 + g)[:, None]
        PopObj = base - base * cum * sin_tail
        scale = 2.0 ** np.arange(1, M + 1)
        out["F"] = PopObj * scale

    def pareto_front(self, n_points=2000, **kwargs):
        # FIX (sparse IGD reference sets): dense simplex sampling instead
        # of a fixed n_partitions=12/6, which under-samples the reference
        # set relative to what a reliable IGD estimate needs.
        M = self.n_obj
        rd = dense_simplex_points(M, n_points=n_points)
        rd = rd / np.linalg.norm(rd, axis=1, keepdims=True)
        scale = 2.0 ** np.arange(1, M + 1)
        return (1.0 - rd) * scale


class MaF7(Problem):
    """DTLZ7-equivalent: mixed continuous/disconnected Pareto front."""

    def __init__(self, n_obj=3, n_var=None, **kwargs):
        if n_var is None:
            n_var = n_obj + 19
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        N, D = X.shape
        M = self.n_obj
        g = 1.0 + 9.0 * np.mean(X[:, M - 1:], axis=1)
        PopObj = np.zeros((N, M))
        PopObj[:, :M - 1] = X[:, :M - 1]
        PopObj[:, M - 1] = (1.0 + g) * (
            M - np.sum(
                PopObj[:, :M - 1] / (1.0 + g[:, None])
                * (1.0 + np.sin(3 * np.pi * PopObj[:, :M - 1])),
                axis=1,
            )
        )
        out["F"] = PopObj

    def pareto_front(self, n_points=2000, **kwargs):
        # FIX (sparse IGD reference sets, review comment on MaF7/LSMOP9's
        # disconnected fronts): M-1=2 previously gave only 13 reference
        # points via n_partitions=12; now densified to >= n_points.
        M = self.n_obj
        if M == 2:
            x = np.linspace(0, 1, n_points)
            y = 2.0 * (2.0 - x / 2.0 * (1.0 + np.sin(3 * np.pi * x)))
            F = np.column_stack([x, y])
            nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
            return F[nd]
        rd = dense_simplex_points(M - 1, n_points=n_points)
        interval = [0, 0.251412, 0.631627, 0.859401]
        med = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        Xg = rd.copy()
        lo_mask = Xg <= med
        hi_mask = ~lo_mask
        Xg[lo_mask] = Xg[lo_mask] * (interval[1] - interval[0]) / med + interval[0]
        Xg[hi_mask] = (Xg[hi_mask] - med) * (interval[3] - interval[2]) / (1 - med) + interval[2]
        last = 2.0 * (M - np.sum(Xg / 2.0 * (1.0 + np.sin(3 * np.pi * Xg)), axis=1))
        return np.column_stack([Xg, last])


def get_maf(which, n_obj=3, n_var=None):
    if which == 4:
        return MaF4(n_obj=n_obj, n_var=n_var)
    elif which == 7:
        return MaF7(n_obj=n_obj, n_var=n_var)
    raise ValueError("Only MaF4 and MaF7 are implemented")
