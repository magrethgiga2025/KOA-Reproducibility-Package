"""
LSMOP1-9 test suite for large-scale multi-objective optimization.

Faithful NumPy port of the reference MATLAB implementation in PlatEMO
(BIMK Group, https://github.com/BIMK/PlatEMO), itself the reference
implementation released alongside the original paper:

    R. Cheng, Y. Jin, M. Olhofer, B. Sendhoff.
    "Test problems for large-scale multiobjective and many-objective
    optimization." IEEE Transactions on Cybernetics, 47(12):4108-4121, 2017.

All objective formulas, sub-component functions (Sphere, Griewank,
Rastrigin, Rosenbrock, Ackley, Schwefel), variable-grouping logic, and
linkage transformations are ported 1:1 from PlatEMO's LSMOP1.m-LSMOP9.m.
Pareto-front sampling for IGD reference sets uses pymoo's Das-Dennis
reference-direction generator in place of PlatEMO's UniformPoint, which
produces equivalent simplex-lattice samples for the simple (non-LSMOP9)
fronts.
"""
import numpy as np
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions


# ----------------------------------------------------------------------
# Sub-component (landscape) functions, ported verbatim from PlatEMO
# ----------------------------------------------------------------------
def _sphere(x):
    return np.sum(x ** 2, axis=1)


def _griewank(x):
    d = x.shape[1]
    idx = np.sqrt(np.arange(1, d + 1))
    return np.sum(x ** 2, axis=1) / 4000.0 - np.prod(np.cos(x / idx), axis=1) + 1.0


def _schwefel(x):
    return np.max(np.abs(x), axis=1)


def _rastrigin(x):
    return np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x) + 10.0, axis=1)


def _rosenbrock(x):
    return np.sum(100.0 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def _ackley(x):
    d = x.shape[1]
    return (20.0 - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=1) / d))
            - np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / d) + np.exp(1.0))


_FUNCS = {
    "sphere": _sphere, "griewank": _griewank, "schwefel": _schwefel,
    "rastrigin": _rastrigin, "rosenbrock": _rosenbrock, "ackley": _ackley,
}

# Odd-group / even-group sub-function assignment per LSMOP1-9 (from PlatEMO)
_LSMOP_FUNCS = {
    1: ("sphere", "sphere"),
    2: ("griewank", "schwefel"),
    3: ("rastrigin", "rosenbrock"),
    4: ("ackley", "griewank"),
    5: ("sphere", "sphere"),
    6: ("rosenbrock", "schwefel"),
    7: ("ackley", "rosenbrock"),
    8: ("griewank", "sphere"),
    9: ("sphere", "ackley"),
}
# Problems 5-9 use the nonlinear (cosine) linkage + spherical/disconnected PF shape
_NONLINEAR_LINKAGE = {5, 6, 7, 8, 9}


def _variable_groups(n_var, n_obj, nk=5):
    """Reproduce PlatEMO's logistic-map-based subcomponent length partition.

    MATLAB: c = 3.8*0.1*(1-0.1); for i=1:M-1, c=[c,3.8*c(end)*(1-c(end))]; end
    This yields exactly M entries (1 initial + M-1 appended).
    """
    c = [3.8 * 0.1 * (1 - 0.1)]
    for _ in range(n_obj - 1):
        c.append(3.8 * c[-1] * (1 - c[-1]))
    c = np.array(c)
    sublen = np.floor(c / c.sum() * (n_var - n_obj + 1) / nk).astype(int)
    sublen = np.maximum(sublen, 1)
    length = np.concatenate([[0], np.cumsum(sublen * nk)])
    return sublen, length


class LSMOP(Problem):
    """Generic LSMOP1-9 problem. `which` selects the variant (1-9)."""

    def __init__(self, which=1, n_obj=3, n_var=None, nk=5, **kwargs):
        if which not in range(1, 10):
            raise ValueError("LSMOP variant must be 1-9")
        self.which = which
        self.nk = nk
        if n_var is None:
            n_var = 100 * n_obj
        self.sublen, self.length = _variable_groups(n_var, n_obj, nk)

        xl = np.zeros(n_var)
        xu = np.concatenate([np.ones(n_obj - 1), 10.0 * np.ones(n_var - n_obj + 1)])
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        N, D = X.shape
        M = self.n_obj
        which = self.which
        X = X.copy()

        if which in _NONLINEAR_LINKAGE:
            link = 1.0 + np.cos((np.arange(M, D + 1) / D) * np.pi / 2.0)
        else:
            link = 1.0 + (np.arange(M, D + 1) / D)
        X[:, M - 1:] = link * X[:, M - 1:] - (X[:, [0]] * 10.0)

        odd_name, even_name = _LSMOP_FUNCS[which]
        odd_f, even_f = _FUNCS[odd_name], _FUNCS[even_name]

        G = np.zeros((N, M))
        for i in range(M):  # 0-indexed group i -> matches MATLAB group i+1
            f = odd_f if (i % 2 == 0) else even_f
            start_base = self.length[i] + M - 1
            for j in range(self.nk):
                lo = start_base + j * self.sublen[i]
                hi = start_base + (j + 1) * self.sublen[i]
                G[:, i] += f(X[:, lo:hi])
        G = G / self.sublen / self.nk

        if which == 9:
            Gsum = 1.0 + np.sum(G, axis=1)
            PopObj = np.zeros((N, M))
            PopObj[:, :M - 1] = X[:, :M - 1]
            PopObj[:, M - 1] = (1.0 + Gsum) * (
                M - np.sum(
                    PopObj[:, :M - 1] / (1.0 + Gsum[:, None])
                    * (1.0 + np.sin(3 * np.pi * PopObj[:, :M - 1])),
                    axis=1,
                )
            )
            out["F"] = PopObj
            return

        ones_col = np.ones((N, 1))
        if which in _NONLINEAR_LINKAGE:
            cum = np.cumprod(np.concatenate([ones_col, np.cos(X[:, :M - 1] * np.pi / 2.0)], axis=1), axis=1)
            cum = np.fliplr(cum)
            tail = np.concatenate([ones_col, np.sin(X[:, M - 2::-1] * np.pi / 2.0)], axis=1)
            shifted = np.concatenate([G[:, 1:], np.zeros((N, 1))], axis=1)
            PopObj = (1.0 + G + shifted) * cum * tail
        else:
            cum = np.cumprod(np.concatenate([ones_col, X[:, :M - 1]], axis=1), axis=1)
            cum = np.fliplr(cum)
            tail = np.concatenate([ones_col, 1.0 - X[:, M - 2::-1]], axis=1)
            PopObj = (1.0 + G) * cum * tail

        out["F"] = PopObj

    def pareto_front(self, n_points=2000, **kwargs):
        # FIX (sparse IGD reference sets): dense simplex sampling
        # (>= n_points, default 2000) replaces the previous fixed
        # n_partitions=12/6, which for LSMOP9's disconnected M=3 front
        # produced only 13 reference points -- far too sparse for a
        # reliable IGD estimate on a disconnected front.
        from problems.ref_dirs_dense import dense_simplex_points
        M = self.n_obj
        if self.which == 9:
            # Disconnected front (DTLZ7-like), 2-objective slice closed form;
            # for M>=3 fall back to dense grid + non-dominated filter.
            if M == 2:
                x = np.linspace(0, 1, n_points)
                y = 2.0 * (2.0 - x / 2.0 * (1.0 + np.sin(3 * np.pi * x)))
                from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                F = np.column_stack([x, y])
                nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
                return F[nd]
            else:
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
        rd = dense_simplex_points(M, n_points=n_points)
        if self.which in _NONLINEAR_LINKAGE:
            rd = rd / np.linalg.norm(rd, axis=1, keepdims=True)
        return rd


def get_lsmop(which, n_obj=3, n_var=None, nk=5):
    """Factory matching pymoo's get_problem-style usage."""
    return LSMOP(which=which, n_obj=n_obj, n_var=n_var, nk=nk)
