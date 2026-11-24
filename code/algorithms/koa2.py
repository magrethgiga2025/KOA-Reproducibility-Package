"""
KOA-II: niche-targeted archive guidance with survival-credit operator allocation.

Redesign of KOA (v3) after the Information Sciences rejection. The design keeps
only what the v3 ablation supported and replaces what it did not:

  KEPT and sharpened
    M1  Archive-guided reproduction. Offspring are pulled towards archive members,
        and the guide is chosen preferentially from archive members whose reference
        niche is *under-populated* in the current population ("niche-targeted
        guidance"). The archive therefore steers search towards the gaps of the
        population, not just towards "good" points.

  REDESIGNED (replaces the hand-written p_exp formula, Eq. 3 of v3)
    M2  Survival-credit operator allocation. Two variation operators compete:
          - G: archive-guided DE   v = x + F(g - x) + F(r1 - r2)
          - S: SBX between population members (population-based recombination)
        The share of offspring produced by G, p_G(t), is updated each generation
        from how many of each operator's offspring actually SURVIVED environmental
        selection, using adaptive pursuit (Thierens, GECCO 2005): quality estimates
        q_G, q_S track the survival rates (learning rate q_lr) and p_G moves by beta
        towards p_max if q_G > q_S, otherwise towards p_min.
        No convergence/diversity formula, no stagnation counters.

    M3  Survival-based parameter adaptation. The SHADE memory for (F, CR) is updated
        with the parameters of guided offspring that survived selection. This removes
        the scalarised "improvement" signal of v3 (and with it the raw-vs-normalised
        objective-scale problem that was suspected on MaF4).

  REMOVED (not supported by the v3 ablation)
    - geometry-adaptive niche fraction rho(t)  (v3: indistinguishable from rho=1)
    - stagnation archive refresh                 (v3: removing it improved ranks)
    - sinusoidal step amplitude, velocities, Levy flights, stagnation boosts

  Environmental selection is standard NSGA-III reference-direction survival with
  hyperplane normalisation (pymoo implementation), applied identically to the
  population and (as truncation) to the archive. This is stated as standard, not
  claimed as a contribution.

Parameters (all others are standard operator constants):
    archive_ratio  |A| / N                         default 1.0
    beta           pursuit rate of p_G              default 0.1
    q_lr           learning rate of q_G, q_S        default 0.3
    p_bounds       [p_min, p_max] for p_G           default (0.1, 0.9)
    H              SHADE memory size                default 5
    repair         "clip" (default; same as PlatEMO and every pymoo baseline)
                   | "midpoint" (JADE-style). Note: on DTLZ6 the optimum lies on the
                   lower bound, so clipping helps there; report this in the paper.

Ablation switches (see experiments/config.py):
    guide      "niche" (default) | "uniform" | "population"
    alloc      "adaptive" (default) | "fixed"   (fixed uses p_fixed)
    p_fixed    used when alloc="fixed" (0.0 = SBX only, 1.0 = guided DE only)
    adapt_params True (default) | False  (False -> F=0.5, CR=0.9 fixed)

Per-generation diagnostics are stored in `self.trace` (dict of lists).
"""
import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival, associate_to_niches
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.display.multi import MultiObjectiveOutput


# ----------------------------------------------------------------------
# Variation operators (vectorised)
# ----------------------------------------------------------------------
def repair_midpoint(V, X, xl, xu):
    """Out-of-bound components are placed midway between parent and bound."""
    V = np.where(V < xl, (X + xl) / 2.0, V)
    V = np.where(V > xu, (X + xu) / 2.0, V)
    return V


def sbx_one_child(X1, X2, xl, xu, rng, eta=20.0, prob_var=0.5):
    """Simulated binary crossover (Deb & Agrawal 1995); returns one child per pair."""
    n, d = X1.shape
    y1, y2 = np.minimum(X1, X2), np.maximum(X1, X2)
    span = np.maximum(y2 - y1, 1e-14)
    u = rng.random((n, d))

    def beta_q(beta):
        alpha = 2.0 - np.power(beta, -(eta + 1.0))
        mask = u <= 1.0 / alpha
        bq = np.empty_like(u)
        bq[mask] = np.power(u[mask] * alpha[mask], 1.0 / (eta + 1.0))
        bq[~mask] = np.power(1.0 / (2.0 - u[~mask] * alpha[~mask]), 1.0 / (eta + 1.0))
        return bq

    beta_l = 1.0 + 2.0 * (y1 - xl) / span
    beta_u = 1.0 + 2.0 * (xu - y2) / span
    c1 = 0.5 * ((y1 + y2) - beta_q(beta_l) * span)
    c2 = 0.5 * ((y1 + y2) + beta_q(beta_u) * span)
    child = np.where(rng.random((n, d)) < 0.5, c1, c2)
    do = (rng.random((n, d)) < prob_var) & (np.abs(X1 - X2) > 1e-14)
    C = np.where(do, child, X1)
    return np.clip(C, xl, xu)


def polynomial_mutation(X, xl, xu, rng, eta=20.0, prob=None):
    n, d = X.shape
    prob = 1.0 / d if prob is None else prob
    mask = rng.random((n, d)) < prob
    if not mask.any():
        return X
    Y = X.copy()
    span = np.maximum(xu - xl, 1e-14)
    d1 = (Y - xl) / span
    d2 = (xu - Y) / span
    u = rng.random((n, d))
    mp = 1.0 / (eta + 1.0)
    low = u < 0.5
    xy = np.where(low, 1.0 - d1, 1.0 - d2)
    val_low = 2.0 * u + (1.0 - 2.0 * u) * np.power(xy, eta + 1.0)
    val_high = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * np.power(xy, eta + 1.0)
    dq = np.where(low, np.power(val_low, mp) - 1.0, 1.0 - np.power(val_high, mp))
    Y = np.where(mask, Y + dq * span, Y)
    return np.clip(Y, xl, xu)


class SuccessHistory:
    """SHADE-style memory (Tanabe & Fukunaga 2013) with survival as success."""

    def __init__(self, H=5, rng=None):
        self.H = H
        self.MF = np.full(H, 0.5)
        self.MCR = np.full(H, 0.5)
        self.k = 0
        self.rng = rng

    def sample(self, n):
        rng = self.rng
        r = rng.integers(0, self.H, n)
        CR = np.clip(rng.normal(self.MCR[r], 0.1), 0.0, 1.0)
        F = np.empty(n)
        todo = np.arange(n)
        while len(todo):
            f = self.MF[r[todo]] + 0.1 * np.tan(np.pi * (rng.random(len(todo)) - 0.5))
            ok = f > 0
            F[todo[ok]] = np.minimum(f[ok], 1.0)
            todo = todo[~ok]
        return F, CR

    def update(self, SF, SCR):
        if len(SF) == 0:
            return
        SF, SCR = np.asarray(SF), np.asarray(SCR)
        self.MF[self.k] = np.sum(SF ** 2) / max(np.sum(SF), 1e-14)   # Lehmer mean
        self.MCR[self.k] = np.mean(SCR)
        self.k = (self.k + 1) % self.H


# ----------------------------------------------------------------------
# Algorithm
# ----------------------------------------------------------------------
class KOA2(Algorithm):

    def __init__(self, ref_dirs, pop_size=None, sampling=None,
                 archive_ratio=1.0, beta=0.1, p_bounds=(0.1, 0.9), H=5,
                 guide="niche", alloc="adaptive", p_fixed=0.5, adapt_params=True,
                 p_init=0.5, eta_c=20.0, eta_m=20.0, repair="clip", q_lr=0.3,
                 output=MultiObjectiveOutput(), **kwargs):
        super().__init__(output=output, **kwargs)
        self.ref_dirs = np.asarray(ref_dirs)
        self.pop_size = len(self.ref_dirs) if pop_size is None else pop_size
        self.sampling = LHS() if sampling is None else sampling
        self.initialization = Initialization(self.sampling)
        self.archive_size = max(2, int(round(archive_ratio * self.pop_size)))
        self.beta, self.p_bounds, self.H = beta, p_bounds, H
        assert guide in ("niche", "uniform", "population")
        assert alloc in ("adaptive", "fixed")
        self.guide, self.alloc, self.p_fixed = guide, alloc, p_fixed
        self.adapt_params = adapt_params
        self.p_init, self.eta_c, self.eta_m = p_init, eta_c, eta_m
        self.repair, self.q_lr = repair, q_lr

    # ------------------------------------------------------------------
    def _setup(self, problem, **kwargs):
        self.rng = np.random.default_rng(self.seed)
        self.survival = ReferenceDirectionSurvival(self.ref_dirs)
        self.arch_survival = ReferenceDirectionSurvival(self.ref_dirs)
        self.memory = SuccessHistory(self.H, self.rng)
        self.p_G = self.p_fixed if self.alloc == "fixed" else self.p_init
        self.q = np.array([0.5, 0.5])   # quality estimates of (G, S)
        self.trace = {k: [] for k in
                      ("p_G", "surv_G", "surv_S", "MF", "MCR", "arch_size", "empty_niches")}

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size,
                                      algorithm=self, random_state=self.rng)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(self.problem, infills, n_survive=len(infills),
                                    random_state=self.rng)
        self.elite = self._truncate_archive(self._nondominated(self.pop))

    # ------------------------------------------------------------------
    @staticmethod
    def _nondominated(pop):
        F = pop.get("F")
        return pop[NonDominatedSorting().do(F, only_non_dominated_front=True)]

    def _truncate_archive(self, A):
        if len(A) <= self.archive_size:
            return A
        I = self.arch_survival.do(self.problem, A, n_survive=self.archive_size,
                                  return_indices=True, random_state=self.rng)
        return A[I]

    def _guide_probabilities(self, G_F):
        """Probability of choosing each candidate guide (M1)."""
        n = len(G_F)
        if self.guide == "uniform" or n == 1:
            return np.full(n, 1.0 / n)
        norm = self.survival.norm
        niche_g, _, _ = associate_to_niches(G_F, self.ref_dirs,
                                            norm.ideal_point, norm.nadir_point)
        niche_p = self.pop.get("niche")
        count = np.bincount(niche_p.astype(int), minlength=len(self.ref_dirs))
        w = 1.0 / (1.0 + count[niche_g])
        return w / w.sum()

    # ------------------------------------------------------------------
    def _infill(self):
        rng, N = self.rng, self.pop_size
        xl, xu = self.problem.xl, self.problem.xu
        X = self.pop.get("X")

        use_G = rng.random(N) < self.p_G
        iG, iS = np.where(use_G)[0], np.where(~use_G)[0]
        Xoff = np.empty_like(X)
        F_used = np.full(N, np.nan)
        CR_used = np.full(N, np.nan)

        # --- G: archive-guided DE ------------------------------------
        if len(iG):
            if self.guide == "population":
                G_X, G_F = X, self.pop.get("F")
            else:
                G_X, G_F = self.elite.get("X"), self.elite.get("F")
            p = self._guide_probabilities(G_F)
            g = G_X[rng.choice(len(G_X), size=len(iG), p=p)]

            if self.adapt_params:
                Fi, CRi = self.memory.sample(len(iG))
            else:
                Fi, CRi = np.full(len(iG), 0.5), np.full(len(iG), 0.9)

            pool = np.vstack([X, self.elite.get("X")]) if self.guide != "population" else X
            r1 = rng.integers(0, N, len(iG))
            r2 = rng.integers(0, len(pool), len(iG))
            Xi = X[iG]
            V = Xi + Fi[:, None] * (g - Xi) + Fi[:, None] * (X[r1] - pool[r2])
            V = repair_midpoint(V, Xi, xl, xu) if self.repair == "midpoint" else np.clip(V, xl, xu)
            mask = rng.random(Xi.shape) < CRi[:, None]
            mask[np.arange(len(iG)), rng.integers(0, X.shape[1], len(iG))] = True
            Xoff[iG] = np.where(mask, V, Xi)
            F_used[iG], CR_used[iG] = Fi, CRi

        # --- S: SBX between population members -------------------------
        if len(iS):
            mates = rng.integers(0, N, len(iS))
            Xoff[iS] = sbx_one_child(X[iS], X[mates], xl, xu, rng, eta=self.eta_c)

        Xoff = polynomial_mutation(Xoff, xl, xu, rng, eta=self.eta_m)
        off = Population.new(X=Xoff)
        off.set("op_G", use_G, "F_de", F_used, "CR_de", CR_used)
        self._n_G, self._n_S = len(iG), len(iS)
        return off

    # ------------------------------------------------------------------
    def _advance(self, infills=None, **kwargs):
        off = infills
        N = self.pop_size
        n0 = len(self.pop)
        self.pop.set("op_G", np.zeros(n0, bool), "is_off", np.zeros(n0, bool),
                     "F_de", np.full(n0, np.nan), "CR_de", np.full(n0, np.nan))
        off.set("is_off", np.ones(len(off), bool))
        merged = Population.merge(self.pop, off)
        # survivors keep their custom attributes (is_off, op_G, F_de, CR_de)
        self.pop = self.survival.do(self.problem, merged, n_survive=N,
                                    random_state=self.rng)

        # --- credit assignment (M2, M3) --------------------------------
        s_off = self.pop.get("is_off").astype(bool)
        s_G = self.pop.get("op_G").astype(bool) & s_off
        opG = off.get("op_G").astype(bool)
        nG, nS = int(opG.sum()), int((~opG).sum())
        rate_G = s_G.sum() / nG if nG else np.nan
        rate_S = (s_off & ~s_G).sum() / nS if nS else np.nan

        if self.alloc == "adaptive":
            # Adaptive pursuit (Thierens 2005) on survival rates: the operator whose
            # offspring survive more often is pursued towards p_max, the other decays
            # towards p_min.
            for j, r in enumerate((rate_G, rate_S)):
                if not np.isnan(r):
                    self.q[j] += self.q_lr * (r - self.q[j])
            p_min, p_max = self.p_bounds
            goal = p_max if self.q[0] > self.q[1] else p_min if self.q[0] < self.q[1] else self.p_G
            self.p_G = float(self.p_G + self.beta * (goal - self.p_G))

        if self.adapt_params and s_G.any():
            self.memory.update(self.pop.get("F_de")[s_G], self.pop.get("CR_de")[s_G])

        # --- archive update ---------------------------------------------
        cand = Population.merge(self.elite, self._nondominated(off))
        self.elite = self._truncate_archive(self._nondominated(cand))

        # --- diagnostics ------------------------------------------------
        niche = self.pop.get("niche").astype(int)
        t = self.trace
        t["p_G"].append(self.p_G)
        t["surv_G"].append(rate_G)
        t["surv_S"].append(rate_S)
        t["MF"].append(float(self.memory.MF.mean()))
        t["MCR"].append(float(self.memory.MCR.mean()))
        t["arch_size"].append(len(self.elite))
        t["empty_niches"].append(int(len(self.ref_dirs) - len(np.unique(niche))))

    def _set_optimum(self, **kwargs):
        # Same output convention as every baseline: non-dominated subset of the population.
        self.opt = self._nondominated(self.pop)
