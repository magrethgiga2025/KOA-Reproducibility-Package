"""
koa_final_v3.py  --  KOA Final Version 3 (CORRECTED release)
==============================================================

This is the SOLE canonical implementation of KOA. Every table, figure,
ablation and sensitivity script in this repository must import
`KOAFinalV3` from this module. Older variants (koa.py, koa_v1_original.py,
koa_v2_1.py, koa_star.py's algorithm class, koa_star_r2.py, koa_v3.py,
algorithms/archived/*) are kept only for provenance / as shared utility
dependencies (see imports below) and MUST NOT be used to generate any
reported number. See README_FIXES.md for the full list of defects this
revision corrects, each traced to a numbered review comment.

--------------------------------------------------------------------
FIXES APPLIED IN THIS REVISION (see README_FIXES.md for full detail)
--------------------------------------------------------------------

FIX 1 (critical, mandatory rerun) -- SHADE memory was recording success
  against NEWLY RESAMPLED (F, CR) instead of the (F, CR) that actually
  generated the successful offspring. The success-history memory was
  therefore not learning anything real. Fixed by storing the exact
  trial parameters at offspring-generation time (`self._trial_params`)
  and reusing them, unchanged, when recording success in `_advance`.

FIX 2 -- SHADE success criterion Delta_i used a raw, unnormalised sum of
  objectives, implicitly scalarising with equal raw-scale weights (the
  reviewer's and our own diagnosis of the MaF4 failure). Default success
  metric is now a normalised improvement,
      Delta_i = max(0, sum_j (f_j(parent) - f_j(child)) / (s_j + eps))
  where s_j is the current per-objective population range. The old raw
  behaviour is retained as `success_metric="raw"` purely so the
  raw-vs-normalised MaF4 diagnostic experiment recommended by the review
  can be run and reported explicitly, rather than silently changing past
  numbers with no ability to show the before/after.

FIX 3 -- Archive "refresh on stagnation" evaluated fresh LHS candidates
  with an EXTRA, uncounted call to `problem.evaluate`, on top of the
  ordinary N offspring evaluated every generation. This silently
  inflated KOA's true function-evaluation budget relative to every
  baseline. Refresh candidates now REPLACE a subset of the ordinary N
  offspring slots for that generation (chosen before the
  exploration/exploitation loop runs) and are evaluated by pymoo's
  normal evaluator exactly like any other offspring -- so KOA consumes
  exactly N function evaluations per generation, no more, no less, in
  every generation including recovery generations.

FIX 4 -- `refresh_frac=0.0` (the KOA-NoRefresh ablation) still evicted and
  replaced at least one archive member every stagnation event because of
  `n_evict = max(1, round(...))`. NoRefresh now genuinely disables refresh
  (`refresh_frac <= 0` returns no refresh slots at all).

FIX 5 -- Front-connectivity c(t) (Eq. 10) was computed on
  `np.concatenate(fronts)`, i.e. every dominance front up to the one being
  split, rather than on the first non-dominated front as the manuscript's
  Eq. (10) specifies. Now computed on `fronts[0]` only.

FIX 6 -- Geometry-adaptive survival (rho(t) = 0.20 + 0.80*c(t)) had no
  ablation isolating it, despite being one of the three headline
  contributions. Added `rho_mode="adaptive"|"fixed"` and `rho_fixed`
  constructor parameters so a KOA-FixedGeometry(rho=0.2/0.6/1.0) variant
  can be constructed with every other mechanism held identical (see
  algorithms/koa_ablation_final.py).

FIX 7 -- Output convention / 120-vs-100 fairness issue. `_set_optimum`
  used to return the full external archive (up to n_archive=120 points)
  as `res.opt` / `res.F`, giving KOA a larger approximation set than
  every population-based baseline (100). `_set_optimum` now returns the
  non-dominated subset of the FINAL POPULATION (<=pop_size points),
  matching every baseline's output convention. The 120-member archive is
  still maintained internally for reproduction and is still available
  for supplementary analysis via `algorithm.koa_archive`, but it is no
  longer used for the headline HV/IGD numbers.

FIX 8 -- Temporal notation. `SearchStateTracker`'s single `window`
  parameter was silently used both as the ideal-point look-back for
  kappa(t) (manuscript's Gamma) and conflated in prose with Gamma_b
  (recovery patience) or Gamma_c (cooldown) -- three logically distinct
  quantities. KOAFinalV3 now takes three explicitly named parameters --
  `gamma_kappa=5` (convergence-rate look-back), `gamma_b=10` (stagnation
  patience), `gamma_c=15` (recovery cooldown) -- documenting that these
  were already numerically distinct in the running code (5, 10, 15);
  only the naming was ambiguous.

Eq. (3) sign was NOT changed. The implementation intentionally reduces
p_exp under stagnation (biasing toward SHADE exploitation) and relies on
the SEPARATE stagnation-boost mechanism (amplitude x1.6 for `gamma_c`
generations) to reintroduce extra exploration once stagnation is
confirmed. This is the code as actually used to generate existing
results; do not "fix" the sign without a full rerun and an explicit note
in the manuscript (see README_FIXES.md item 2).
"""

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.algorithms.moo.nsga3 import (
    HyperplaneNormalization,
    associate_to_niches,
    calc_niche_count,
    niching,
)
from pymoo.util.ref_dirs import get_reference_directions

from algorithms.koa_v2 import (
    ReferenceVectorArchive,
    SuccessHistoryArchive,
    SearchStateTracker,
    _polynomial_mutation,
    _shade_de_trial,
    _levy_flight,
)
from algorithms.koa_star import (
    _front_connectivity,
    _crowding_truncate,
)


# ======================================================================
# Fix D: scale-adaptive normalisation helpers
# ======================================================================

def _objective_scale_ratio(F_nds):
    """Max/min ratio of per-objective ranges. Returns inf if any range=0."""
    ranges = F_nds.max(axis=0) - F_nds.min(axis=0)
    if (ranges < 1e-10).any():
        return np.inf
    return float(ranges.max() / ranges.min())


def _zscore_normalise_like(F, F_ref):
    """Z-score normalise F using mean/std computed from F_ref."""
    mu = F_ref.mean(axis=0)
    sig = F_ref.std(axis=0)
    sig = np.where(sig < 1e-10, 1.0, sig)
    return (F - mu) / sig


# ======================================================================
# FIX 3 + FIX 4: archive refresh that costs zero extra evaluations
# ======================================================================

def _select_refresh_slots(archive, n_offspring, refresh_frac, rng):
    """
    Decide which archive members to evict and which OFFSPRING SLOTS
    (indices into the N ordinary offspring generated this generation)
    should be replaced by fresh LHS samples instead of an
    exploration/exploitation trial.

    This performs NO evaluation itself (Fix 3): the caller must generate
    fresh X for the returned offspring slots and let them go through the
    ordinary per-generation evaluation exactly like every other offspring,
    so no extra function evaluations are consumed.

    Returns (slot_idx, evict_idx) or (None, None) if refresh is disabled
    or not currently possible.
    """
    if refresh_frac <= 0:                      # FIX 4: genuinely off
        return None, None
    if archive.X is None or len(archive.X) < 4:
        return None, None

    n_evict = max(1, int(np.round(len(archive.X) * refresh_frac)))
    n_evict = min(n_evict, n_offspring)         # never exceed offspring count
    cd = calc_crowding_distance(archive.F)
    evict_idx = np.argsort(cd)[:n_evict]

    slot_idx = rng.choice(n_offspring, size=n_evict, replace=False)
    return slot_idx, evict_idx


# ======================================================================
# Geometry-adaptive truncate with Fix D (scale) + Fix 5 (connectivity
# front) + Fix 6 (fixed-rho ablation hook) integrated
# ======================================================================

def _geometry_adaptive_truncate_v3(F, n_survive, ref_dirs,
                                    norm: HyperplaneNormalization,
                                    rng, theta=0.15, scale_thresh=50.0,
                                    rho_mode="adaptive", rho_fixed=0.6):
    """
    NSGA-III niche-balanced survival with:
      - geometry-adaptive (or fixed, for ablation) niche fraction
      - scale-adaptive normalisation fallback (Fix D)
      - connectivity computed on the first non-dominated front only
        (Fix 5), matching manuscript Eq. (10)
    """
    fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)
    if len(fronts) == 0:
        return np.arange(min(n_survive, len(F)))

    until_last = (np.concatenate(fronts[:-1]).astype(int)
                  if len(fronts) > 1 else np.array([], dtype=int))
    last_front = np.asarray(fronts[-1], dtype=int)

    if len(until_last) >= n_survive:
        return until_last[:n_survive]

    n_remaining = n_survive - len(until_last)

    # ---- FIX 6: geometry-adaptive vs fixed niche fraction ------------
    if rho_mode == "fixed":
        niche_frac = float(np.clip(rho_fixed, 0.0, 1.0))
    else:
        # ---- FIX 5: connectivity uses ONLY the first non-dominated
        # front, matching the manuscript's Eq. (10) definition of c(t)
        # on the current non-dominated set, not every dominance front
        # up to the split point.
        first_front = np.asarray(fronts[0], dtype=int)
        connectivity = _front_connectivity(F[first_front], threshold_frac=theta)
        niche_frac = float(np.clip(0.20 + 0.80 * connectivity, 0.20, 1.00))

    n_niche = int(round(n_remaining * niche_frac))
    n_crowd = n_remaining - n_niche

    # Fix D: choose normalisation based on objective scale ratio.
    # (Association still uses the full nds set -- until_last + last_front
    #  -- for niche counting, independent of the connectivity fix above.)
    nds_idx = np.concatenate(fronts).astype(int)
    F_nds = F[nds_idx]
    scale_ratio = _objective_scale_ratio(F_nds)

    if scale_ratio > scale_thresh and len(F_nds) >= 3:
        F_assoc = _zscore_normalise_like(F, F_nds)
        ideal_use = F_assoc.min(axis=0)
        nadir_use = F_assoc.max(axis=0)
    else:
        F_assoc = F
        ideal_use = norm.ideal_point
        nadir_use = norm.nadir_point

    try:
        niche_of_ind, dist_to_niche, _ = associate_to_niches(
            F_assoc, ref_dirs, ideal_use, nadir_use
        )
    except Exception:
        return _crowding_truncate(F, n_survive)

    niche_count = (calc_niche_count(len(ref_dirs), niche_of_ind[until_last])
                   if len(until_last) > 0
                   else np.zeros(len(ref_dirs), dtype=int))

    chosen = set()
    if n_niche > 0:
        try:
            surv = niching(
                F[last_front], n_niche, niche_count,
                niche_of_ind[last_front], dist_to_niche[last_front],
                random_state=rng,
            )
            chosen.update(surv.tolist() if hasattr(surv, 'tolist') else list(surv))
        except Exception:
            pass

    if n_crowd > 0:
        remaining = [i for i in range(len(last_front)) if i not in chosen]
        if remaining:
            cd = calc_crowding_distance(F[last_front[remaining]])
            order = np.argsort(-cd)
            for i in order[:n_crowd]:
                chosen.add(remaining[i])

    survivors_in_last = np.array(sorted(chosen), dtype=int)
    return np.concatenate([
        until_last, last_front[survivors_in_last]
    ]).astype(int)


# ======================================================================
# KOA Final v3 (corrected)
# ======================================================================

class KOAFinalV3(Algorithm):
    """
    KOA Final Version 3 -- Corrected implementation release.

    Parameters
    ----------
    pop_size       : population size (default 100)
    n_archive      : archive capacity (default 120), used internally for
                     reproduction only -- final reported output is the
                     population's non-dominated subset (Fix 7)
    seed           : random seed
    init_X         : optional (pop_size, n_var) array of pre-generated
                     initial decision vectors. When given, KOA uses this
                     EXACT initial population instead of drawing its own
                     LHS sample, so it can share common-random-numbers
                     initialisation with every baseline algorithm for a
                     given (problem, seed) pair.
    alpha0         : base velocity step amplitude (default 0.30)
    A              : sinusoidal amplitude (default 0.15)
    tau            : oscillation period in gens (default 50)
    shade_memory   : SHADE history slots (default 10)
    eta_PI         : polynomial mutation index (default 20)
    gamma_kappa    : generations of look-back for the convergence-rate
                     signal kappa(t) (default 5)
    gamma_b        : consecutive non-improving generations before a
                     stagnation/recovery event fires (default 10)
    gamma_c        : cooldown generations after a recovery event before
                     another one can fire (default 15)
    pop_seed_gens  : gens to use population (not archive) as SHADE pool
                     -- Fix E, default 30
    refresh_frac   : fraction of archive refreshed on stagnation,
                     replacing that many ordinary offspring slots rather
                     than consuming extra evaluations -- Fix C/3,
                     default 0.25. refresh_frac<=0 disables refresh
                     entirely (Fix 4 / KOA-NoRefresh).
    scale_thresh   : scale ratio threshold for z-score norm -- Fix D,
                     default 50.0
    theta          : connectivity threshold for niche fraction (0.15)
    rho_mode       : "adaptive" (default) uses rho(t)=0.20+0.80*c(t);
                     "fixed" uses a constant `rho_fixed` for every
                     generation with every other mechanism unchanged --
                     this is the KOA-FixedGeometry / KOA-NoGeometry
                     ablation hook (Fix 6).
    rho_fixed      : constant niche fraction used when rho_mode="fixed"
    lambda_delta   : diversity-push weight in Eq. (3) (default 0.20)
    lambda_kappa   : convergence-pull weight in Eq. (3) (default 0.15)
    a_start        : exploration weight at t=0 in Eq. (3) (default 0.40)
    a_end          : exploration weight at t=T in Eq. (3) (default 0.10)
    p_exp_min      : lower clip bound for p_exp (default 0.05)
    p_exp_max      : upper clip bound for p_exp (default 0.80)
    success_metric : "normalized" (default, Fix 2) or "raw" (legacy,
                     kept ONLY to reproduce the raw-vs-normalised MaF4
                     diagnostic comparison)
    eps            : numerical stabiliser (1e-9)
    """

    def __init__(self,
                 pop_size=100, n_archive=120, seed=None, init_X=None,
                 alpha0=0.30, A=0.15, tau=50,
                 shade_memory=10,
                 eta_PI=20,
                 gamma_kappa=5, gamma_b=10, gamma_c=15,
                 pop_seed_gens=30,       # Fix E
                 refresh_frac=0.25,      # Fix C / Fix 3 / Fix 4
                 scale_thresh=50.0,      # Fix D
                 theta=0.15,
                 rho_mode="adaptive",    # Fix 6
                 rho_fixed=0.6,
                 lambda_delta=0.20,
                 lambda_kappa=0.15,
                 a_start=0.40,
                 a_end=0.10,
                 p_exp_min=0.05,
                 p_exp_max=0.80,
                 success_metric="normalized",  # Fix 2
                 levy_prob=0.15,
                 de_scale=0.1,
                 eps=1e-9,
                 **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.n_archive = n_archive
        self.seed = seed
        self.init_X = init_X
        self.alpha0 = alpha0
        self.A = A
        self.tau = tau
        self.shade_memory = shade_memory
        self.eta_PI = eta_PI

        # Fix 8: three explicitly distinct temporal constants
        self.gamma_kappa = gamma_kappa
        self.gamma_b = gamma_b
        self.gamma_c = gamma_c
        # backward-compatible aliases used elsewhere in this file
        self.stag_patience = gamma_b
        self.stag_cooldown = gamma_c

        self.pop_seed_gens = pop_seed_gens
        self.refresh_frac = refresh_frac
        self.scale_thresh = scale_thresh
        self.theta = theta
        self.rho_mode = rho_mode
        self.rho_fixed = rho_fixed
        self.lambda_delta = lambda_delta
        self.lambda_kappa = lambda_kappa
        self.a_start = a_start
        self.a_end = a_end
        self.p_exp_min = p_exp_min
        self.p_exp_max = p_exp_max
        self.success_metric = success_metric
        self.levy_prob = levy_prob   # was hardcoded 0.15
        self.de_scale = de_scale     # was hardcoded 0.1 in SuccessHistoryArchive
        self.eps = eps

        self.rng = np.random.default_rng(seed)
        self._boost_gens = 0
        self._last_branch = None
        self._last_F_parent = None
        self._trial_params = None      # Fix 1: (F_de, CR_de) actually used per offspring
        self.velocities = None
        self._pending_refresh = None   # (slot_idx, evict_idx) from Fix 3/4

        # Diagnostic-only: p_exp(t) trajectory extremes actually visited
        # during this run. Exists so parameter-sensitivity studies (e.g.
        # varying p_exp_min/p_exp_max) can report whether the configured
        # clip bounds ever actually bound anything on a given problem,
        # rather than silently producing identical runs when they don't.
        # Read via `algorithm.p_exp_trace` after a run.
        self.p_exp_trace = {"min": None, "max": None, "sum": 0.0, "n": 0}

    # ------------------------------------------------------------------
    def _setup(self, problem, **kwargs):
        self.problem = problem
        n_obj = problem.n_obj
        self.koa_archive = ReferenceVectorArchive(self.n_archive, n_obj)
        self.shade = SuccessHistoryArchive(self.shade_memory)
        self.tracker = SearchStateTracker(
            window=self.gamma_kappa,
            stagnation_patience=self.gamma_b,
            cooldown=self.gamma_c,
        )
        self._norm = HyperplaneNormalization(n_obj)

        n_part = self.pop_size - 1 if n_obj == 2 else 12
        self._surv_ref_dirs = get_reference_directions(
            "das-dennis", n_obj, n_partitions=n_part
        )

    # ------------------------------------------------------------------
    def _initialize_infill(self):
        if self.init_X is not None:
            # Shared common-random-numbers initial population.
            X = np.asarray(self.init_X, dtype=float)
            assert X.shape[0] == self.pop_size, (
                f"init_X has {X.shape[0]} rows, expected pop_size={self.pop_size}"
            )
        else:
            sampling = LHS()
            X = sampling.do(self.problem, self.pop_size,
                             random_state=self.rng).get("X")
        self.velocities = np.zeros_like(X)
        return Population.new(X=X)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        F = self.pop.get("F")
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        self._norm.update(F, nds=nds)
        self.koa_archive.update(self.pop.get("X"), F)
        self.tracker.update(
            F, len(self.koa_archive.X) if self.koa_archive.X is not None else 0
        )

    # ------------------------------------------------------------------
    def _state_probabilities(self, t, T):
        """
        Eq. (3): p_exp(t) = a_start*(1-progress) + a_end*progress
                            + lambda_delta*(1-delta_ratio)
                            - lambda_kappa*(1-kappa)
        Coefficients are constructor parameters (were hardcoded), so the
        parameter-sensitivity study can vary them directly on this class
        instead of monkey-patching an older algorithm variant.
        """
        conv_rate = self.tracker.convergence_rate()
        div_ratio = self.tracker.diversity_ratio()
        progress = float(np.clip(t / max(T, 1), 0.0, 1.0))

        p_exp = self.a_start * (1.0 - progress) + self.a_end * progress
        p_exp += (1.0 - div_ratio) * self.lambda_delta   # diversity push
        p_exp -= (1.0 - conv_rate) * self.lambda_kappa   # convergence pull
        p_exp = float(np.clip(p_exp, self.p_exp_min, self.p_exp_max))

        t = self.p_exp_trace
        t["min"] = p_exp if t["min"] is None else min(t["min"], p_exp)
        t["max"] = p_exp if t["max"] is None else max(t["max"], p_exp)
        t["sum"] += p_exp
        t["n"] += 1
        return p_exp, None

    # ------------------------------------------------------------------
    def _infill(self):
        problem = self.problem
        N, D = self.pop_size, problem.n_var
        xl, xu = problem.xl, problem.xu
        rng = self.rng

        t = self.n_gen if self.n_gen is not None else 0
        T = (self.termination.n_max_gen
             if hasattr(self.termination, 'n_max_gen') else 300)

        # ── Fix 3/4: stagnation recovery via slot substitution, NOT
        #    extra evaluations. Decide slots/evictions now (no eval yet).
        self._pending_refresh = None
        if self.tracker.is_stagnant():
            self.tracker.trigger_recovery()
            self._boost_gens = 3
            slot_idx, evict_idx = _select_refresh_slots(
                self.koa_archive, N, self.refresh_frac, rng
            )
            if slot_idx is not None:
                self._pending_refresh = (slot_idx, evict_idx)

        boosting = self._boost_gens > 0
        if boosting:
            self._boost_gens -= 1

        p_exp, _ = self._state_probabilities(t, T)

        X = self.pop.get("X")
        self._last_F_parent = self.pop.get("F").copy()
        archive_X = (self.koa_archive.X
                     if self.koa_archive.X is not None else X)

        alpha_t = self.alpha0 + self.A * np.sin(2.0 * np.pi * t / self.tau)
        if boosting:
            alpha_t *= 1.6

        # ── Fix E: use population as SHADE pool during cold-start ─────
        use_pop_as_shade_pool = (t <= self.pop_seed_gens)

        refresh_slots = (set(self._pending_refresh[0].tolist())
                          if self._pending_refresh is not None else set())

        s = rng.random(N)
        X_new = np.zeros_like(X)
        branch = np.full(N, -1, dtype=int)  # -1 = refresh slot (Fix 3)
        # (F_de, CR_de) actually used to build each exploitation offspring
        # -- Fix 1: this is what must be reused (not resampled) later.
        trial_params = np.full((N, 2), np.nan)

        # Fresh LHS candidates for refresh slots (Fix 3): sampled now, but
        # NOT evaluated here -- they are returned as ordinary, unevaluated
        # offspring and go through pymoo's normal per-generation evaluator,
        # so they cost exactly the 1 FE per slot that any offspring costs.
        if refresh_slots:
            fresh = LHS().do(problem, len(refresh_slots),
                              random_state=rng).get("X")
            fresh_iter = iter(fresh)

        for i in range(N):
            if i in refresh_slots:
                X_new[i] = next(fresh_iter)
                continue

            if s[i] < p_exp:
                # ── Exploration: archive-seeded velocity ───────────────
                branch[i] = 0
                if len(archive_X) > 0:
                    x_guide = archive_X[rng.integers(0, len(archive_X))]
                else:
                    others = np.delete(np.arange(N), i)
                    x_guide = X[rng.choice(others)]

                diff = x_guide - X[i]
                norm = np.linalg.norm(diff) + self.eps
                self.velocities[i] = (0.6 * self.velocities[i]
                                       + alpha_t * diff / norm)
                if rng.random() < self.levy_prob:
                    self.velocities[i] += _levy_flight(D, 1.5, rng, size=1)[0]
                x_prime = np.clip(X[i] + self.velocities[i], xl, xu)

            else:
                # ── Exploitation: SHADE with Fix E pool selection ──────
                branch[i] = 1
                F_de, CR_de = self.shade.sample_params(rng, de_scale=self.de_scale)
                trial_params[i] = (F_de, CR_de)   # Fix 1: remember these

                if use_pop_as_shade_pool:
                    pool = X
                    x_base = X[rng.integers(0, N)]
                else:
                    if len(archive_X) > 0:
                        nb = self.koa_archive.neighborhood_sample(
                            i, min(3, len(archive_X)), rng)
                        pool = nb if nb is not None else archive_X
                        x_base = pool[rng.integers(0, len(pool))]
                    else:
                        pool = X
                        x_base = X[i]

                x_prime = _shade_de_trial(
                    X[i], x_base, pool, F_de, CR_de, xl, xu, rng
                )

            if np.any(~np.isfinite(x_prime)):
                x_prime = X[i].copy()
            x_prime = np.clip(x_prime, xl, xu)
            x_prime = _polynomial_mutation(
                x_prime, prob=1.0 / D, eta=self.eta_PI,
                xl=xl, xu=xu, rng=rng
            )
            X_new[i] = x_prime

        self._last_branch = branch
        self._trial_params = trial_params
        return Population.new(X=X_new)

    # ------------------------------------------------------------------
    def _advance(self, infills=None, **kwargs):
        P_old = self.pop
        P_new = infills

        # ── Fix 3/4: apply pending archive refresh using the ALREADY
        # evaluated refresh-slot offspring (P_new), no extra evaluation.
        if self._pending_refresh is not None:
            slot_idx, evict_idx = self._pending_refresh
            self._pending_refresh = None

            X_refresh = P_new.get("X")[slot_idx]
            F_refresh = P_new.get("F")[slot_idx]

            keep_mask = np.ones(len(self.koa_archive.X), dtype=bool)
            keep_mask[evict_idx] = False
            if keep_mask.any():
                self.koa_archive.X = self.koa_archive.X[keep_mask]
                self.koa_archive.F = self.koa_archive.F[keep_mask]
            else:
                self.koa_archive.X = None
                self.koa_archive.F = None
            self.koa_archive.update(X_refresh, F_refresh)

        X_union = np.vstack([P_old.get("X"), P_new.get("X")])
        F_union = np.vstack([P_old.get("F"), P_new.get("F")])

        nds = NonDominatedSorting().do(F_union, only_non_dominated_front=True)
        self._norm.update(F_union, nds=nds)

        # ── Survival with Fix D (scale-adaptive norm.) + Fix 5
        #    (connectivity on first front) + Fix 6 (fixed-rho ablation) ──
        if self._norm.nadir_point is not None:
            keep = _geometry_adaptive_truncate_v3(
                F_union, self.pop_size,
                self._surv_ref_dirs, self._norm, self.rng,
                theta=self.theta,
                scale_thresh=self.scale_thresh,
                rho_mode=self.rho_mode,
                rho_fixed=self.rho_fixed,
            )
        else:
            keep = _crowding_truncate(F_union, self.pop_size)

        self.pop = Population.new(X=X_union[keep], F=F_union[keep])

        # ── FIX 1 + FIX 2: SHADE memory update ──────────────────────────
        # Use the EXACT (F_de, CR_de) that generated each offspring (Fix 1,
        # not a freshly resampled pair), and a normalised improvement
        # score by default (Fix 2), with the legacy raw-sum score kept
        # only for the raw-vs-normalised MaF4 diagnostic.
        if self._last_branch is not None and self._last_F_parent is not None:
            exploit_idx = np.where(self._last_branch == 1)[0]
            old_F = self._last_F_parent
            new_F = P_new.get("F")

            if self.success_metric == "normalized" and len(exploit_idx) > 0:
                pop_F = self.pop.get("F")
                s_j = np.maximum(pop_F.max(axis=0) - pop_F.min(axis=0), self.eps)
            else:
                s_j = None

            for i in exploit_idx:
                if i >= len(old_F) or i >= len(new_F):
                    continue
                if self.success_metric == "raw":
                    delta = float(max(0.0, old_F[i].sum() - new_F[i].sum()))
                else:
                    delta = float(max(
                        0.0, np.sum((old_F[i] - new_F[i]) / s_j)
                    ))
                if delta > 0:
                    F_de, CR_de = self._trial_params[i]   # Fix 1: reuse, don't resample
                    self.shade.record_success(F_de, CR_de, delta)
        self.shade.update_memory()

        self.koa_archive.update(P_new.get("X"), P_new.get("F"))
        n_arch = (len(self.koa_archive.X)
                  if self.koa_archive.X is not None else 0)
        self.tracker.update(self.pop.get("F"), n_arch)

    # ------------------------------------------------------------------
    def _set_optimum(self, **kwargs):
        """
        FIX 7: report the non-dominated subset of the FINAL POPULATION
        (<=pop_size points), matching every population-based baseline's
        output convention. The larger 120-member external archive is
        still built and kept (`self.koa_archive`) for reproduction and
        for anyone who explicitly wants to inspect/report it separately,
        but it is no longer what `res.F` / `res.opt` returns.
        """
        F = self.pop.get("F")
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        self.opt = self.pop[nd]
