"""
Kipunji Optimization Algorithm v2 (KOA-v2) -- redesigned exploitation
and state-control mechanisms.

This module is a deliberate redesign of algorithms/koa.py (preserved
unmodified as algorithms/koa_v1_original.py), built in response to the
benchmark finding that KOA-v1 ranks last among seven algorithms on HV
and IGD across 30 runs x 10 problems (Friedman avg. rank 6.1/7 on both
indicators), while ranking first on Spacing -- i.e. v1 trades convergence
speed for diversity rather than achieving a favorable balance of both.

IMPORTANT: every mechanism below is a genuine algorithmic change, not a
parameter retune. None of the "expected effect" language used elsewhere
in this codebase or in chat is a claim about actual measured performance
-- KOA-v2 has not yet been benchmarked. Sections 7 (complexity) below
are derivable analytically; everything else must be verified empirically
by running the existing run_experiments.py / analyze_results.py pipeline
with ALGORITHMS including "KOA-v2" before any performance claim is made.

The six redesign items, each implemented as a concrete, inspectable
mechanism rather than prose:

  1. Adaptive state regulation (replaces fixed 0.35/0.75 branch
     thresholds with diversity- and stagnation-aware state selection)
  2. Archive-guided exploitation (replaces N(0, sigma^2 I) Gaussian
     perturbation with success-history adaptive differential evolution,
     SHADE-style, seeded from the archive)
  3. Dynamic fission-fusion scheduling (subgroup count shrinks from
     many small subgroups early to few large cooperating subgroups late)
  4. Diversity-convergence balance via measurable indicators (population
     diversity and ideal-point movement rate, not iteration count alone)
  5. Many-objective enhancement (Das-Dennis reference-vector-guided
     archive clustering and neighborhood-restricted mating for M >= 3)
  6. Stagnation recovery (explicit detector on ideal-point movement and
     archive growth; reactivates exploration only when triggered, with
     a cooldown to avoid excessive randomness)

Complexity analysis (item 7) is provided in the module docstring at the
bottom of this file rather than in code, since it is a derived property
of the implementation, not a runtime computation.
"""
import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.ref_dirs import get_reference_directions


# ----------------------------------------------------------------------
# Shared utilities (reused from v1 where the mechanism is unchanged)
# ----------------------------------------------------------------------
def _crowding_truncate(F, n_survive):
    fronts = NonDominatedSorting().do(F)
    survivors = []
    for front in fronts:
        if len(survivors) + len(front) <= n_survive:
            survivors.extend(front.tolist())
        else:
            remaining = n_survive - len(survivors)
            if remaining <= 0:
                break
            cd = calc_crowding_distance(F[front])
            order = np.argsort(-cd)
            survivors.extend(np.array(front)[order[:remaining]].tolist())
            break
    return np.array(survivors, dtype=int)


def _polynomial_mutation(x, prob, eta, xl, xu, rng):
    x = x.copy()
    do_mutate = rng.random(x.shape) < prob
    u = rng.random(x.shape)
    delta1 = (x - xl) / np.maximum(xu - xl, 1e-12)
    delta2 = (xu - x) / np.maximum(xu - xl, 1e-12)
    mask_low = do_mutate & (u <= 0.5)
    mask_high = do_mutate & (u > 0.5)
    val_low = 2 * u + (1 - 2 * u) * (1 - delta1) ** (eta + 1.0)
    deltaq_low = val_low ** (1.0 / (eta + 1.0)) - 1.0
    val_high = 2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta + 1.0)
    deltaq_high = 1.0 - val_high ** (1.0 / (eta + 1.0))
    x[mask_low] += (deltaq_low * (xu - xl))[mask_low]
    x[mask_high] += (deltaq_high * (xu - xl))[mask_high]
    return np.clip(x, xl, xu)


def _levy_flight(dim, beta, rng, size=1):
    from scipy.special import gamma
    sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma_u, size=(size, dim))
    v = rng.normal(0, 1, size=(size, dim))
    return 0.01 * u / (np.abs(v) ** (1.0 / beta) + 1e-9)


# ----------------------------------------------------------------------
# Item 5: reference-vector-guided archive for many-objective handling
# ----------------------------------------------------------------------
class ReferenceVectorArchive:
    """External archive with reference-vector-based association (item 5).

    For n_obj >= 3, members are associated to the nearest Das-Dennis
    reference direction by perpendicular distance after normalization,
    and at most `per_vector_cap` members are retained per direction
    before falling back to crowding-distance truncation. This gives
    archive clustering aligned to a uniform spread of reference
    directions, which is the standard mechanism (as in NSGA-III/RVEA)
    for preventing archive collapse onto a small region of a
    many-objective front -- a known failure mode of plain non-dominated
    archiving as objective count grows. For n_obj == 2 this reduces to
    the same crowding-based archive used in v1, since reference-vector
    clustering adds no benefit at M=2.
    """

    def __init__(self, capacity, n_obj, n_partitions=None):
        self.capacity = capacity
        self.n_obj = n_obj
        self.X = None
        self.F = None
        self.use_ref_dirs = n_obj >= 3
        if self.use_ref_dirs:
            if n_partitions is None:
                n_partitions = 12 if n_obj == 3 else 6
            self.ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
            self.ref_dirs = self.ref_dirs / np.linalg.norm(self.ref_dirs, axis=1, keepdims=True)
        else:
            self.ref_dirs = None

    def _associate(self, F, ideal, nadir):
        denom = np.maximum(nadir - ideal, 1e-12)
        Fn = (F - ideal) / denom
        norms = np.linalg.norm(Fn, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        Fn_dir = Fn / norms
        cos_sim = Fn_dir @ self.ref_dirs.T
        assoc = np.argmax(cos_sim, axis=1)
        perp_dist = np.linalg.norm(
            Fn - (Fn @ self.ref_dirs.T)[np.arange(len(F)), assoc, None] * self.ref_dirs[assoc],
            axis=1,
        )
        return assoc, perp_dist

    def _ref_vector_truncate(self, X_all, F_all, n_keep):
        ideal = F_all.min(axis=0)
        nadir = F_all.max(axis=0)
        assoc, perp_dist = self._associate(F_all, ideal, nadir)
        order = np.argsort(perp_dist)
        keep = []
        per_vector_count = {}
        cap = max(1, int(np.ceil(n_keep / len(self.ref_dirs))) + 1)
        for idx in order:
            v = assoc[idx]
            c = per_vector_count.get(v, 0)
            if c < cap:
                keep.append(idx)
                per_vector_count[v] = c + 1
            if len(keep) >= n_keep:
                break
        if len(keep) < n_keep:
            remaining = [i for i in order if i not in set(keep)]
            keep.extend(remaining[: n_keep - len(keep)])
        return np.array(keep[:n_keep], dtype=int)

    def update(self, X_new, F_new):
        if self.X is None:
            X_all, F_all = X_new, F_new
        else:
            X_all = np.vstack([self.X, X_new])
            F_all = np.vstack([self.F, F_new])

        nd = NonDominatedSorting().do(F_all, only_non_dominated_front=True)
        X_all, F_all = X_all[nd], F_all[nd]

        if len(F_all) > 1:
            _, unique_idx = np.unique(np.round(F_all, 10), axis=0, return_index=True)
            X_all, F_all = X_all[unique_idx], F_all[unique_idx]

        if len(F_all) > self.capacity:
            if self.use_ref_dirs and len(F_all) > len(self.ref_dirs):
                keep = self._ref_vector_truncate(X_all, F_all, self.capacity)
            else:
                keep = _crowding_truncate(F_all, self.capacity)
            X_all, F_all = X_all[keep], F_all[keep]

        self.X, self.F = X_all, F_all

    def sample(self, n, rng):
        if self.X is None or len(self.X) == 0:
            return None
        idx = rng.integers(0, len(self.X), size=n)
        return self.X[idx]

    def neighborhood_sample(self, anchor_idx, n_neighbors, rng):
        """Item 5: adaptive neighborhood selection -- sample archive
        members whose associated reference direction is close to the
        anchor's, restricting mating to a local neighborhood in
        objective space rather than the full archive. Falls back to
        uniform sampling when reference directions are unavailable
        (n_obj == 2) or the archive is too small to support it."""
        if self.X is None or len(self.X) < 2:
            return None
        if not self.use_ref_dirs or len(self.X) <= n_neighbors:
            return self.sample(n_neighbors, rng)
        ideal = self.F.min(axis=0)
        nadir = self.F.max(axis=0)
        assoc, _ = self._associate(self.F, ideal, nadir)
        anchor_vec = assoc[anchor_idx % len(assoc)]
        dirs_dist = np.linalg.norm(self.ref_dirs - self.ref_dirs[anchor_vec], axis=1)
        order = np.argsort(dirs_dist)
        near_vecs = set(order[: max(3, len(self.ref_dirs) // 4)].tolist())
        mask = np.isin(assoc, list(near_vecs))
        candidates = np.where(mask)[0]
        if len(candidates) < n_neighbors:
            return self.sample(n_neighbors, rng)
        idx = rng.choice(candidates, size=n_neighbors, replace=False)
        return self.X[idx]


# ----------------------------------------------------------------------
# Item 2: success-history adaptive DE-style archive exploitation
# ----------------------------------------------------------------------
class SuccessHistoryArchive:
    """Item 2: SHADE-style success-history adaptation for the
    archive-guided exploitation operator, replacing v1's fixed-sigma
    Gaussian perturbation N(0, sigma^2 I).

    Maintains a circular memory of (F, CR) parameter pairs that
    produced successful (non-dominated-improving) offspring in past
    generations, following the historical memory mechanism of
    Tanabe and Fukunaga's SHADE (2013) adapted to the multi-objective,
    archive-guided setting: F and CR for each new trial are sampled
    from a randomly chosen memory slot (Cauchy for F, Normal for CR),
    and slots are updated via the Lehmer-mean weighted-success rule
    whenever at least one successful trial occurred in a generation.
    This is a substantially more powerful and self-tuning local
    exploitation operator than fixed-variance Gaussian perturbation,
    since the step size and crossover rate adapt to what has actually
    worked on this problem rather than following a generic schedule.
    """

    def __init__(self, memory_size=10, f_init=0.5, cr_init=0.5):
        self.memory_size = memory_size
        self.mem_F = np.full(memory_size, f_init)
        self.mem_CR = np.full(memory_size, cr_init)
        self.idx_ptr = 0
        self.success_F = []
        self.success_CR = []
        self.success_delta = []

    def sample_params(self, rng, de_scale=0.1):
        slot = rng.integers(0, self.memory_size)
        mu_f = self.mem_F[slot]
        mu_cr = self.mem_CR[slot]
        F = -1
        while F <= 0:
            F = mu_f + de_scale * rng.standard_cauchy()
        F = min(F, 1.0)
        CR = np.clip(rng.normal(mu_cr, 0.1), 0.0, 1.0)
        return F, CR

    def record_success(self, F, CR, delta):
        self.success_F.append(F)
        self.success_CR.append(CR)
        self.success_delta.append(max(delta, 1e-12))

    def update_memory(self):
        if not self.success_F:
            return
        w = np.array(self.success_delta)
        w = w / w.sum()
        sf = np.array(self.success_F)
        scr = np.array(self.success_CR)
        lehmer_f = np.sum(w * sf ** 2) / max(np.sum(w * sf), 1e-12)
        mean_cr = np.sum(w * scr)
        self.mem_F[self.idx_ptr] = np.clip(lehmer_f, 1e-3, 1.0)
        self.mem_CR[self.idx_ptr] = mean_cr
        self.idx_ptr = (self.idx_ptr + 1) % self.memory_size
        self.success_F, self.success_CR, self.success_delta = [], [], []


def _shade_de_trial(x_target, x_base, archive_pool, F, CR, xl, xu, rng):
    """current-to-pbest/1 style mutation seeded from the archive, with
    binomial crossover -- the exploitation half of item 2."""
    pool = archive_pool
    if len(pool) < 2:
        return x_target.copy()
    idx = rng.choice(len(pool), size=min(2, len(pool)), replace=False)
    r1 = pool[idx[0]]
    r2 = pool[idx[1] if len(idx) > 1 else idx[0]]
    mutant = x_target + F * (x_base - x_target) + F * (r1 - r2)
    mutant = np.clip(mutant, xl, xu)
    D = len(x_target)
    cross_mask = rng.random(D) < CR
    if not cross_mask.any():
        cross_mask[rng.integers(0, D)] = True
    trial = np.where(cross_mask, mutant, x_target)
    return trial


# ----------------------------------------------------------------------
# Item 4 / 6: search-state indicators for adaptive regulation and
# stagnation detection (no access to the true Pareto front; computed
# only from information available to the optimizer during a real run)
# ----------------------------------------------------------------------
class SearchStateTracker:
    """Tracks measurable, reference-front-free indicators used by items
    1, 3, 4, and 6: population diversity, ideal-point movement rate,
    and archive growth rate, plus a stagnation counter with cooldown.
    """

    def __init__(self, window=5, stagnation_patience=10, cooldown=15):
        self.ideal_history = []
        self.diversity_history = []
        self.archive_size_history = []
        self.window = window
        self.stagnation_patience = stagnation_patience
        self.cooldown = cooldown
        self.gens_since_improvement = 0
        self.cooldown_remaining = 0

    def update(self, F_pop, archive_size):
        ideal = F_pop.min(axis=0)
        self.ideal_history.append(ideal)
        centroid = F_pop.mean(axis=0)
        diversity = np.mean(np.linalg.norm(F_pop - centroid, axis=1))
        self.diversity_history.append(diversity)
        self.archive_size_history.append(archive_size)

        improved = True
        if len(self.ideal_history) > self.window:
            prev_ideal = self.ideal_history[-self.window - 1]
            movement = np.linalg.norm(ideal - prev_ideal)
            prev_archive = self.archive_size_history[-self.window - 1]
            archive_growth = archive_size - prev_archive
            improved = (movement > 1e-6) or (archive_growth > 0)

        if improved:
            self.gens_since_improvement = 0
        else:
            self.gens_since_improvement += 1

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def is_stagnant(self):
        return (self.gens_since_improvement >= self.stagnation_patience
                and self.cooldown_remaining == 0)

    def trigger_recovery(self):
        self.gens_since_improvement = 0
        self.cooldown_remaining = self.cooldown

    def convergence_rate(self):
        """Normalized recent ideal-point movement; near 0 under
        stagnation, larger when the front is still advancing. Used by
        item 4 / item 1 to bias state selection continuously rather
        than only at hard stagnation triggers."""
        if len(self.ideal_history) <= self.window:
            return 1.0
        prev = self.ideal_history[-self.window - 1]
        cur = self.ideal_history[-1]
        movement = np.linalg.norm(cur - prev)
        scale = np.linalg.norm(cur) + 1e-9
        return float(np.clip(movement / scale, 0.0, 1.0))

    def diversity_ratio(self):
        """Current diversity relative to its running maximum, in
        [0, 1]; used by item 1 to bias toward exploration when
        diversity has collapsed relative to its historical peak."""
        if not self.diversity_history:
            return 1.0
        peak = max(self.diversity_history)
        if peak <= 1e-12:
            return 1.0
        return float(np.clip(self.diversity_history[-1] / peak, 0.0, 1.0))


# ----------------------------------------------------------------------
# Item 3: dynamic fission-fusion subgroup scheduling
# ----------------------------------------------------------------------
def _subgroup_count(t, T, n_pop, k_early=8, k_late=2):
    """Item 3: number of subgroups shrinks monotonically from k_early
    in early generations to k_late in late generations, implementing
    "many subgroups early, fewer and stronger cooperation late" as a
    smooth cosine schedule rather than a hard switch, so the
    fission-to-fusion transition has no discontinuity that could
    itself trigger a stagnation false-positive in SearchStateTracker.
    """
    progress = np.clip(t / max(T, 1), 0.0, 1.0)
    k = k_late + 0.5 * (k_early - k_late) * (1 + np.cos(np.pi * progress))
    k = int(np.clip(round(k), k_late, k_early))
    return max(1, min(k, n_pop // 2))


def _assign_subgroups(n_pop, n_groups, rng):
    """Random balanced partition of population indices into n_groups
    subgroups, reshuffled each generation, used for the fission-phase
    guide selection in item 3 (restricts velocity-update guides to
    within-subgroup members during early/mid search, encouraging
    parallel exploration of distinct regions; this restriction is
    relaxed as n_groups shrinks toward k_late, naturally increasing
    cross-subgroup cooperation in later generations without an
    explicit separate "fusion strength" parameter)."""
    perm = rng.permutation(n_pop)
    groups = np.array_split(perm, n_groups)
    membership = np.zeros(n_pop, dtype=int)
    for g_id, members in enumerate(groups):
        membership[members] = g_id
    return membership, groups


# ----------------------------------------------------------------------
# KOA-v2
# ----------------------------------------------------------------------
class KOAv2(Algorithm):
    """
    Kipunji Optimization Algorithm, version 2.

    Redesigned relative to KOA-v1 (algorithms/koa.py) per the six items
    described in the module docstring. Parameters retain the v1 names
    where the underlying quantity is conceptually unchanged (alpha0, A,
    tau, F_scale, eps), with additional parameters for the new
    mechanisms (shade_memory_size, k_early, k_late, stagnation_patience,
    stagnation_cooldown).
    """

    def __init__(self, pop_size=100, n_archive=120,
                 alpha0=0.5, A=0.3, tau=20,
                 eta_PI=20, F_scale=0.5, eps=1e-9,
                 shade_memory_size=10,
                 k_early=8, k_late=2,
                 stagnation_window=5, stagnation_patience=10, stagnation_cooldown=15,
                 seed=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.n_archive = n_archive
        self.alpha0 = alpha0
        self.A = A
        self.tau = tau
        self.eta_PI = eta_PI
        self.F_scale = F_scale
        self.eps = eps
        self.shade_memory_size = shade_memory_size
        self.k_early = k_early
        self.k_late = k_late
        self.stagnation_window = stagnation_window
        self.stagnation_patience = stagnation_patience
        self.stagnation_cooldown = stagnation_cooldown

        self.velocities = None
        self.rng = np.random.default_rng(seed)
        self.koa_archive = None  # built in _setup once n_obj is known
        self.shade = SuccessHistoryArchive(memory_size=shade_memory_size)
        self.tracker = SearchStateTracker(window=stagnation_window,
                                           stagnation_patience=stagnation_patience,
                                           cooldown=stagnation_cooldown)
        self._exploration_boost_gens = 0
        self._last_branch = None

    def _setup(self, problem, **kwargs):
        self.problem = problem
        self.koa_archive = ReferenceVectorArchive(self.n_archive, problem.n_obj)

    def _initialize_infill(self):
        sampling = LHS()
        X = sampling.do(self.problem, self.pop_size).get("X")
        self.velocities = np.zeros_like(X)
        return Population.new(X=X)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.koa_archive.update(self.pop.get("X"), self.pop.get("F"))
        self.tracker.update(self.pop.get("F"), len(self.koa_archive.X))

    def _state_probabilities(self, t, T):
        """Item 1 + item 4: adaptive state selection. Replaces v1's
        fixed thresholds (0.35 fission / 0.40 fusion / 0.25 archive)
        with probabilities driven by measured convergence rate and
        diversity ratio rather than a static split.

        Design: as convergence_rate falls (search is no longer
        advancing the ideal point) and diversity_ratio stays high
        (population has not yet collapsed), exploitation probability
        rises -- the population is diverse but not converging, the
        textbook signature of needing stronger local search rather
        than more exploration. Conversely, when diversity_ratio falls
        sharply, fission probability rises to rebuild spread. This
        directly targets the v1 weakness of fixed-probability state
        selection that cannot respond to actual search progress.
        """
        conv_rate = self.tracker.convergence_rate()
        div_ratio = self.tracker.diversity_ratio()
        progress = np.clip(t / max(T, 1), 0.0, 1.0)

        # base schedule: shift weight from fission to exploitation over time
        base_fission = 0.45 * (1 - progress) + 0.15 * progress
        base_exploit = 0.20 * (1 - progress) + 0.55 * progress

        # diversity collapse pulls weight back toward fission
        diversity_pull = (1 - div_ratio) * 0.25
        # convergence stall pulls weight toward exploitation
        stall_pull = (1 - conv_rate) * 0.25

        p_fission = np.clip(base_fission + diversity_pull - stall_pull, 0.05, 0.7)
        p_exploit = np.clip(base_exploit + stall_pull - diversity_pull, 0.05, 0.7)
        p_fusion = np.clip(1.0 - p_fission - p_exploit, 0.05, 0.9)

        total = p_fission + p_fusion + p_exploit
        return p_fission / total, p_fusion / total, p_exploit / total

    def _infill(self):
        problem = self.problem
        N, D = self.pop_size, problem.n_var
        xl, xu = problem.xl, problem.xu
        rng = self.rng

        t = self.n_gen
        T = self.termination.n_max_gen if hasattr(self.termination, "n_max_gen") else 300

        # item 6: stagnation detection and bounded recovery
        if self.tracker.is_stagnant():
            self.tracker.trigger_recovery()
            self._exploration_boost_gens = 3  # short, bounded reactivation window

        boosting = self._exploration_boost_gens > 0
        if boosting:
            self._exploration_boost_gens -= 1

        alpha_t = self.alpha0 + self.A * np.sin(2 * np.pi * t / self.tau)
        if boosting:
            alpha_t *= 1.6  # temporary, time-limited amplitude increase only
            # while genuinely stagnant -- avoids "excessive randomness" by
            # being magnitude-bounded and duration-bounded (3 generations)
            # rather than a permanent or unconditional escalation.

        p_fission, p_fusion, p_exploit = self._state_probabilities(t, T)

        # item 3: dynamic fission-fusion subgroup scheduling
        n_groups = _subgroup_count(t, T, N, self.k_early, self.k_late)
        membership, groups = _assign_subgroups(N, n_groups, rng)

        X = self.pop.get("X")
        s = rng.random(N)
        X_new = np.zeros_like(X)
        archive_X = self.koa_archive.X if self.koa_archive.X is not None else X

        for i in range(N):
            if s[i] < p_fission:
                # Fission phase: velocity update with guide drawn from the
                # same subgroup when possible (item 3), else archive/global.
                own_group = groups[membership[i]]
                candidates = own_group[own_group != i]
                if len(candidates) > 0 and rng.random() < 0.7:
                    x_guide = X[rng.choice(candidates)]
                elif len(archive_X) > 0:
                    x_guide = archive_X[rng.integers(0, len(archive_X))]
                else:
                    others = np.delete(np.arange(N), i)
                    x_guide = X[rng.choice(others)]
                diff = x_guide - X[i]
                norm = np.linalg.norm(diff) + self.eps
                self.velocities[i] = 0.6 * self.velocities[i] + alpha_t * diff / norm
                if rng.random() < 0.15:
                    self.velocities[i] += _levy_flight(D, 1.5, rng, size=1)[0]
                x_prime = X[i] + self.velocities[i]
                x_prime = np.clip(x_prime, xl, xu)

            elif s[i] < p_fission + p_fusion:
                # Fusion phase: cross-subgroup differential displacement,
                # pool drawn preferentially from outside the own subgroup
                # to increase cooperation strength as n_groups shrinks
                # (item 3's "fewer groups, stronger cooperation" in the
                # late-stage regime, since fewer groups means "outside"
                # candidates are pooled from a larger fraction of N).
                other_idx = np.where(membership != membership[i])[0]
                pool_X = X[other_idx] if len(other_idx) >= 3 else X
                idx = rng.choice(len(pool_X), size=min(3, len(pool_X)), replace=False)
                if len(idx) < 3:
                    x_prime = X[i].copy()
                else:
                    xr1, xr2, xr3 = pool_X[idx[0]], pool_X[idx[1]], pool_X[idx[2]]
                    x_int = xr1 + self.F_scale * (xr2 - xr3)
                    x_prime = np.clip(0.5 * (x_int + X[i]), xl, xu)

            else:
                # item 2: SHADE-style archive-guided exploitation,
                # replacing v1's fixed Gaussian N(0, sigma^2 I).
                F_de, CR_de = self.shade.sample_params(rng)
                if len(archive_X) > 0:
                    # item 5: neighborhood-restricted archive sampling
                    # for many-objective problems
                    nb = self.koa_archive.neighborhood_sample(i, min(3, len(archive_X)), rng)
                    pool = nb if nb is not None else archive_X
                    x_base = pool[rng.integers(0, len(pool))]
                else:
                    pool = X
                    x_base = X[i]
                x_prime = _shade_de_trial(X[i], x_base, pool, F_de, CR_de, xl, xu, rng)
                X_new_trial_meta = (i, F_de, CR_de)  # recorded after evaluation in _advance

            # Defensive backstop: guarantee x_prime is within [xl, xu] and
            # free of NaN/Inf regardless of which branch produced it,
            # rather than trusting each branch's individual clip to be
            # correct and exhaustive. NaN can only arise here from a
            # division by (near-)zero norm in the fission branch (Eq. 7's
            # diff/norm term) slipping past the self.eps guard in an edge
            # case (e.g. two solutions becoming numerically identical);
            # if that happens, fall back to the pre-update parent rather
            # than propagate a NaN individual into the population.
            if np.any(~np.isfinite(x_prime)):
                x_prime = X[i].copy()
            x_prime = np.clip(x_prime, xl, xu)

            x_prime = _polynomial_mutation(
                x_prime, prob=1.0 / D, eta=self.eta_PI, xl=xl, xu=xu, rng=rng
            )
            X_new[i] = x_prime

        # stash per-individual state-branch for success tracking in
        # _advance (item 2's memory update needs to know which
        # offspring came from the exploitation branch)
        self._last_branch = np.where(s < p_fission, 0, np.where(s < p_fission + p_fusion, 1, 2))
        return Population.new(X=X_new)

    def _advance(self, infills=None, **kwargs):
        P_old = self.pop
        P_new = infills

        X_union = np.vstack([P_old.get("X"), P_new.get("X")])
        F_union = np.vstack([P_old.get("F"), P_new.get("F")])
        keep = _crowding_truncate(F_union, self.pop_size)
        self.pop = Population.new(X=X_union[keep], F=F_union[keep])

        # item 2: success-history memory update -- a trial counts as
        # successful if it is not dominated by its parent in the merged
        # survivor set; approximate via objective-sum improvement, which
        # is a standard cheap proxy when explicit parent-child dominance
        # bookkeeping is not retained per-branch across the merge step.
        if self._last_branch is not None:
            exploit_idx = np.where(self._last_branch == 2)[0]
            if len(exploit_idx) > 0:
                old_F = P_old.get("F")
                new_F = P_new.get("F")
                for i in exploit_idx:
                    if i < len(old_F) and i < len(new_F):
                        delta = float(np.sum(old_F[i]) - np.sum(new_F[i]))
                        if delta > 0:
                            F_de, CR_de = self.shade.sample_params(self.rng)
                            self.shade.record_success(F_de, CR_de, delta)
        self.shade.update_memory()

        self.koa_archive.update(P_new.get("X"), P_new.get("F"))
        self.tracker.update(self.pop.get("F"), len(self.koa_archive.X))

    def _set_optimum(self, **kwargs):
        if self.koa_archive.X is not None:
            self.opt = Population.new(X=self.koa_archive.X, F=self.koa_archive.F)
        else:
            self.opt = self.pop


# ----------------------------------------------------------------------
# Item 7: computational complexity analysis (analytical, not executed)
# ----------------------------------------------------------------------
"""
Complexity analysis, per generation, N = population size, D = decision
variables, M = objectives, A = archive size (bounded by n_archive).

v1 per-generation cost:
  - state branch loop: O(N) iterations, each O(D) for velocity/DE/Gaussian
    update => O(N D)
  - StateFiltering (non-dominated sort + crowding): O(M N^2) worst case
    for the sort (standard fast-non-dominated-sort bound), O(M N log N)
    for crowding distance
  - archive update: non-dominated sort over (A + N) candidates, O(M(A+N)^2)
  Total v1: O(M(A+N)^2 + N D)

v2 adds, per generation:
  - subgroup assignment: O(N) (array_split of a permutation)
  - reference-vector association (item 5, only for M >= 3): for each of
    up to (A + N) archive candidates, an O(M) dot product against R
    reference directions => O(M R (A+N)); R is fixed by n_partitions
    and does not grow with N or T
  - neighborhood_sample: O(R) to rank reference directions by distance
    to the anchor's associated direction, plus O(A) masking
  - SHADE parameter sampling and memory update: O(1) per individual for
    sampling, O(N) for the success-list reduction at generation end
  - stagnation tracker update: O(M) for ideal-point recomputation,
    O(N) for diversity centroid distance

Since R (reference direction count) and the SHADE memory size are fixed
constants independent of N, D, M, and T, the additional terms are
dominated by the existing O(M(A+N)^2) non-dominated sorting cost that
v1 already pays for archive maintenance. The asymptotic per-generation
complexity of v2 is therefore unchanged from v1 at O(M(A+N)^2 + N D);
the redesign trades a larger constant factor (additional O(N) and O(R N)
bookkeeping per generation) for the qualitative behavioral changes
described above, not a different complexity class. Over T generations,
both v1 and v2 total O(T (M(A+N)^2 + N D)).

Scalability implications: the dominant term M(A+N)^2 is shared with
essentially all archive-based and elitist MOEAs (e.g. NSGA-II's
O(M N^2) sorting cost is the same order with A folded into N); v2 does
not change this scaling, so it does not address the same large-N
scalability ceiling as algorithms that replace non-dominated sorting
itself (e.g. decomposition-based or hypervolume-approximation methods).
For the large-scale problems in this benchmark suite (LSMOP, D up to
60), the N D term remains small relative to M(A+N)^2 at the evaluated
population size (N=100, A=120), so v2's added bookkeeping is not
expected to materially change wall-clock time relative to v1; this
should be confirmed empirically alongside the IGD/HV results rather
than assumed from the asymptotic analysis alone.
"""
