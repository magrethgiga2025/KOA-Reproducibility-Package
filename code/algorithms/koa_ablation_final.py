"""
Ablation variants for KOA Final V3.

FIX (missing geometry ablation, the single most important gap identified
by the review): three fixed-rho variants and one legacy-Delta_i variant
are added below, on top of the five variants already present.

Variants
--------
  KOA                 - full algorithm (all fixes active, defaults)
  KOA-NoState         - fixed p_exp=0.30, no stagnation boost
  KOA-NoArchive       - offspring seeded from population only
  KOA-NoRefresh       - refresh_frac=0 (genuinely off after Fix 4)
  KOA-NoScaleNorm     - Fix D off (always hyperplane normalisation)
  KOA-NoPopSeed       - Fix E off (always use archive as SHADE pool)
  KOA-FixedRho0.2     - geometry-adaptive survival OFF: rho fixed at 0.20
  KOA-FixedRho0.6     - rho fixed at 0.60
  KOA-FixedRho1.0     - rho fixed at 1.00 (always full NSGA-III niching)
  KOA-RawDelta        - legacy raw-summed-objective SHADE success score,
                        for the raw-vs-normalised MaF4 diagnostic only
                        (NOT part of the main ablation table)
"""
import types
from algorithms.koa_final_v3 import KOAFinalV3


def build_ablation(variant, pop_size=100, n_archive=120, seed=None, init_X=None):

    if variant == "KOA":
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X)

    elif variant == "KOA-NoState":
        algo = KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X, gamma_b=10_000)

        def _fixed(self, t, T):
            return 0.30, 0.70
        algo._state_probabilities = types.MethodType(_fixed, algo)
        return algo

    elif variant == "KOA-NoArchive":
        algo = KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X)
        _orig = KOAFinalV3._infill

        def _no_archive(self):
            saved = self.koa_archive.X
            self.koa_archive.X = None
            r = _orig(self)
            self.koa_archive.X = saved
            return r
        algo._infill = types.MethodType(_no_archive, algo)
        return algo

    elif variant == "KOA-NoRefresh":
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X, refresh_frac=0.0)

    elif variant == "KOA-NoScaleNorm":
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X, scale_thresh=1e9)

    elif variant == "KOA-NoPopSeed":
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X, pop_seed_gens=0)

    # ---- Geometry-adaptive-survival ablation (previously missing) -----
    elif variant == "KOA-FixedRho0.2":
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X, rho_mode="fixed", rho_fixed=0.2)

    elif variant == "KOA-FixedRho0.6":
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X, rho_mode="fixed", rho_fixed=0.6)

    elif variant == "KOA-FixedRho1.0":
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X, rho_mode="fixed", rho_fixed=1.0)

    # ---- Raw-vs-normalised Delta_i diagnostic (MaF4), not part of the
    # main ablation table -- run only on the problems needed for that
    # specific diagnostic (see run_delta_diagnostic.py).
    elif variant == "KOA-RawDelta":
        return KOAFinalV3(pop_size=pop_size, n_archive=n_archive, seed=seed,
                           init_X=init_X, success_metric="raw")

    raise ValueError(f"Unknown ablation variant: {variant}")


# ======================================================================
# Full 2^3 factorial ablation (State x Archive x Geometry)
# ======================================================================
#
# The review's strongest ablation request: single-mechanism ablations
# (above) show what happens when ONE mechanism is removed from the full
# algorithm, but cannot distinguish additive effects from interactions
# between mechanisms. A full factorial isolates all three headline
# contributions (state regulation, archive-guided reproduction,
# geometry-adaptive survival) in every combination, so e.g.
# "State+Geometry" (archive off) can be compared against
# "State only" + "Geometry only" to test whether their combined effect
# is more or less than the sum of their individual effects.
#
# "Geometry off" uses rho_mode="fixed", rho_fixed=0.6 -- the single
# fixed-rho value that performed most competitively of {0.2, 0.6, 1.0}
# in the single-mechanism ablation, used here as the representative
# "non-adaptive" baseline for geometry, analogous to how "State off"
# uses a single representative fixed p_exp rather than sweeping it.

FIXED_RHO_FOR_FACTORIAL = 0.6
FIXED_P_EXP_FOR_FACTORIAL = 0.30


def build_factorial_variant(state_on: bool, archive_on: bool, geometry_on: bool,
                             pop_size=100, n_archive=120, seed=None, init_X=None):
    """One cell of the 2x2x2 factorial design. Returns (algo, label)."""
    kwargs = dict(pop_size=pop_size, n_archive=n_archive, seed=seed, init_X=init_X)

    if not geometry_on:
        kwargs["rho_mode"] = "fixed"
        kwargs["rho_fixed"] = FIXED_RHO_FOR_FACTORIAL

    algo = KOAFinalV3(**kwargs)

    if not state_on:
        def _fixed(self, t, T):
            return FIXED_P_EXP_FOR_FACTORIAL, None
        algo._state_probabilities = types.MethodType(_fixed, algo)

    if not archive_on:
        _orig_infill = KOAFinalV3._infill

        def _no_archive(self):
            saved = self.koa_archive.X
            self.koa_archive.X = None
            r = _orig_infill(self)
            self.koa_archive.X = saved
            return r
        algo._infill = types.MethodType(_no_archive, algo)

    label = "".join([
        "S" if state_on else "s",
        "A" if archive_on else "a",
        "G" if geometry_on else "g",
    ])
    return algo, label


# All 8 cells, "SAG" = full KOA (every mechanism on), "sag" = all three
# mechanisms disabled simultaneously (a genuine "vanilla" control absent
# from the single-mechanism ablation table, since KOA-NoState,
# KOA-NoArchive etc. each disable only one mechanism at a time).
FACTORIAL_CELLS = [
    (True,  True,  True),   # SAG -- full KOA
    (True,  True,  False),  # SAg
    (True,  False, True),   # SaG
    (True,  False, False),  # Sag
    (False, True,  True),   # sAG
    (False, True,  False),  # sAg
    (False, False, True),   # saG
    (False, False, False),  # sag -- all three off
]



# Variants that make up the MAIN ablation table (Table VII equivalent).
# KOA-RawDelta is intentionally excluded -- it is a separate, targeted
# diagnostic (see run_delta_diagnostic.py) and reported in its own table.
MAIN_ABLATION_VARIANTS = [
    "KOA", "KOA-NoState", "KOA-NoArchive", "KOA-NoRefresh",
    "KOA-NoScaleNorm", "KOA-NoPopSeed",
    "KOA-FixedRho0.2", "KOA-FixedRho0.6", "KOA-FixedRho1.0",
]
