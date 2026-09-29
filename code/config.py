"""
KOA Final V3 experiment configuration -- 16 problems, 30 runs, 7 algorithms.

FE-budget fix: every algorithm below is built with an identical
pop_size (see algorithms/factory.py Fix 9) and run for an identical
number of generations, so every algorithm consumes exactly
FE_BUDGET = POP_SIZE * N_GEN function evaluations per run (the initial
population's evaluation is not separately counted by pymoo's
evaluator.n_eval; see README_FIXES.md for the exact accounting used in
the manuscript's Experimental Details section).
"""
POP_SIZE  = 100
N_ARCHIVE = 120
N_GEN     = 300
N_RUNS    = 30
RANDOM_SEEDS = list(range(1, N_RUNS + 1))

# Every algorithm is guaranteed (Fix 9, algorithms/factory.py) to use
# exactly POP_SIZE individuals per generation, so this is the exact,
# identical function-evaluation budget for every algorithm x problem
# x seed combination in this study.
FE_BUDGET = POP_SIZE * N_GEN

# GPU/device pinning for shared-server compliance -- see device.py.
# This codebase is CPU-only (pymoo/NumPy); this setting does not make
# anything run on a GPU, it only prevents this job from grabbing every
# visible GPU on a shared multi-tenant machine.
DEFAULT_GPU_DEVICE = 0

ALGORITHMS = ["KOA", "NSGA-II", "NSGA-III", "MOEA/D", "RVEA", "AGE-MOEA2", "CMOPSO"]

PROBLEMS = [
    ("ZDT1",   2, "zdt",   dict(name="zdt1")),
    ("ZDT2",   2, "zdt",   dict(name="zdt2")),
    ("ZDT3",   2, "zdt",   dict(name="zdt3")),
    ("DTLZ1",  3, "dtlz",  dict(name="dtlz1", n_var=7,  n_obj=3)),
    ("DTLZ2",  3, "dtlz",  dict(name="dtlz2", n_var=12, n_obj=3)),
    ("DTLZ4",  3, "dtlz",  dict(name="dtlz4", n_var=12, n_obj=3)),
    ("DTLZ6",  3, "dtlz",  dict(name="dtlz6", n_var=12, n_obj=3)),
    ("DTLZ7",  3, "dtlz",  dict(name="dtlz7", n_var=22, n_obj=3)),
    ("WFG1",   3, "wfg",   dict(name="wfg1", n_var=10, n_obj=3)),
    ("WFG4",   3, "wfg",   dict(name="wfg4", n_var=10, n_obj=3)),
    ("WFG6",   3, "wfg",   dict(name="wfg6", n_var=10, n_obj=3)),
    ("WFG9",   3, "wfg",   dict(name="wfg9", n_var=10, n_obj=3)),
    ("LSMOP1", 3, "lsmop", dict(which=1, n_obj=3, n_var=60)),
    ("LSMOP9", 3, "lsmop", dict(which=9, n_obj=3, n_var=60)),
    ("MaF4",   3, "maf",   dict(which=4, n_obj=3, n_var=None)),
    ("MaF7",   3, "maf",   dict(which=7, n_obj=3, n_var=None)),
]

PROBLEM_NAMES = [p[0] for p in PROBLEMS]


def build_problem(kind, kwargs):
    if kind == "zdt":
        from pymoo.problems import get_problem
        return get_problem(kwargs["name"])
    elif kind == "dtlz":
        from pymoo.problems import get_problem
        return get_problem(kwargs["name"], n_var=kwargs["n_var"], n_obj=kwargs["n_obj"])
    elif kind == "wfg":
        from pymoo.problems import get_problem
        return get_problem(kwargs["name"], n_var=kwargs["n_var"], n_obj=kwargs["n_obj"])
    elif kind == "lsmop":
        from problems.lsmop import LSMOP
        return LSMOP(which=kwargs["which"], n_obj=kwargs["n_obj"], n_var=kwargs["n_var"])
    elif kind == "maf":
        from problems.maf import MaF4, MaF7
        if kwargs["which"] == 4:
            return MaF4(n_obj=kwargs["n_obj"], n_var=kwargs["n_var"])
        elif kwargs["which"] == 7:
            return MaF7(n_obj=kwargs["n_obj"], n_var=kwargs["n_var"])
        raise ValueError(kwargs["which"])
    raise ValueError(f"Unknown kind: {kind}")
