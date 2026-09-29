"""
utils/fs_safety.py

FIX (crash on MOEA/D error logging): algorithm names are used verbatim
inside log filenames (e.g. f"error_{problem}_{algo_name}_seed{seed}.log").
"MOEA/D" contains a literal "/", which the filesystem reads as a path
separator, so writing that "filename" required a directory called
"MOEA" to already exist and crashed with FileNotFoundError instead --
turning a normal, already-caught per-run error into a fatal crash that
killed the entire multiprocessing pool and the whole benchmark run.

`safe_name()` is used everywhere an algorithm/variant name is
interpolated into a filesystem path.

FIX (race condition corrupting pymoo's cached Pareto-front files):
pymoo caches remote true-PF reference files (used by DTLZ1/3/5/6/7,
etc.) by downloading them ONCE into a file inside the pymoo package
install itself (e.g. .../pymoo/data/pymoo/pf/dtlz6-3d.pf) the first time
`problem.pareto_front()` is called for that problem. There is no lock
around that download. When many `run_experiments.py` worker processes
call `pareto_front()` for the SAME problem at nearly the same moment
(exactly what happens once N worker processes all start on DTLZ6 at
once), several of them see the file does not exist yet and call
`urllib.request.urlretrieve` into the SAME destination path
concurrently -- the interleaved writes truncate/corrupt the file (this
is exactly the null-byte garbage seen in the traceback: the file is
half of one download's bytes and half of another's, or half-written
when the failing run's process read it).

`prewarm_pareto_fronts()` builds every problem ONCE in the main process
and calls `.pareto_front()` on it, sequentially, BEFORE any
multiprocessing pool is created. This forces every remote file to be
downloaded exactly once, by a single process, before any worker can
race on it. Every run_*.py entry point in this repository that spawns a
Pool now calls this first.
"""
import re
import sys


def safe_name(name: str) -> str:
    """Make an algorithm/variant name safe to embed in a filename.
    'MOEA/D' -> 'MOEA-D'. Idempotent and collision-free among the
    algorithm/variant names actually used in this repository."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "-", str(name))


def prewarm_pareto_fronts(problem_specs, build_problem):
    """
    problem_specs: iterable of (name, n_obj, kind, kwargs) as used
                    throughout config.PROBLEMS.
    build_problem: config.build_problem

    Called once, single-threaded, in the main process before any
    multiprocessing.Pool is created, so any remote PF file pymoo needs
    to download gets downloaded exactly once by exactly one process.
    Safe to call redundantly / on every script start -- pymoo's own
    cache check (`if not os.path.exists(f)`) makes this a no-op once
    the file is already present.
    """
    print("Pre-warming pymoo's Pareto-front reference cache "
          "(single-threaded, avoids a multiprocessing download race)...")
    failures = []
    for name, n_obj, kind, kwargs in problem_specs:
        try:
            problem = build_problem(kind, kwargs)
            pf = problem.pareto_front()
            if pf is None or len(pf) == 0:
                failures.append((name, "empty/None pareto_front()"))
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))

    if failures:
        print("WARNING: pareto_front() failed to pre-warm for:")
        for name, err in failures:
            print(f"    {name}: {err}")
        print("This usually means a cached reference file under "
              "<your-venv>/lib/python*/site-packages/pymoo/data/pymoo/pf/ "
              "is corrupted (often from an earlier run that hit the exact "
              "race condition this pre-warm step exists to prevent). Fix "
              "with, e.g.:\n"
              "    python -c \"import pymoo,glob,os; "
              "print(os.path.dirname(pymoo.__file__))\"\n"
              "then delete the corrupted .pf file(s) under "
              "<that path>/data/pymoo/pf/ and re-run this script -- it "
              "will re-download them safely before starting any workers.")
    else:
        print("Pre-warm OK for all problems.")
    return failures
