"""
Main experiment runner.

Usage (on your own machine, with as many CPU cores as available):

    python run_experiments.py --n-jobs 32 --gpu 0

This launches POP_SIZE/N_GEN/N_RUNS as configured in config.py across all
(problem x algorithm) pairs, 30 independent runs each, using multiprocessing
across runs. Results are checkpointed incrementally to results/raw_runs.csv
so an interrupted job can be resumed without losing completed work --
re-running the script skips any (problem, algorithm, seed) combination
already present in the checkpoint file.

This is a CPU-bound workload -- population-based MOEAs do not use a GPU in
pymoo, and nothing in this repository performs tensor/GPU computation.
`--gpu` only pins CUDA_VISIBLE_DEVICES so this job behaves as a good
citizen on a shared GPU server (see device.py); it has no effect on
speed or results. --n-jobs should be set to roughly your machine's
physical core count.

FIXES applied relative to the previous runner (see README_FIXES.md):
  * FIX 7  -- HV/IGD are computed from `res.F`, which for KOA now comes
    from the final population's non-dominated subset (<=pop_size points,
    same convention as every baseline), not the 120-member archive.
  * FIX 9  -- every algorithm is guaranteed an identical pop_size (and
    therefore identical FE budget) via algorithms/factory.py.
  * FIX (common init) -- one shared LHS initial population is generated
    per (problem, seed) and passed to every algorithm being compared.
  * FIX (AGE-MOEA2 reseeding) -- AGE-MOEA2 numerical failures are no
    longer retried under a DIFFERENT seed, which used to silently break
    the paired-seed protocol. A transient failure is now recorded
    honestly as a failed run under its ORIGINAL seed; the analysis
    scripts drop that (problem, seed) row for paired tests rather than
    substituting a value obtained under a different random draw.
  * fe_budget is now recorded per row for transparency, and n_eval (the
    algorithm's own actual evaluator count) is recorded alongside it so
    any future accounting discrepancy is immediately visible in the
    CSV rather than assumed away.
"""
import argparse
import sys
import time
import traceback
import multiprocessing as mp
from pathlib import Path

# --- GPU/device pinning must happen before heavy imports (device.py) ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import pin_device  # noqa: E402

import numpy as np
import pandas as pd

from config import PROBLEMS, ALGORITHMS, POP_SIZE, N_ARCHIVE, N_GEN, N_RUNS, FE_BUDGET, build_problem
from algorithms.factory import build_algorithm
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population
from utils.fs_safety import safe_name, prewarm_pareto_fronts

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_CSV = RESULTS_DIR / "raw_runs.csv"
LOG_DIR = Path(__file__).resolve().parent / "logs"

RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

CSV_COLUMNS = ["problem", "algorithm", "seed", "HV", "IGD", "Spacing", "Spread",
               "n_solutions", "n_eval", "fe_budget", "wall_seconds", "status", "error"]


def _single_run(args):
    """Run one (problem, algorithm, seed) combination. Designed to be
    picklable for multiprocessing.Pool -- the problem is passed as a
    plain (kind, kwargs) spec and reconstructed inside the worker via
    config.build_problem(), since pymoo Problem instances built from
    closures cannot be pickled across process boundaries."""
    problem_name, n_obj, kind, kwargs, algo_name, seed = args
    from pymoo.optimize import minimize

    t0 = time.time()
    metrics = {"HV": np.nan, "IGD": np.nan, "Spacing": np.nan, "Spread": np.nan}
    n_sol = 0
    n_eval = 0
    status = "error"
    err = ""

    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        if pf is None or len(pf) == 0:
            raise RuntimeError(f"No analytical Pareto front for {problem_name}")
        ref_point = compute_reference_point(pf)

        # FIX (common init): one shared initial population per
        # (problem, seed), used by EVERY algorithm compared on this row.
        init_X = shared_initial_population(problem, POP_SIZE, seed)

        algo = build_algorithm(algo_name, n_obj=n_obj, pop_size=POP_SIZE,
                                n_archive=N_ARCHIVE, seed=seed, init_X=init_X)
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)

        # FIX 7: res.F now uses the same "final-population, non-dominated
        # subset" output convention for KOA as for every baseline.
        F = res.F
        n_eval = int(res.algorithm.evaluator.n_eval)

        if F is None or len(F) == 0:
            metrics = {"HV": 0.0, "IGD": np.inf, "Spacing": np.inf, "Spread": np.inf}
            n_sol = 0
            status, err = "empty_front", ""
        else:
            metrics = compute_all_metrics(F, pf, ref_point)
            n_sol = len(F)
            status, err = "ok", ""

    except Exception as e:
        # FIX (AGE-MOEA2 reseeding): no retry under a different seed --
        # a failure is recorded honestly, under the ORIGINAL seed, so
        # the paired-seed protocol is never silently broken. Analysis
        # scripts must drop this (problem, seed) pair for paired tests.
        metrics = {"HV": np.nan, "IGD": np.nan, "Spacing": np.nan, "Spread": np.nan}
        n_sol = 0
        status = "error"
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        log_path = LOG_DIR / f"error_{safe_name(problem_name)}_{safe_name(algo_name)}_seed{seed}.log"
        with open(log_path, "w") as f:
            f.write(tb)

    wall = time.time() - t0
    return {
        "problem": problem_name, "algorithm": algo_name, "seed": seed,
        "HV": metrics["HV"], "IGD": metrics["IGD"],
        "Spacing": metrics["Spacing"], "Spread": metrics["Spread"],
        "n_solutions": n_sol, "n_eval": n_eval, "fe_budget": FE_BUDGET,
        "wall_seconds": round(wall, 2),
        "status": status, "error": err,
    }


def load_checkpoint():
    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
        done = set(zip(df["problem"], df["algorithm"], df["seed"]))
        return df, done
    return pd.DataFrame(columns=CSV_COLUMNS), set()


def build_task_list(done):
    tasks = []
    for problem_name, n_obj, kind, kwargs in PROBLEMS:
        for algo_name in ALGORITHMS:
            for seed in range(1, N_RUNS + 1):
                if (problem_name, algo_name, seed) in done:
                    continue
                tasks.append((problem_name, n_obj, kind, kwargs, algo_name, seed))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1),
                         help="Number of parallel worker processes")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                         help="Flush results to CSV every N completed runs")
    parser.add_argument("--gpu", type=int, default=0,
                         help="GPU device index to pin via CUDA_VISIBLE_DEVICES "
                              "('device port'). Has no effect on results or "
                              "speed -- this workload is CPU-only -- it only "
                              "keeps this job off other GPUs on a shared server.")
    args = parser.parse_args()
    pin_device(args.gpu)

    # FIX (race condition corrupting pymoo's cached PF files, e.g.
    # DTLZ6): download every remote reference file ONCE, single-threaded,
    # in this main process, before any worker process can race on it.
    prewarm_pareto_fronts(PROBLEMS, build_problem)

    df_existing, done = load_checkpoint()
    tasks = build_task_list(done)
    total_all = len(PROBLEMS) * len(ALGORITHMS) * N_RUNS

    print(f"Total (problem x algorithm x seed) combinations: {total_all}")
    print(f"Already completed (resumed from checkpoint): {len(done)}")
    print(f"Remaining to run: {len(tasks)}")
    print(f"Using {args.n_jobs} worker processes; pinned to GPU device {args.gpu} "
          f"(no compute effect -- CPU-only workload)")
    print(f"FE budget per run (identical across all algorithms, Fix 9): {FE_BUDGET}")

    if not tasks:
        print("Nothing to do -- all runs already completed.")
        return

    results_buffer = []
    n_completed = 0
    t_start = time.time()

    with mp.Pool(processes=args.n_jobs) as pool:
        for result in pool.imap_unordered(_single_run, tasks):
            results_buffer.append(result)
            n_completed += 1

            if n_completed % args.checkpoint_every == 0 or n_completed == len(tasks):
                chunk = pd.DataFrame(results_buffer)
                header = not RAW_CSV.exists()
                chunk.to_csv(RAW_CSV, mode="a", header=header, index=False)
                results_buffer = []

                elapsed = time.time() - t_start
                rate = n_completed / elapsed
                remaining = (len(tasks) - n_completed) / rate if rate > 0 else float("inf")
                print(f"[{n_completed}/{len(tasks)}] "
                      f"elapsed={elapsed/60:.1f}min "
                      f"est_remaining={remaining/60:.1f}min "
                      f"last={result['problem']}/{result['algorithm']}/seed{result['seed']} "
                      f"status={result['status']}")

    print(f"\nDone. Total wall time: {(time.time()-t_start)/60:.1f} minutes.")
    print(f"Raw results written incrementally to: {RAW_CSV}")


if __name__ == "__main__":
    main()
