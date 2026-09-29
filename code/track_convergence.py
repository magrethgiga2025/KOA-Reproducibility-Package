"""
Convergence tracking: records per-generation IGD for a small number of
representative seeds, for the IGD-convergence-curve figure (Fig. 3 style
in the original manuscript). Kept separate from run_experiments.py
because per-generation logging is far more expensive than final-only
metrics and only a handful of seeds are needed for a representative
trajectory plot (the main Table 2 statistics use the full 30-run results
from run_experiments.py instead).

Per-task errors (e.g. an upstream pymoo bug triggering on a degenerate
front -- this has been observed with AGE-MOEA2's survival-score
computation on certain seeds) are caught, logged to logs/, and skipped
rather than crashing the whole multiprocessing pool. Results are
checkpointed incrementally and the script is resumable, matching
run_experiments.py's behaviour.

Usage:
    python track_convergence.py --n-seeds 5 --n-jobs 8
"""
import argparse
import sys
import time
import traceback
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import pin_device  # noqa: E402

from config import PROBLEMS, ALGORITHMS, POP_SIZE, N_ARCHIVE, N_GEN, build_problem
from algorithms.factory import build_algorithm
from utils.metrics import igd
from utils.fs_safety import safe_name, prewarm_pareto_fronts

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONV_CSV = RESULTS_DIR / "convergence_trajectories.csv"
LOG_DIR = Path(__file__).resolve().parent / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Record IGD every `LOG_EVERY` generations to keep file size and runtime sane
LOG_EVERY = 5


def _track_one(args):
    """Track one (problem, algorithm, seed) run. Never raises -- any
    exception is caught, logged to disk, and an empty record list is
    returned so the calling pool keeps going."""
    problem_name, n_obj, kind, kwargs, algo_name, seed = args
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()

        algo = build_algorithm(algo_name, n_obj=n_obj, pop_size=POP_SIZE,
                                n_archive=N_ARCHIVE, seed=seed)
        algo.setup(problem, termination=("n_gen", N_GEN), seed=seed, verbose=False)

        records = []
        last_logged_gen = -1
        # DESIGN NOTE: algo.n_gen is not a reliable mid-run progress indicator
        # across all pymoo algorithm classes -- for MOEA/D specifically it does
        # not advance per logical generation the way it does for NSGA-II/KOA
        # (confirmed empirically: n_gen stayed pinned while next() kept
        # executing). algo.evaluator.n_eval (function evaluations), however,
        # is incremented consistently by every algorithm and is the basis
        # pymoo's own ('n_gen', N) termination criterion uses internally.
        # One "generation-equivalent" = pop_size evaluations beyond the
        # initial population, which gives a fair, comparable x-axis across
        # all algorithm types including MOEA/D's per-neighbour update scheme.
        safety_cap = N_GEN * POP_SIZE * 5  # generous upper bound on total evals
        while algo.has_next() and algo.evaluator.n_eval < safety_cap:
            algo.next()
            n_eval = algo.evaluator.n_eval
            gen_equiv = max(0, (n_eval - POP_SIZE)) // POP_SIZE
            if gen_equiv != last_logged_gen and (gen_equiv % LOG_EVERY == 0 or gen_equiv >= N_GEN):
                F = algo.pop.get("F")
                val = igd(F, pf)
                records.append({
                    "problem": problem_name, "algorithm": algo_name, "seed": seed,
                    "generation": gen_equiv, "IGD": val,
                })
                last_logged_gen = gen_equiv
            if gen_equiv >= N_GEN:
                break
        return records, None

    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"{type(e).__name__}: {e}"
        log_path = LOG_DIR / f"conv_error_{safe_name(problem_name)}_{safe_name(algo_name)}_seed{seed}.log"
        try:
            log_path.write_text(tb)
        except Exception:
            pass  # don't let logging failures mask the original error report
        return [], {
            "problem": problem_name, "algorithm": algo_name, "seed": seed,
            "error": err_msg,
        }


def load_checkpoint():
    if CONV_CSV.exists():
        df = pd.read_csv(CONV_CSV)
        done = set(zip(df["problem"], df["algorithm"], df["seed"]))
        return done
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5,
                         help="Number of seeds to track per (problem, algorithm)")
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--checkpoint-every", type=int, default=10,
                         help="Flush results to CSV every N completed tasks")
    parser.add_argument("--gpu", type=int, default=0,
                         help="CUDA_VISIBLE_DEVICES pin ('device port'); "
                              "no effect on this CPU-only workload's results.")
    args = parser.parse_args()
    pin_device(args.gpu)

    prewarm_pareto_fronts(PROBLEMS, build_problem)

    done = load_checkpoint()

    tasks = []
    for problem_name, n_obj, kind, kwargs in PROBLEMS:
        for algo_name in ALGORITHMS:
            for seed in range(1, args.n_seeds + 1):
                if (problem_name, algo_name, seed) in done:
                    continue
                tasks.append((problem_name, n_obj, kind, kwargs, algo_name, seed))

    total_all = len(PROBLEMS) * len(ALGORITHMS) * args.n_seeds
    print(f"Total (problem x algorithm x seed) combinations: {total_all}")
    print(f"Already completed (resumed from checkpoint): {len(done)}")
    print(f"Remaining to run: {len(tasks)}")

    if not tasks:
        print("Nothing to do -- all tracked runs already completed.")
        return

    buffer = []
    errors = []
    n_completed = 0
    t0 = time.time()

    with mp.Pool(processes=args.n_jobs) as pool:
        for records, error in pool.imap_unordered(_track_one, tasks):
            buffer.extend(records)
            if error is not None:
                errors.append(error)
            n_completed += 1

            if n_completed % args.checkpoint_every == 0 or n_completed == len(tasks):
                if buffer:
                    chunk = pd.DataFrame(buffer)
                    header = not CONV_CSV.exists()
                    chunk.to_csv(CONV_CSV, mode="a", header=header, index=False)
                    buffer = []
                elapsed = time.time() - t0
                print(f"[{n_completed}/{len(tasks)}] elapsed={elapsed/60:.1f}min "
                      f"({len(errors)} errors so far)")

    if errors:
        print(f"\nWARNING: {len(errors)} (problem, algorithm, seed) combinations "
              f"failed during convergence tracking and were skipped. "
              f"See logs/conv_error_*.log for tracebacks. Affected combinations:")
        for e in errors:
            print(f"  {e['problem']}/{e['algorithm']}/seed{e['seed']}: {e['error']}")
        print("\nThis does not affect Table 2 / run_experiments.py results -- "
              "only the convergence-curve figure for these specific lines will "
              "be missing or based on fewer seeds.")

    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.1f} minutes.")
    print(f"Results written incrementally to: {CONV_CSV}")


if __name__ == "__main__":
    main()
