"""
run_many_objective.py
======================
Answers the review's single biggest "not attempted at all" gap:
KOA's many-objective scalability was never evaluated beyond M=3, despite
several baselines (NSGA-III, RVEA, AGE-MOEA2) being specifically designed
for higher-dimensional objective spaces.

Runs all 7 algorithms x DTLZ2, DTLZ7, WFG4, MaF4 x M in {3, 5, 8, 10}
x 30 seeds, using the exact same fairness machinery as the main study
(algorithms/factory.py's exact-count reference directions, shared LHS
initial population, identical FE budget for every algorithm at every M).

This is a genuinely large compute job: 4 problems x 4 M-values x 7
algorithms x 30 seeds = 3,360 runs, the same order of magnitude as the
entire original main study, and reference-direction construction for
NSGA-III/MOEA-D/RVEA gets more expensive as M grows (Riesz s-Energy
optimisation; cached per-process via lru_cache in factory.py so repeat
seeds on the same worker don't re-pay this cost, but the first run at
each M on each worker does). Expect this to be the longest-running of
the follow-up studies. --n-jobs should be your full physical core
count; use --gpu for shared-server device pinning only (this workload
does not use a GPU, see device.py).

Output: results/many_objective_raw.csv
Columns: problem, M, algorithm, seed, HV, IGD, n_eval, fe_budget, status
"""
import argparse
import sys
import time
import traceback
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from device import pin_device  # noqa: E402

import numpy as np
import pandas as pd
from pymoo.optimize import minimize

from config import POP_SIZE, N_ARCHIVE, N_GEN, FE_BUDGET, ALGORITHMS
from algorithms.factory import build_algorithm
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population
from utils.fs_safety import safe_name
from utils.many_objective import (
    build_many_objective_problem, many_objective_pareto_front,
    MANY_OBJECTIVE_PROBLEMS, M_VALUES,
)

RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "many_objective_raw.csv"

N_RUNS = 30


def _run_one(args):
    prob_name, M, algo_name, seed = args
    t0 = time.time()
    try:
        problem, _ = build_many_objective_problem(prob_name, M)
        pf = many_objective_pareto_front(problem, prob_name, M, n_points=1000)
        if pf is None or len(pf) == 0:
            raise RuntimeError("empty/None pareto_front()")
        ref = compute_reference_point(pf)

        init_X = shared_initial_population(problem, POP_SIZE, seed)
        algo = build_algorithm(algo_name, n_obj=M, pop_size=POP_SIZE,
                                n_archive=N_ARCHIVE, seed=seed, init_X=init_X)
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)

        F = res.F
        n_eval = int(res.algorithm.evaluator.n_eval)
        if F is None or len(F) == 0:
            m = {"HV": 0.0, "IGD": np.inf, "Spacing": np.inf, "Spread": np.inf}
            status = "empty_front"
        else:
            m = compute_all_metrics(F, pf, ref)
            status = "ok"
    except Exception as e:
        m = {"HV": np.nan, "IGD": np.nan, "Spacing": np.nan, "Spread": np.nan}
        n_eval = 0
        status = f"error:{type(e).__name__}"
        tb = traceback.format_exc()
        log_path = LOG_DIR / f"error_manyobj_{safe_name(prob_name)}_M{M}_{safe_name(algo_name)}_seed{seed}.log"
        with open(log_path, "w") as f:
            f.write(tb)

    return {"problem": prob_name, "M": M, "algorithm": algo_name, "seed": seed,
            "HV": m["HV"], "IGD": m["IGD"], "Spacing": m["Spacing"], "Spread": m["Spread"],
            "n_eval": n_eval, "fe_budget": FE_BUDGET,
            "wall_seconds": round(time.time() - t0, 2), "status": status}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--problems", nargs="+", default=MANY_OBJECTIVE_PROBLEMS,
                         choices=MANY_OBJECTIVE_PROBLEMS)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    pin_device(args.gpu)

    done = set()
    rows = []
    if OUT_CSV.exists():
        df0 = pd.read_csv(OUT_CSV)
        rows = df0.to_dict("records")
        done = set(zip(df0.problem, df0.M, df0.algorithm, df0.seed))

    tasks = [(pname, M, algo, seed)
              for pname in args.problems
              for M in args.m_values
              for algo in ALGORITHMS
              for seed in range(1, args.n_runs + 1)
              if (pname, M, algo, seed) not in done]

    total = len(args.problems) * len(args.m_values) * len(ALGORITHMS) * args.n_runs
    print(f"Many-objective study: {total} total | {len(done)} done | "
          f"{len(tasks)} remaining")
    print(f"Problems={args.problems}  M values={args.m_values}  "
          f"algorithms={ALGORITHMS}")
    if not tasks:
        print("Nothing to run.")
        return

    t0 = time.time()
    with mp.Pool(args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks)):
            rows.append(result)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                elapsed = (time.time() - t0) / 60
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(tasks) - i - 1) / rate if rate > 0 else float("inf")
                print(f"  [{i+1}/{len(tasks)}] elapsed={elapsed:.1f}min "
                      f"est_remaining={remaining:.1f}min "
                      f"last={result['problem']}/M{result['M']}/{result['algorithm']}/"
                      f"seed{result['seed']} status={result['status']}")

    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.1f} min. "
          f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
