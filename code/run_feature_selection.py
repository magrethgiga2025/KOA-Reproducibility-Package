"""
run_feature_selection.py
========================
Real-world application: bi-objective wrapper feature selection
(f1 = fraction of selected features, f2 = inner-CV k-NN error; see fs_problem.py).

Uses EXACTLY the same machinery as the benchmark study:
  * algorithms from algorithms/factory.py (KOA = KOAFinalV3, same operators
    and settings for every baseline),
  * one shared LHS initial population per (dataset, seed), given to every algorithm,
  * identical budget: POP_SIZE individuals x N_GEN_FS generations for every algorithm,
  * paired seeds 1..30; seed k also fixes the train/test split of run k.

Usage (from the code/ directory):
    python run_feature_selection.py                         # everything
    python run_feature_selection.py --algorithms KOA NSGA-II --runs 2 --datasets Wine   # quick check
    python run_feature_selection.py --n-jobs 16

Each finished run is saved to results/feature_selection/<dataset>/<algorithm>/run_XX.npz
and skipped on restart. Analyse with:  python analyze_feature_selection.py
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from joblib import Parallel, delayed

from config import POP_SIZE, N_ARCHIVE, ALGORITHMS
from fs_problem import FeatureSelectionProblem, as_pymoo, final_front

N_GEN_FS = 100          # 100 individuals x 100 generations = 10,000 evaluations per run
N_RUNS_FS = 30
OUT_DIR = ROOT / "results" / "feature_selection"

DEFAULT_DATASETS = ["Wine", "WDBC", "Ionosphere", "Sonar", "Digits",
                    "HillValley", "Musk1", "Arrhythmia"]


def out_path(dataset, algo, seed):
    return OUT_DIR / dataset / algo.replace("/", "_") / f"run_{seed:02d}.npz"


def one_run(dataset, algo_name, seed):
    path = out_path(dataset, algo_name, seed)
    if path.exists():
        return f"skip  {dataset:<11} {algo_name:<10} seed {seed}"
    try:
        from pymoo.optimize import minimize
        from algorithms.factory import build_algorithm
        from utils.common_init import shared_initial_population

        fs = FeatureSelectionProblem(dataset, seed)
        problem = as_pymoo(fs)
        init_X = shared_initial_population(problem, POP_SIZE, seed)
        algo = build_algorithm(algo_name, n_obj=2, pop_size=POP_SIZE,
                               n_archive=N_ARCHIVE, seed=seed, init_X=init_X)
        t0 = time.time()
        res = minimize(problem, algo, ("n_gen", N_GEN_FS), seed=seed, verbose=False)
        wall = time.time() - t0

        X = res.pop.get("X")                    # final population, every algorithm
        fr = final_front(fs, X)                 # unique non-dominated subsets + test error
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, n_evals=int(res.algorithm.evaluator.n_eval),
                            runtime=wall, n_features=fs.n_var, **fr)
        return (f"done  {dataset:<11} {algo_name:<10} seed {seed}  "
                f"evals={res.algorithm.evaluator.n_eval}  {wall:.0f}s")
    except Exception as e:
        log = ROOT / "logs" / f"error_fs_{dataset}_{algo_name.replace('/', '_')}_seed{seed}.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text(traceback.format_exc())
        return f"FAIL  {dataset:<11} {algo_name:<10} seed {seed}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--algorithms", nargs="+", default=ALGORITHMS, choices=ALGORITHMS)
    ap.add_argument("--runs", type=int, default=N_RUNS_FS)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    jobs = [(d, alg, s) for d in a.datasets for alg in a.algorithms
            for s in range(1, a.runs + 1)]
    print(f"{len(jobs)} runs = {len(a.datasets)} datasets x {len(a.algorithms)} algorithms "
          f"x {a.runs} seeds | pop={POP_SIZE}, generations={N_GEN_FS}")
    for msg in Parallel(n_jobs=a.n_jobs, return_as="generator")(
            delayed(one_run)(d, alg, s) for d, alg, s in jobs):
        print(msg, flush=True)


if __name__ == "__main__":
    main()
