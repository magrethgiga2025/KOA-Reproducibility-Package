"""
run_full_param_sensitivity.py
================================
One-at-a-time sensitivity sweep for every KOA parameter the review
listed as unexamined (theta and the Eq.3 coefficients already have
their own scripts -- run_theta_sensitivity.py, run_weight_sensitivity.py;
archive capacity has its own -- run_capacity_sensitivity.py):

    alpha0 (base step amplitude), A (oscillation amplitude),
    tau (oscillation period), L_H (SHADE memory size),
    Gamma_b (stagnation patience), Gamma_c (recovery cooldown),
    Levy probability

Each parameter is varied one at a time around its default while every
other parameter is held at its default value, across one representative
problem per suite (same set used elsewhere in this repository).

Also records `p_exp_trace` (min/max/mean of the actual p_exp(t) values
visited during each run -- see koa_final_v3.py) for every run, so that,
unlike the earlier ad hoc weight-sensitivity check, whether a parameter
change actually affects the algorithm's behaviour (rather than
producing a silently unreachable configuration) is visible directly in
the output CSV rather than inferred after the fact.

Output: results/full_param_sensitivity.csv
Usage: python run_full_param_sensitivity.py --n-jobs 16 --gpu 0
"""
import argparse
import sys
import time
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from device import pin_device  # noqa: E402

import numpy as np
import pandas as pd
from pymoo.optimize import minimize

from config import build_problem, POP_SIZE, N_ARCHIVE, N_GEN, FE_BUDGET
from algorithms.koa_final_v3 import KOAFinalV3
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population
from utils.fs_safety import prewarm_pareto_fronts

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "full_param_sensitivity.csv"

PROBLEMS = [
    ("ZDT3",   2, "zdt",   {"name": "zdt3"}),
    ("DTLZ2",  3, "dtlz",  {"name": "dtlz2", "n_var": 12, "n_obj": 3}),
    ("WFG4",   3, "wfg",   {"name": "wfg4", "n_var": 10, "n_obj": 3}),
    ("LSMOP9", 3, "lsmop", {"which": 9, "n_obj": 3, "n_var": 60}),
    ("MaF4",   3, "maf",   {"which": 4, "n_obj": 3, "n_var": None}),
]
N_SEEDS = 30

# Each entry: kwarg name -> (default_value, [values to test, INCLUDING default])
PARAM_GRID = {
    "alpha0":        (0.30, [0.15, 0.30, 0.45, 0.60]),
    "A":             (0.15, [0.05, 0.15, 0.25]),
    "tau":           (50,   [25, 50, 100]),
    "shade_memory":  (10,   [5, 10, 20]),   # L_H
    "gamma_b":       (10,   [5, 10, 20]),   # Gamma_b
    "gamma_c":       (15,   [10, 15, 25]),  # Gamma_c
    "levy_prob":     (0.15, [0.05, 0.15, 0.30]),
}


def _run_one(args):
    param, value, pname, n_obj, kind, kwargs, seed = args
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        ref = compute_reference_point(pf)
        init_X = shared_initial_population(problem, POP_SIZE, seed)
        algo = KOAFinalV3(pop_size=POP_SIZE, n_archive=N_ARCHIVE, seed=seed,
                           init_X=init_X, **{param: value})
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
        F = res.F
        if F is None or len(F) == 0:
            raise ValueError("empty front")
        m = compute_all_metrics(F, pf, ref)
        n_eval = int(res.algorithm.evaluator.n_eval)
        trace = res.algorithm.p_exp_trace
        p_exp_min = trace["min"]
        p_exp_max = trace["max"]
        p_exp_mean = trace["sum"] / trace["n"] if trace["n"] else np.nan
        status = "ok"
    except Exception as e:
        m = {"HV": np.nan, "IGD": np.nan, "Spacing": np.nan, "Spread": np.nan}
        n_eval = 0
        p_exp_min = p_exp_max = p_exp_mean = np.nan
        status = f"error:{type(e).__name__}:{e}"

    return {"param": param, "value": value, "problem": pname, "seed": seed,
            "n_eval": n_eval, "fe_budget": FE_BUDGET, "status": status,
            "p_exp_min": p_exp_min, "p_exp_max": p_exp_max, "p_exp_mean": p_exp_mean,
            **m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--params", nargs="+", default=list(PARAM_GRID.keys()),
                         choices=list(PARAM_GRID.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    pin_device(args.gpu)

    prewarm_pareto_fronts(PROBLEMS, build_problem)

    rows, done = [], set()
    if OUT_CSV.exists():
        df0 = pd.read_csv(OUT_CSV)
        rows = df0.to_dict("records")
        done = {(r["param"], r["value"], r["problem"], r["seed"]) for r in rows}

    tasks = []
    for param in args.params:
        _, values = PARAM_GRID[param]
        for value in values:
            for pname, n_obj, kind, kw in PROBLEMS:
                for seed in range(1, args.n_seeds + 1):
                    if (param, value, pname, seed) not in done:
                        tasks.append((param, value, pname, n_obj, kind, kw, seed))

    print(f"Full parameter sensitivity: {len(tasks)} remaining "
          f"(params={args.params})")
    if not tasks:
        print("Nothing to run.")
        return

    t0 = time.time()
    with mp.Pool(args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks)):
            rows.append(result)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  [{i+1}/{len(tasks)}] elapsed={(time.time()-t0)/60:.1f}min "
                      f"{result['param']}={result['value']} {result['problem']} "
                      f"status={result['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    ok = df[df.status == "ok"]
    print("\n=== FULL PARAMETER SENSITIVITY (mean IGD) ===")
    for param in args.params:
        sub = ok[ok.param == param]
        print(f"\n{param}:")
        print(sub.groupby(["problem", "value"])["IGD"].mean().unstack().to_string())


if __name__ == "__main__":
    main()
