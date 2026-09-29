"""
run_weight_sensitivity.py -- state-regulation coefficient sensitivity
for Eq. (3) (lambda_delta, lambda_kappa, a_start/a_end, and the p_exp
clip bounds).

FIX (sensitivity scripts using an outdated algorithm): this script used
to build `KOAStarR2` via `algorithms/koa_star_r2.py` and monkey-patch its
`_state_probabilities` method -- i.e. it was NOT sensitivity analysis of
the algorithm actually reported in the manuscript (`KOAFinalV3`). It now
imports `KOAFinalV3` directly and passes the coefficients as ordinary
constructor arguments, since Eq. (3)'s coefficients are now real,
first-class parameters of the canonical class (see koa_final_v3.py).

FIX (sensitivity coverage): the review noted the sensitivity study
should "cover at least one benchmark from every suite" rather than 2-3
hand-picked problems. This script now runs on one representative
problem from EVERY suite in config.PROBLEMS (ZDT, DTLZ, WFG, LSMOP,
MaF) by default, selectable with --problems.

Also sweeps the p_exp clip bounds (p_exp_min, p_exp_max), which the
review separately flagged as unexamined.

Usage:
    python run_weight_sensitivity.py --n-jobs 16 --gpu 0
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
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population
from utils.fs_safety import prewarm_pareto_fronts
from algorithms.koa_final_v3 import KOAFinalV3

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "weight_sensitivity.csv"

# (lambda_delta, lambda_kappa, a_start, a_end, p_exp_min, p_exp_max)
CONFIGS = {
    "default":  dict(lambda_delta=0.20, lambda_kappa=0.15, a_start=0.40, a_end=0.10, p_exp_min=0.05, p_exp_max=0.80),
    "config_A": dict(lambda_delta=0.25, lambda_kappa=0.10, a_start=0.30, a_end=0.20, p_exp_min=0.05, p_exp_max=0.80),  # stronger diversity push
    "config_B": dict(lambda_delta=0.15, lambda_kappa=0.20, a_start=0.50, a_end=0.05, p_exp_min=0.05, p_exp_max=0.80),  # stronger convergence pull
    "config_C": dict(lambda_delta=0.25, lambda_kappa=0.25, a_start=0.25, a_end=0.25, p_exp_min=0.05, p_exp_max=0.80),  # equal weights
    "narrow_clip": dict(lambda_delta=0.20, lambda_kappa=0.15, a_start=0.40, a_end=0.10, p_exp_min=0.15, p_exp_max=0.60),
    "wide_clip":   dict(lambda_delta=0.20, lambda_kappa=0.15, a_start=0.40, a_end=0.10, p_exp_min=0.01, p_exp_max=0.95),
}

# One representative problem per benchmark suite (review: "cover at
# least one benchmark from every suite", not just 2-3 hand-picked ones).
DEFAULT_PROBLEMS = [
    ("ZDT3",   2, "zdt",   {"name": "zdt3"}),
    ("DTLZ2",  3, "dtlz",  {"name": "dtlz2", "n_var": 12, "n_obj": 3}),
    ("WFG4",   3, "wfg",   {"name": "wfg4", "n_var": 10, "n_obj": 3}),
    ("LSMOP9", 3, "lsmop", {"which": 9, "n_obj": 3, "n_var": 60}),
    ("MaF4",   3, "maf",   {"which": 4, "n_obj": 3, "n_var": None}),
]
N_SEEDS = 30


def _run_one(args):
    cfg_name, cfg, pname, n_obj, kind, kwargs, seed = args
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        ref = compute_reference_point(pf)
        init_X = shared_initial_population(problem, POP_SIZE, seed)
        algo = KOAFinalV3(pop_size=POP_SIZE, n_archive=N_ARCHIVE, seed=seed,
                           init_X=init_X, **cfg)
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
        F = res.F
        if F is None or len(F) == 0:
            raise ValueError("empty front")
        m = compute_all_metrics(F, pf, ref)
        n_eval = int(res.algorithm.evaluator.n_eval)
        status = "ok"
    except Exception as e:
        m = {"IGD": np.nan, "HV": np.nan, "Spacing": np.nan, "Spread": np.nan}
        n_eval = 0
        status = f"error:{type(e).__name__}:{e}"

    return {"config": cfg_name, "problem": pname, "seed": seed,
            "n_eval": n_eval, "fe_budget": FE_BUDGET, "status": status, **m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    pin_device(args.gpu)

    prewarm_pareto_fronts(DEFAULT_PROBLEMS, build_problem)

    rows, done = [], set()
    if OUT_CSV.exists():
        df_ex = pd.read_csv(OUT_CSV)
        rows = df_ex.to_dict("records")
        done = {(r["config"], r["problem"], r["seed"]) for r in rows}

    tasks = [(cfg_name, cfg, pname, n_obj, kind, kwargs, seed)
              for cfg_name, cfg in CONFIGS.items()
              for pname, n_obj, kind, kwargs in DEFAULT_PROBLEMS
              for seed in range(1, args.n_seeds + 1)
              if (cfg_name, pname, seed) not in done]

    print(f"Weight sensitivity: {len(tasks)} remaining "
          f"(configs={list(CONFIGS)}, problems={[p[0] for p in DEFAULT_PROBLEMS]})")
    t0 = time.time()

    with mp.Pool(args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks)):
            rows.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  [{i+1}/{len(tasks)}] "
                      f"{result['config']}/{result['problem']}/s{result['seed']} "
                      f"elapsed={(time.time()-t0)/60:.1f}min status={result['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== WEIGHT SENSITIVITY SUMMARY (mean IGD, % change from default) ===")
    ok = df[df.status == "ok"]
    pivot = ok.groupby(["problem", "config"])["IGD"].mean().unstack()
    if "default" in pivot.columns:
        ref = pivot["default"]
        for col in pivot.columns:
            pivot[f"{col}_norm"] = (pivot[col] / ref * 100 - 100).round(1)
        print(pivot[[c for c in pivot.columns if "norm" in c]].to_string())
    else:
        print(pivot.to_string())


if __name__ == "__main__":
    main()
