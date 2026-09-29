"""
run_delta_diagnostic.py
========================
Targeted diagnostic requested by the review: compare the ORIGINAL raw,
unnormalised SHADE success score

    Delta_i = max(0, sum_j f_j(parent) - sum_j f_j(child))

against the FIXED, normalised default (Fix 2 in koa_final_v3.py)

    Delta_i = max(0, sum_j (f_j(parent) - f_j(child)) / (s_j + eps))

on MaF4 (and, for context, MaF7 and DTLZ4 as scaled/unscaled controls),
to experimentally verify the manuscript's diagnosis that MaF4's
catastrophic HV/IGD failure is driven by the raw objective-sum success
metric letting one large-scale objective dominate the SHADE memory
update.

This is intentionally a SEPARATE, small, targeted study -- not part of
the main ablation table (run_ablation_final.py) -- because it isolates
one specific mechanism on the one problem where it is diagnostic,
rather than diluting the main ablation with a variant that is only
interesting on a single benchmark.

Usage:
    python run_delta_diagnostic.py --n-jobs 8 --gpu 0
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
OUT_CSV = RESULTS_DIR / "delta_diagnostic.csv"

VARIANTS = {"raw": "raw", "normalized": "normalized"}
PROBLEMS = [
    ("MaF4",  3, "maf",  {"which": 4, "n_obj": 3, "n_var": None}),   # badly scaled -- the failure case
    ("MaF7",  3, "maf",  {"which": 7, "n_obj": 3, "n_var": None}),   # disconnected, well scaled -- control
    ("DTLZ4", 3, "dtlz", {"name": "dtlz4", "n_var": 12, "n_obj": 3}),  # well scaled -- control
]
N_SEEDS = 30


def _run_one(args):
    metric_name, success_metric, pname, n_obj, kind, kwargs, seed = args
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        ref = compute_reference_point(pf)
        init_X = shared_initial_population(problem, POP_SIZE, seed)
        algo = KOAFinalV3(pop_size=POP_SIZE, n_archive=N_ARCHIVE, seed=seed,
                           init_X=init_X, success_metric=success_metric)
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
        F = res.F
        if F is None or len(F) == 0:
            raise ValueError("empty front")
        m = compute_all_metrics(F, pf, ref)
        status = "ok"
    except Exception as e:
        m = {"HV": np.nan, "IGD": np.nan, "Spacing": np.nan, "Spread": np.nan}
        status = f"error:{type(e).__name__}:{e}"
    return {"delta_metric": metric_name, "problem": pname, "seed": seed,
            "fe_budget": FE_BUDGET, "status": status, **m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    pin_device(args.gpu)

    prewarm_pareto_fronts(PROBLEMS, build_problem)

    rows, done = [], set()
    if OUT_CSV.exists():
        df_ex = pd.read_csv(OUT_CSV)
        rows = df_ex.to_dict("records")
        done = {(r["delta_metric"], r["problem"], r["seed"]) for r in rows}

    tasks = [(name, sm, pname, n_obj, kind, kw, seed)
              for name, sm in VARIANTS.items()
              for pname, n_obj, kind, kw in PROBLEMS
              for seed in range(1, args.n_seeds + 1)
              if (name, pname, seed) not in done]

    print(f"Delta_i diagnostic: {len(tasks)} remaining")
    t0 = time.time()
    with mp.Pool(args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks)):
            rows.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  [{i+1}/{len(tasks)}] elapsed={(time.time()-t0)/60:.1f}min "
                      f"{result['delta_metric']}/{result['problem']}/s{result['seed']} "
                      f"status={result['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== RAW vs NORMALISED Delta_i (mean HV / mean IGD) ===")
    ok = df[df.status == "ok"]
    for pname, *_ in PROBLEMS:
        sub = ok[ok.problem == pname]
        print(f"\n{pname}:")
        for name in VARIANTS:
            s = sub[sub.delta_metric == name]
            print(f"  {name:12s}  HV={s['HV'].mean():.4f}±{s['HV'].std():.4f}  "
                  f"IGD={s['IGD'].mean():.4f}±{s['IGD'].std():.4f}  (n={len(s)})")


if __name__ == "__main__":
    main()
