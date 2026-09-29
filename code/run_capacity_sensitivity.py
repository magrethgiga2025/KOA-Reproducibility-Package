"""
run_capacity_sensitivity.py
=============================
Archive-capacity sensitivity: N_arch/N in {0.5, 1.0, 1.2, 1.5, 2.0},
i.e. N_arch in {50, 100, 120, 150, 200} at the study's N=100 -- exactly
the ratios the review names explicitly. Answers "sensitivity to
N_arch is particularly important and deserves sensitivity analysis."

Covers one representative problem per benchmark suite (same set used
by run_theta_sensitivity.py / run_weight_sensitivity.py, for the same
"cover every suite" reason).

Output: results/capacity_sensitivity.csv
Usage: python run_capacity_sensitivity.py --n-jobs 16 --gpu 0
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

from config import build_problem, POP_SIZE, N_GEN, FE_BUDGET
from algorithms.koa_final_v3 import KOAFinalV3
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population
from utils.fs_safety import prewarm_pareto_fronts

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "capacity_sensitivity.csv"

# N_arch/N ratios the review names explicitly.
RATIOS = [0.5, 1.0, 1.2, 1.5, 2.0]

PROBLEMS = [
    ("ZDT3",   2, "zdt",   {"name": "zdt3"}),
    ("DTLZ2",  3, "dtlz",  {"name": "dtlz2", "n_var": 12, "n_obj": 3}),
    ("WFG4",   3, "wfg",   {"name": "wfg4", "n_var": 10, "n_obj": 3}),
    ("LSMOP9", 3, "lsmop", {"which": 9, "n_obj": 3, "n_var": 60}),
    ("MaF4",   3, "maf",   {"which": 4, "n_obj": 3, "n_var": None}),
]
N_SEEDS = 30


def _run_one(args):
    ratio, pname, n_obj, kind, kwargs, seed = args
    n_arch = max(1, int(round(POP_SIZE * ratio)))
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        ref = compute_reference_point(pf)
        init_X = shared_initial_population(problem, POP_SIZE, seed)
        algo = KOAFinalV3(pop_size=POP_SIZE, n_archive=n_arch, seed=seed, init_X=init_X)
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
        F = res.F
        if F is None or len(F) == 0:
            raise ValueError("empty front")
        m = compute_all_metrics(F, pf, ref)
        n_eval = int(res.algorithm.evaluator.n_eval)
        status = "ok"
    except Exception as e:
        m = {"HV": np.nan, "IGD": np.nan, "Spacing": np.nan, "Spread": np.nan}
        n_eval = 0
        status = f"error:{type(e).__name__}:{e}"

    return {"ratio": ratio, "n_arch": n_arch, "problem": pname, "seed": seed,
            "n_eval": n_eval, "fe_budget": FE_BUDGET, "status": status, **m}


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
        df0 = pd.read_csv(OUT_CSV)
        rows = df0.to_dict("records")
        done = {(r["ratio"], r["problem"], r["seed"]) for r in rows}

    tasks = [(ratio, pname, n_obj, kind, kw, seed)
              for ratio in RATIOS
              for pname, n_obj, kind, kw in PROBLEMS
              for seed in range(1, args.n_seeds + 1)
              if (ratio, pname, seed) not in done]

    print(f"Capacity sensitivity: {len(tasks)} remaining "
          f"(ratios={RATIOS}, problems={[p[0] for p in PROBLEMS]})")
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
                      f"ratio={result['ratio']} {result['problem']} status={result['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    ok = df[df.status == "ok"]
    print("\n=== CAPACITY SENSITIVITY (mean IGD) ===")
    print(ok.groupby(["problem", "ratio"])["IGD"].mean().unstack().to_string())


if __name__ == "__main__":
    main()
