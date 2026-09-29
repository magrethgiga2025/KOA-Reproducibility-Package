"""
run_ablation_final.py  --  Ablation study for KOA Final V3.

Variants (algorithms.koa_ablation_final.MAIN_ABLATION_VARIANTS):
    KOA, KOA-NoState, KOA-NoArchive, KOA-NoRefresh, KOA-NoScaleNorm,
    KOA-NoPopSeed, KOA-FixedRho0.2, KOA-FixedRho0.6, KOA-FixedRho1.0

The KOA-FixedRho* variants are the previously-missing geometry-adaptive-
survival ablation (Fix 6): everything about KOA is held identical except
that rho(t) is replaced by a constant. KOA-RawDelta (the raw-vs-
normalised Delta_i diagnostic) is intentionally NOT part of this table --
see run_delta_diagnostic.py, which runs it only on MaF4/MaF7 as the
targeted diagnostic the review asked for.

Usage:
    python run_ablation_final.py --n-jobs 16 --gpu 0
"""
import argparse, sys, time, traceback
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from device import pin_device  # noqa: E402

import numpy as np
import pandas as pd
from pymoo.optimize import minimize

from config import PROBLEMS, PROBLEM_NAMES, POP_SIZE, N_ARCHIVE, N_GEN, FE_BUDGET, build_problem
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population
from utils.fs_safety import prewarm_pareto_fronts
from algorithms.koa_ablation_final import build_ablation, MAIN_ABLATION_VARIANTS

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "ablation_final_raw.csv"

VARIANTS = MAIN_ABLATION_VARIANTS
N_RUNS = 30


def _run_one(args):
    prob_name, n_obj, kind, kwargs, variant, seed = args
    t0 = time.time()
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        if pf is None or len(pf) == 0:
            raise RuntimeError(f"No PF for {prob_name}")
        ref = compute_reference_point(pf)

        # Same shared initial population used across every variant, for
        # the same reason it is used across every algorithm in
        # run_experiments.py -- ablation variants are only comparable to
        # each other if they share the same starting point per seed.
        init_X = shared_initial_population(problem, POP_SIZE, seed)

        algo = build_ablation(variant, pop_size=POP_SIZE, n_archive=N_ARCHIVE,
                               seed=seed, init_X=init_X)
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)

        # Fix 7: res.F is the final population's non-dominated subset for
        # every variant (all are KOAFinalV3 configurations).
        F = res.F
        n_eval = int(res.algorithm.evaluator.n_eval)
        if F is None or len(F) == 0:
            m = {"HV": 0.0, "IGD": np.inf}
            status = "empty"
        else:
            m = compute_all_metrics(F, pf, ref)
            status = "ok"
    except Exception as e:
        m = {"HV": np.nan, "IGD": np.nan}
        n_eval = 0
        status = f"error:{type(e).__name__}"
        traceback.print_exc()
    return {"Problem": prob_name, "Variant": variant, "seed": seed,
            "HV": m["HV"], "IGD": m["IGD"], "n_eval": n_eval, "fe_budget": FE_BUDGET,
            "wall_s": round(time.time() - t0, 2), "status": status}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--gpu", type=int, default=0,
                         help="CUDA_VISIBLE_DEVICES pin ('device port'); "
                              "no effect on this CPU-only workload's results.")
    args = parser.parse_args()
    pin_device(args.gpu)

    prewarm_pareto_fronts(PROBLEMS, build_problem)

    done = set()
    if OUT_CSV.exists():
        df0 = pd.read_csv(OUT_CSV)
        done = set(zip(df0.Problem, df0.Variant, df0.seed))

    tasks = [(pname, n_obj, kind, kw, var, seed)
              for pname, n_obj, kind, kw in PROBLEMS
              for var in VARIANTS
              for seed in range(1, args.n_runs + 1)
              if (pname, var, seed) not in done]

    total = len(PROBLEMS) * len(VARIANTS) * args.n_runs
    print(f"Ablation final: {total} total | {len(done)} done | "
          f"{len(tasks)} remaining | variants={VARIANTS}")
    if not tasks:
        print("Nothing to run.")
        return

    rows = (list(pd.read_csv(OUT_CSV).to_dict("records"))
            if OUT_CSV.exists() else [])
    t0 = time.time()

    with mp.Pool(args.n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks)):
            rows.append(result)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  [{i+1}/{len(tasks)}] "
                      f"{result['Problem']}/{result['Variant']} "
                      f"elapsed={(time.time()-t0)/60:.1f}min "
                      f"status={result['status']}")

    df = pd.read_csv(OUT_CSV)
    ok = df[df.status == "ok"]
    print("\n=== ABLATION SUMMARY (mean IGD, all problems) ===")
    for v in VARIANTS:
        sub = ok[ok.Variant == v]["IGD"].replace([np.inf, -np.inf], np.nan)
        print(f"  {v:22s}: mean={sub.mean():.4f}  std={sub.std():.4f}")


if __name__ == "__main__":
    main()
