"""
run_factorial_ablation.py
===========================
Full 2x2x2 factorial ablation (State regulation x Archive-guided search
x Geometry-adaptive survival), answering the review's specific request
for evidence beyond "removing one component at a time doesn't help" --
a factorial design can actually distinguish additive contributions from
interaction effects, which the single-mechanism ablation table cannot.

Run on 6 representative problems -- reusing the exact same "one per
suite, chosen to show a different optimisation characteristic" set the
manuscript already justifies for its convergence-curve figure (Section
5.3): ZDT3 (disconnected), DTLZ6 (deceptive distance function), WFG6
(non-separable), LSMOP1 (large-scale), MaF7 (disconnected+many-region),
MaF4 (severely scaled) -- rather than all 16, matching the review's own
phrasing ("for representative benchmark problems, compare State only,
Archive only, Geometry only, ...").

Output: results/factorial_ablation_raw.csv
Columns: problem, cell, state_on, archive_on, geometry_on, seed, HV, IGD,
         n_eval, fe_budget, status

Usage:
    python run_factorial_ablation.py --n-jobs 16 --gpu 0
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

from config import POP_SIZE, N_ARCHIVE, N_GEN, FE_BUDGET, build_problem
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population
from utils.fs_safety import prewarm_pareto_fronts
from algorithms.koa_ablation_final import build_factorial_variant, FACTORIAL_CELLS

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "factorial_ablation_raw.csv"

# Same representative-problem set as Section 5.3's convergence figure.
REPRESENTATIVE_PROBLEMS = [
    ("ZDT3",   2, "zdt",   dict(name="zdt3")),
    ("DTLZ6",  3, "dtlz",  dict(name="dtlz6", n_var=12, n_obj=3)),
    ("WFG6",   3, "wfg",   dict(name="wfg6", n_var=10, n_obj=3)),
    ("LSMOP1", 3, "lsmop", dict(which=1, n_obj=3, n_var=60)),
    ("MaF7",   3, "maf",   dict(which=7, n_obj=3, n_var=None)),
    ("MaF4",   3, "maf",   dict(which=4, n_obj=3, n_var=None)),
]
N_SEEDS = 30


def _run_one(args):
    pname, n_obj, kind, kwargs, cell, seed = args
    state_on, archive_on, geometry_on = cell
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        ref = compute_reference_point(pf)
        init_X = shared_initial_population(problem, POP_SIZE, seed)

        algo, label = build_factorial_variant(
            state_on, archive_on, geometry_on,
            pop_size=POP_SIZE, n_archive=N_ARCHIVE, seed=seed, init_X=init_X,
        )
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
        F = res.F
        if F is None or len(F) == 0:
            raise ValueError("empty front")
        m = compute_all_metrics(F, pf, ref)
        n_eval = int(res.algorithm.evaluator.n_eval)
        status = "ok"
    except Exception as e:
        label = "".join(["S" if state_on else "s",
                          "A" if archive_on else "a",
                          "G" if geometry_on else "g"])
        m = {"HV": np.nan, "IGD": np.nan, "Spacing": np.nan, "Spread": np.nan}
        n_eval = 0
        status = f"error:{type(e).__name__}:{e}"

    return {"problem": pname, "cell": label,
            "state_on": state_on, "archive_on": archive_on, "geometry_on": geometry_on,
            "seed": seed, "HV": m["HV"], "IGD": m["IGD"],
            "Spacing": m["Spacing"], "Spread": m["Spread"],
            "n_eval": n_eval, "fe_budget": FE_BUDGET, "status": status}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    pin_device(args.gpu)

    prewarm_pareto_fronts(REPRESENTATIVE_PROBLEMS, build_problem)

    rows, done = [], set()
    if OUT_CSV.exists():
        df0 = pd.read_csv(OUT_CSV)
        rows = df0.to_dict("records")
        done = {(r["problem"], r["cell"], r["seed"]) for r in rows}

    tasks = []
    for pname, n_obj, kind, kwargs in REPRESENTATIVE_PROBLEMS:
        for cell in FACTORIAL_CELLS:
            label = "".join(["S" if cell[0] else "s",
                              "A" if cell[1] else "a",
                              "G" if cell[2] else "g"])
            for seed in range(1, args.n_seeds + 1):
                if (pname, label, seed) not in done:
                    tasks.append((pname, n_obj, kind, kwargs, cell, seed))

    total = len(REPRESENTATIVE_PROBLEMS) * len(FACTORIAL_CELLS) * args.n_seeds
    print(f"Factorial ablation: {total} total | {len(done)} done | "
          f"{len(tasks)} remaining | 8 cells x 6 problems x {args.n_seeds} seeds")
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
                      f"{result['problem']}/{result['cell']} status={result['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    ok = df[df.status == "ok"]
    print("\n=== FACTORIAL CELL MEANS (IGD, across all 6 problems -- see "
          "analyze_ablation.py for the proper per-problem-rank version) ===")
    print(ok.groupby("cell")["IGD"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
