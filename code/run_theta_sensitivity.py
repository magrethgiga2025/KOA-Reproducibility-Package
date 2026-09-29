"""
run_theta_sensitivity.py
========================
Sensitivity analysis for the connectivity threshold theta in KOA's
geometry-adaptive survival mechanism (Eq. 10/13).

FIX (sensitivity scripts using an outdated algorithm): this script used
to build `KOAStar` (algorithms/koa_star.py), an earlier, superseded
algorithm variant -- NOT the algorithm any reported number comes from
(`KOAFinalV3`). It now imports `KOAFinalV3` directly.

FIX (sensitivity coverage): the review's "only three problems, and the
breadth of the robustness claim exceeds the evidence" comment is
addressed by covering one representative problem from EVERY benchmark
suite (ZDT, DTLZ, WFG, LSMOP, MaF), not just ZDT3/LSMOP9/MaF4.

Evaluates theta in {0.05, 0.10, 0.15, 0.20, 0.25} on each problem below,
30 independent seeds per (theta, problem) pair, sharing the common
initial population per seed used elsewhere in this repository.

Outputs
-------
  results/theta_sensitivity.csv
      Columns: theta, problem, seed, IGD, HV, n_eval, fe_budget, status
  plots/theta_sensitivity.pdf / .png

Usage
-----
  python run_theta_sensitivity.py --n-jobs 16 --gpu 0
"""
import argparse
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymoo.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from device import pin_device  # noqa: E402

from config import build_problem, POP_SIZE, N_ARCHIVE, N_GEN, FE_BUDGET
from algorithms.koa_final_v3 import KOAFinalV3
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population
from utils.fs_safety import prewarm_pareto_fronts

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
THETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]

# One representative problem per benchmark suite (Fix: sensitivity
# coverage). ZDT3/LSMOP9/MaF4 are kept from the original study because
# they are the disconnected/badly-scaled cases where theta is most
# likely to matter; DTLZ2 and WFG6 are added for a regular/continuous
# front and a deceptive-but-continuous front respectively.
PROBLEMS = [
    ("ZDT3",   2, "zdt",   {"name": "zdt3"}),
    ("DTLZ2",  3, "dtlz",  {"name": "dtlz2", "n_var": 12, "n_obj": 3}),
    ("WFG6",   3, "wfg",   {"name": "wfg6", "n_var": 10, "n_obj": 3}),
    ("LSMOP9", 3, "lsmop", {"which": 9, "n_obj": 3, "n_var": 60}),
    ("MaF4",   3, "maf",   {"which": 4, "n_obj": 3, "n_var": None}),
]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
THETA_CSV = RESULTS_DIR / "theta_sensitivity.csv"


# -----------------------------------------------------------------------
# Single run helper
# -----------------------------------------------------------------------
def _run_one(args):
    theta, prob_name, n_obj, kind, kwargs, seed = args
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        ref = compute_reference_point(pf)
        init_X = shared_initial_population(problem, POP_SIZE, seed)
        algo = KOAFinalV3(pop_size=POP_SIZE, n_archive=N_ARCHIVE,
                           seed=seed, init_X=init_X, theta=theta)
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
        F = res.F
        if F is None or len(F) == 0:
            raise ValueError("empty front")
        m = compute_all_metrics(F, pf, ref)
        n_eval = int(res.algorithm.evaluator.n_eval)
        return dict(theta=theta, problem=prob_name, seed=seed,
                    status="ok", n_eval=n_eval, fe_budget=FE_BUDGET, **m)
    except Exception as e:
        return dict(theta=theta, problem=prob_name, seed=seed,
                    status=f"error:{type(e).__name__}:{e}",
                    n_eval=0, fe_budget=FE_BUDGET,
                    IGD=np.nan, HV=np.nan, Spacing=np.nan, Spread=np.nan)


# -----------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------
def run_sensitivity(n_seeds=30, n_jobs=1):
    tasks = []
    for theta in THETA_VALUES:
        for pname, n_obj, kind, kwargs in PROBLEMS:
            for seed in range(1, n_seeds + 1):
                tasks.append((theta, pname, n_obj, kind, kwargs, seed))

    rows, done = [], set()
    if THETA_CSV.exists():
        df_existing = pd.read_csv(THETA_CSV)
        rows = df_existing.to_dict("records")
        done = {(r["theta"], r["problem"], r["seed"]) for r in rows}

    remaining = [t for t in tasks if (t[0], t[1], t[5]) not in done]
    total = len(tasks)
    print(f"Theta sensitivity: {total} total, {len(done)} done, "
          f"{len(remaining)} remaining, problems={[p[0] for p in PROBLEMS]}")

    if not remaining:
        print("All runs complete. Generating plots.")
        return pd.DataFrame(rows)

    t0 = time.time()
    if n_jobs == 1:
        for i, task in enumerate(remaining):
            result = _run_one(task)
            rows.append(result)
            if (i + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(THETA_CSV, index=False)
                elapsed = (time.time() - t0) / 60
                eta = elapsed / (i + 1) * (len(remaining) - i - 1)
                print(f"  [{i+1}/{len(remaining)}] "
                      f"theta={task[0]} {task[1]}/seed{task[5]} "
                      f"| elapsed={elapsed:.1f}min eta={eta:.1f}min "
                      f"status={result['status']}")
    else:
        with Pool(n_jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_one, remaining)):
                rows.append(result)
                if (i + 1) % 10 == 0:
                    pd.DataFrame(rows).to_csv(THETA_CSV, index=False)
                    elapsed = (time.time() - t0) / 60
                    print(f"  [{i+1}/{len(remaining)}] "
                          f"elapsed={elapsed:.1f}min status={result['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(THETA_CSV, index=False)
    print(f"\nRuns complete. Wall time: {(time.time()-t0)/60:.1f} min")
    return df


# -----------------------------------------------------------------------
# Figure generation
# -----------------------------------------------------------------------
def plot_sensitivity(df=None):
    if df is None:
        df = pd.read_csv(THETA_CSV)

    ok = df[df.status == "ok"]
    if len(ok) == 0:
        print("No valid results to plot.")
        return

    plt.rcParams.update({
        "font.size": 8, "font.family": "serif",
        "axes.labelsize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    prob_order = [p[0] for p in PROBLEMS]
    marker = ["o", "s", "^", "D", "v"]
    color = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5),
                              gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    for j, pname in enumerate(prob_order):
        sub = ok[ok.problem == pname]
        means = sub.groupby("theta")["IGD"].mean()
        stds = sub.groupby("theta")["IGD"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    label=pname, marker=marker[j % len(marker)],
                    color=color[j % len(color)],
                    linewidth=1.0, markersize=4, capsize=2)

    ax.axvline(0.15, color="gray", linestyle="--", linewidth=0.8,
               label=r"$\theta=0.15$ (default)")
    ax.set_xlabel(r"Connectivity threshold $\theta$")
    ax.set_ylabel("Mean IGD (30 runs)")
    ax.set_title(r"(a) IGD vs $\theta$ across benchmark suites", fontsize=8, pad=4)
    ax.set_xticks(THETA_VALUES)
    ax.legend(frameon=False, ncol=1)

    ax2 = axes[1]
    for j, pname in enumerate(prob_order):
        sub = ok[ok.problem == pname]
        means = sub.groupby("theta")["IGD"].mean()
        ref_val = means[0.15] if 0.15 in means.index else means.median()
        norm = means / (ref_val + 1e-10)
        ax2.plot(norm.index, norm.values, label=pname,
                 marker=marker[j % len(marker)], color=color[j % len(color)],
                 linewidth=1.0, markersize=4)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.axvline(0.15, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(r"Connectivity threshold $\theta$")
    ax2.set_ylabel(r"Normalised IGD (1.0 = $\theta$=0.15)")
    ax2.set_title("(b) Relative sensitivity", fontsize=8, pad=4)
    ax2.set_xticks(THETA_VALUES)
    ax2.legend(frameon=False, ncol=1)

    fig.tight_layout()
    out_pdf = PLOTS_DIR / "theta_sensitivity.pdf"
    out_png = PLOTS_DIR / "theta_sensitivity.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf.name} and {out_png.name}")

    print("\n=== THETA SENSITIVITY SUMMARY ===")
    print(f"{'Problem':8s}  {'theta':6s}  {'Mean IGD':12s}  {'Std IGD':10s}")
    for pname in prob_order:
        sub = ok[ok.problem == pname]
        for theta in THETA_VALUES:
            row = sub[sub.theta == theta]["IGD"]
            tag = " <-- default" if abs(theta - 0.15) < 0.001 else ""
            print(f"{pname:8s}  {theta:.2f}    "
                  f"{row.mean():.4e}      {row.std():.2e}{tag}")
        print()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    pin_device(args.gpu)

    prewarm_pareto_fronts(PROBLEMS, build_problem)

    df = None if args.plot_only else run_sensitivity(n_seeds=args.n_seeds, n_jobs=args.n_jobs)
    plot_sensitivity(df)
    print(f"\nOutputs in {RESULTS_DIR}/ and {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
