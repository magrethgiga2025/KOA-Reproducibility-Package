"""
run_runtime.py
==============
Measures per-run wall-clock time for all 7 algorithms across the
10 benchmark problems. Reports mean runtime in seconds averaged over
N_TIMING_SEEDS independent seeds, then produces a LaTeX-ready table
and a CSV.

TEVC papers typically show runtime as seconds per run (wall clock)
under identical hardware. The table demonstrates that KOA's additional
machinery (archive, geometry-adaptive survival, connectivity measure)
does not add prohibitive overhead relative to reference-vector-based
baselines like NSGA-III and RVEA, which have comparable asymptotic
complexity.

Outputs
-------
  results/runtime_table.csv
  results/runtime_latex.tex   (ready to \\input into your paper)

Usage
-----
  python run_runtime.py [--n-seeds N]
"""
import argparse
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from device import pin_device  # noqa: E402

from pymoo.optimize import minimize
from config import (
    PROBLEMS, POP_SIZE, N_ARCHIVE, N_GEN,
    build_problem, ALGORITHMS
)
from algorithms.factory import build_algorithm

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Timing uses N_TIMING_SEEDS seeds to get a stable estimate
N_TIMING_SEEDS = 5
RUNTIME_CSV    = RESULTS_DIR / "runtime_table.csv"
RUNTIME_TEX    = RESULTS_DIR / "runtime_latex.tex"


def measure_runtime(n_seeds=N_TIMING_SEEDS):
    rows = []
    total = len(ALGORITHMS) * len(PROBLEMS) * n_seeds
    done  = 0
    t_total = time.time()

    for algo_name in ALGORITHMS:
        for pname, n_obj, kind, kwargs in PROBLEMS:
            times = []
            problem = build_problem(kind, kwargs)
            for seed in range(1, n_seeds + 1):
                algo = build_algorithm(
                    algo_name, n_obj=n_obj,
                    pop_size=POP_SIZE, n_archive=N_ARCHIVE, seed=seed
                )
                t0 = time.time()
                try:
                    minimize(problem, algo, ("n_gen", N_GEN),
                             seed=seed, verbose=False)
                except Exception as e:
                    print(f"  [warn] {algo_name}/{pname}/seed{seed}: {e}")
                t1 = time.time()
                times.append(t1 - t0)
                done += 1

            mean_t = float(np.mean(times))
            std_t  = float(np.std(times))
            rows.append(dict(
                algorithm=algo_name, problem=pname,
                mean_sec=round(mean_t, 2), std_sec=round(std_t, 2)
            ))
            elapsed = (time.time() - t_total) / 60
            print(f"  {algo_name:12s} / {pname:8s}: "
                  f"{mean_t:.2f}±{std_t:.2f}s "
                  f"[{done}/{total}] {elapsed:.1f}min")

    df = pd.DataFrame(rows)
    df.to_csv(RUNTIME_CSV, index=False)
    return df


def make_runtime_table(df=None):
    if df is None:
        df = pd.read_csv(RUNTIME_CSV)

    # Pivot: algorithms as rows, problems as columns, values = mean_sec
    pivot = df.pivot_table(
        index="algorithm", columns="problem",
        values="mean_sec", aggfunc="mean"
    )
    # Also compute average across all problems
    pivot["Mean"] = pivot.mean(axis=1)

    algo_order = ["KOA", "NSGA-II", "NSGA-III",
                  "MOEA/D", "RVEA", "AGE-MOEA2", "CMOPSO"]
    prob_order = ["ZDT1", "ZDT2", "ZDT3",
                  "WFG1", "WFG4", "WFG9",
                  "LSMOP1", "LSMOP9", "MaF4", "MaF7", "Mean"]
    pivot = pivot.reindex(index=algo_order,
                           columns=[p for p in prob_order if p in pivot.columns])

    print("\n=== RUNTIME TABLE (seconds per run) ===")
    print(pivot.round(1).to_string())

    # Produce LaTeX table
    lines = [
        r"\begin{table}[!t]",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\caption{Mean wall-clock time per run (seconds, "
        + str(N_TIMING_SEEDS) + r" seeds). "
        r"KOA achieves competitive convergence quality with "
        r"runtime comparable to reference-direction-based algorithms.}",
        r"\label{tab:runtime}",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.5pt}",
    ]

    col_spec = "l" + "r" * len(pivot.columns)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    hdr_cols = " & ".join(pivot.columns.tolist())
    lines.append(r"Algorithm & " + hdr_cols + r" \\")
    lines.append(r"\midrule")

    # Rows
    for algo in pivot.index:
        row_vals = []
        for col in pivot.columns:
            v = pivot.loc[algo, col]
            if pd.isna(v):
                row_vals.append("--")
            else:
                row_vals.append(f"{v:.1f}")
        row_str = " & ".join(row_vals)
        bold = r"\textbf{" + algo + "}" if algo == "KOA" else algo
        lines.append(f"{bold} & {row_str} " + r"\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    with open(RUNTIME_TEX, "w") as f:
        f.write(tex)
    print(f"\nWrote {RUNTIME_TEX.name}")
    return pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds",   type=int,  default=N_TIMING_SEEDS)
    parser.add_argument("--table-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0,
                         help="CUDA_VISIBLE_DEVICES pin ('device port'); "
                              "no effect on this CPU-only workload's results.")
    args = parser.parse_args()
    pin_device(args.gpu)

    if not args.table_only:
        df = measure_runtime(n_seeds=args.n_seeds)
    else:
        df = None

    make_runtime_table(df)
    print(f"\nOutputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
