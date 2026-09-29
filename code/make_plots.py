"""
IEEE-style figure generation.

Per instruction: no plot titles (figures are referenced by caption in the
LaTeX/Word source, not by an in-image title), clean axis labels, vector
output (PDF) suitable for direct inclusion in an IEEE TEVC submission,
plus PNG previews for quick viewing.

Generates:
  plots/convergence_<problem>.pdf     -- IGD vs generation, one line per algorithm
  plots/pareto_front_<problem>.pdf    -- final population in objective space
                                          (2D scatter for M=2, 3D for M=3)
  plots/stability_cv.pdf              -- coefficient of variation per algorithm/problem
  plots/friedman_ranks.pdf            -- average Friedman rank bar chart per metric

Usage:
    python make_plots.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROBLEM_NAMES, ALGORITHMS, build_problem, PROBLEMS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RAW_CSV = RESULTS_DIR / "raw_runs.csv"
CONV_CSV = RESULTS_DIR / "convergence_trajectories.csv"
FRIEDMAN_CSV = RESULTS_DIR / "friedman_ranking.csv"

# IEEE single-column width is ~3.5in; double-column ~7.16in. Use double-column
# width for multi-panel figures, single-column for individual plots.
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.2)

# Consistent, colorblind-friendlier palette + distinct markers/linestyles so
# figures remain legible in greyscale print (IEEE print requirement).
STYLE = {
    "KOA":       dict(color="#d62728", marker="o", linestyle="-",  linewidth=1.4),
    "NSGA-II":   dict(color="#1f77b4", marker="s", linestyle="--", linewidth=1.0),
    "NSGA-III":  dict(color="#2ca02c", marker="^", linestyle="-.", linewidth=1.0),
    "MOEA/D":    dict(color="#ff7f0e", marker="D", linestyle=":",  linewidth=1.0),
    "RVEA":      dict(color="#9467bd", marker="v", linestyle="--", linewidth=1.0),
    "AGE-MOEA2": dict(color="#8c564b", marker="P", linestyle="-.", linewidth=1.0),
    "CMOPSO":    dict(color="#17becf", marker="X", linestyle=":",  linewidth=1.0),
}

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.labelsize": 9,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.markersize": 3.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_convergence(problem_name):
    if not CONV_CSV.exists():
        print(f"  [skip] {CONV_CSV} not found -- run track_convergence.py first")
        return
    df = pd.read_csv(CONV_CSV)
    sub = df[df["problem"] == problem_name]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for algo in ALGORITHMS:
        a = sub[sub["algorithm"] == algo]
        if a.empty:
            continue
        traj = a.groupby("generation")["IGD"].mean()
        style = STYLE.get(algo, {})
        ax.plot(traj.index, traj.values, label=algo,
                markevery=max(1, len(traj) // 10), **style)

    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("IGD (log scale)")
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    out_pdf = PLOTS_DIR / f"convergence_{problem_name}.pdf"
    out_png = PLOTS_DIR / f"convergence_{problem_name}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_pdf.name}")


def _get_one_run_front(problem_name, algo_name, seed_pref=1):
    """Re-run once (small, fast) to get an actual final population for the
    Pareto-front scatter plot. Final metrics CSV stores summary stats only,
    not the raw objective vectors, so this regenerates one representative
    front per (problem, algorithm) using the same protocol settings."""
    from pymoo.optimize import minimize
    from algorithms.factory import build_algorithm
    from config import POP_SIZE, N_ARCHIVE, N_GEN

    spec = next(p for p in PROBLEMS if p[0] == problem_name)
    _, n_obj, kind, kwargs = spec
    problem = build_problem(kind, kwargs)
    algo = build_algorithm(algo_name, n_obj=n_obj, pop_size=POP_SIZE,
                            n_archive=N_ARCHIVE, seed=seed_pref)
    res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed_pref, verbose=False)
    return res.F, problem.pareto_front()


def plot_pareto_front(problem_name, regenerate=True):
    if not regenerate:
        print(f"  [skip] {problem_name}: regenerate=False and no cached fronts available")
        return

    spec = next(p for p in PROBLEMS if p[0] == problem_name)
    _, n_obj, kind, kwargs = spec

    fronts = {}
    pf = None
    for algo in ALGORITHMS:
        try:
            F, pf = _get_one_run_front(problem_name, algo)
            fronts[algo] = F
        except Exception as e:
            print(f"  [warn] {problem_name}/{algo} front generation failed: {e}")

    if n_obj == 2:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        if pf is not None:
            order = np.argsort(pf[:, 0])
            ax.plot(pf[order, 0], pf[order, 1], color="black", linewidth=1.0,
                    linestyle="-", label="True PF", zorder=1)
        for algo, F in fronts.items():
            style = STYLE.get(algo, {})
            ax.scatter(F[:, 0], F[:, 1], s=8, label=algo,
                       color=style.get("color"), marker=style.get("marker", "o"),
                       alpha=0.75, zorder=2)
        ax.set_xlabel("$f_1$")
        ax.set_ylabel("$f_2$")
        ax.legend(loc="best", frameon=False, ncol=2, fontsize=5.5)
        fig.tight_layout()

    else:  # n_obj == 3
        fig = plt.figure(figsize=FIG_SINGLE)
        ax = fig.add_subplot(111, projection="3d")
        if pf is not None:
            ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], s=2, color="black",
                       alpha=0.25, label="True PF")
        for algo, F in fronts.items():
            style = STYLE.get(algo, {})
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=6, label=algo,
                       color=style.get("color"), marker=style.get("marker", "o"),
                       alpha=0.8)
        ax.set_xlabel("$f_1$", labelpad=-2)
        ax.set_ylabel("$f_2$", labelpad=-2)
        ax.set_zlabel("$f_3$", labelpad=-2)
        ax.tick_params(labelsize=5)
        ax.legend(loc="center left", frameon=False, fontsize=5, bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()

    out_pdf = PLOTS_DIR / f"pareto_front_{problem_name}.pdf"
    out_png = PLOTS_DIR / f"pareto_front_{problem_name}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_pdf.name}")


def plot_stability_cv():
    if not RAW_CSV.exists():
        print(f"  [skip] {RAW_CSV} not found")
        return
    df = pd.read_csv(RAW_CSV)
    ok = df[df["status"] == "ok"]

    cv = (ok.groupby(["problem", "algorithm"])["IGD"]
          .agg(lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan)
          .unstack("algorithm").reindex(columns=ALGORITHMS))

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    x = np.arange(len(cv.index))
    width = 0.8 / len(ALGORITHMS)
    for i, algo in enumerate(ALGORITHMS):
        style = STYLE.get(algo, {})
        ax.bar(x + i * width, cv[algo].values, width=width, label=algo,
               color=style.get("color"))
    ax.set_xticks(x + width * (len(ALGORITHMS) - 1) / 2)
    ax.set_xticklabels(cv.index, rotation=30, ha="right")
    ax.set_ylabel("Coefficient of Variation (IGD)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6)
    fig.tight_layout()
    out_pdf = PLOTS_DIR / "stability_cv.pdf"
    fig.savefig(out_pdf)
    fig.savefig(PLOTS_DIR / "stability_cv.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {out_pdf.name}")


def plot_friedman_ranks():
    if not FRIEDMAN_CSV.exists():
        print(f"  [skip] {FRIEDMAN_CSV} not found -- run analyze_results.py first")
        return
    df = pd.read_csv(FRIEDMAN_CSV)
    metrics = df["Metric"].unique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(FIG_DOUBLE[0], 2.6), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = df[df["Metric"] == metric].set_index("Algorithm").reindex(ALGORITHMS)
        colors = [STYLE.get(a, {}).get("color", "gray") for a in ALGORITHMS]
        ax.bar(ALGORITHMS, sub["Avg_Rank"].values, color=colors)
        ax.set_ylabel(f"Avg. Rank ({metric})")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_pdf = PLOTS_DIR / "friedman_ranks.pdf"
    fig.savefig(out_pdf)
    fig.savefig(PLOTS_DIR / "friedman_ranks.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {out_pdf.name}")


def main():
    print("Generating convergence plots...")
    for p in PROBLEM_NAMES:
        plot_convergence(p)

    print("Generating Pareto front plots (re-running one seed per algorithm)...")
    for p in PROBLEM_NAMES:
        plot_pareto_front(p)

    print("Generating stability (CV) plot...")
    plot_stability_cv()

    print("Generating Friedman ranking plot...")
    plot_friedman_ranks()

    print(f"\nAll figures written to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
