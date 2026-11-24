"""
make_paper_figures.py -- figures for the KOA manuscript (JKSU-CIS submission).

Usage (from the code/ folder on the server):
    python make_paper_figures.py results figures

Conventions
  * The algorithm called "KOA-II" in the experiment files is labelled "KOA"
    in every figure. The old KOA v3 rows ("KOA" in the files) are excluded.
  * No figure has a title: everything is described in the LaTeX caption.
    Sub-panels are identified by "(a)", "(b)", ... in their x-axis label.
  * Each figure is saved as PDF (vector, for LaTeX) and PNG (300 dpi, preview).
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
KOA_C = "#2a78d6"      # KOA (blue)
BASE_C = "#b4b2a9"     # baselines (neutral gray)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
WIN_C, TIE_C, LOSS_C = "#2a78d6", "#d6d5cf", "#eb6834"
FS_COLORS = {"KOA": "#2a78d6", "NSGA-II": "#eb6834", "GDE3": "#1baf7a", "SMS-EMOA": "#52514e"}
FS_MARKERS = {"KOA": "o", "NSGA-II": "s", "GDE3": "^", "SMS-EMOA": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
FULL_W = 6.85   # inches (~174 mm, Springer full text width)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


def rename(s):
    """KOA-II -> KOA, KOA-II-X -> KOA-X."""
    return s.replace("KOA-II", "KOA").replace("AGE-MOEA2", "AGE-MOEA-II")


# ---------------------------------------------------------------- data
raw = pd.read_csv(os.path.join(RES, "raw_runs.csv"))
raw = raw[(raw.status == "ok") & (raw.algorithm != "KOA")].copy()   # drop KOA v3
raw["algorithm"] = raw.algorithm.map(rename)

BASELINES = ["NSGA-II", "NSGA-III", "MOEA/D", "RVEA", "AGE-MOEA-II", "CMOPSO", "SMS-EMOA", "GDE3"]
MAIN = BASELINES + ["KOA"]
main = raw[raw.algorithm.isin(MAIN)]
PROBLEMS = ["ZDT1", "ZDT2", "ZDT3", "DTLZ1", "DTLZ2", "DTLZ4", "DTLZ6", "DTLZ7",
            "WFG1", "WFG4", "WFG6", "WFG9", "LSMOP1", "LSMOP9", "MaF4", "MaF7"]


def avg_ranks(df, algs, metric, higher):
    m = df.groupby(["problem", "algorithm"])[metric].mean().unstack()[algs]
    return m.rank(axis=1, ascending=not higher).mean()


def nemenyi_cd(k, n, alpha_q=None):
    # q_0.05 for the two-tailed Nemenyi test (Demsar 2006), k = 2..10
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
         8: 3.031, 9: 3.102, 10: 3.164}[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def rank_panel(ax, ranks, label, cd=None, highlight="KOA"):
    r = ranks.sort_values(ascending=False)       # best at the top
    y = np.arange(len(r))
    colors = [KOA_C if a == highlight else BASE_C for a in r.index]
    ax.barh(y, r.values, height=0.62, color=colors, edgecolor="white", linewidth=0.8)
    for yi, v in zip(y, r.values):
        ax.text(v + 0.08, yi, f"{v:.2f}", va="center", ha="left", fontsize=7.5, color=INK2,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(r.index)
    for t in ax.get_yticklabels():
        if t.get_text() == highlight:
            t.set_fontweight("bold"); t.set_color(INK)
    ax.set_xlim(0, max(r.values) * 1.15 + 0.3)
    ax.set_xlabel(label)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    if cd is not None:
        x0 = r[highlight]
        ax.axvline(x0 + cd, color=INK2, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        ax.text(x0 + cd, len(r) - 0.35, f"CD = {cd:.2f}", fontsize=7, color=INK2,
                ha="center", va="bottom",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_ylim(-0.6, len(r) + 0.15)


# ============================================================ Fig 1
# Average Friedman ranks on the 16 benchmark problems (9 algorithms)
n_prob = main.problem.nunique()
cd = nemenyi_cd(len(MAIN), n_prob)
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
rank_panel(axes[0], avg_ranks(main, MAIN, "HV", True), "(a) Average Friedman rank, HV", cd)
rank_panel(axes[1], avg_ranks(main, MAIN, "IGD", False), "(b) Average Friedman rank, IGD", cd)
fig.tight_layout(w_pad=2.5)
save(fig, "fig_friedman_ranks")

# ============================================================ Fig 2
# W/T/L of KOA against each baseline (Holm-corrected Wilcoxon, 16 problems).
# Counts are the official output of analyze_results_koa2.py (18 Nov 2026).
WTL = {
    "HV": {"NSGA-II": (13, 2, 1), "NSGA-III": (13, 3, 0), "MOEA/D": (13, 2, 1),
           "RVEA": (13, 3, 0), "AGE-MOEA-II": (12, 4, 0), "CMOPSO": (16, 0, 0),
           "SMS-EMOA": (4, 3, 9), "GDE3": (15, 0, 1)},
    "IGD": {"NSGA-II": (12, 3, 1), "NSGA-III": (12, 4, 0), "MOEA/D": (12, 3, 1),
            "RVEA": (11, 5, 0), "AGE-MOEA-II": (9, 5, 2), "CMOPSO": (16, 0, 0),
            "SMS-EMOA": (11, 3, 2), "GDE3": (13, 1, 2)},
}
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7), sharey=True)
order = sorted(BASELINES, key=lambda b: -WTL["IGD"][b][0])[::-1]
for ax, metric, lab in zip(axes, ["HV", "IGD"], ["(a) Number of problems, HV", "(b) Number of problems, IGD"]):
    y = np.arange(len(order))
    w = np.array([WTL[metric][b][0] for b in order])
    t = np.array([WTL[metric][b][1] for b in order])
    l = np.array([WTL[metric][b][2] for b in order])
    ax.barh(y, w, color=WIN_C, height=0.62, edgecolor="white", linewidth=1.2)
    ax.barh(y, t, left=w, color=TIE_C, height=0.62, edgecolor="white", linewidth=1.2)
    ax.barh(y, l, left=w + t, color=LOSS_C, height=0.62, edgecolor="white", linewidth=1.2)
    for yi, (wi, ti, li) in enumerate(zip(w, t, l)):
        if wi >= 2: ax.text(wi / 2, yi, str(wi), ha="center", va="center", color="white", fontsize=7.5)
        if ti >= 2: ax.text(wi + ti / 2, yi, str(ti), ha="center", va="center", color=INK, fontsize=7.5)
        if li >= 2: ax.text(wi + ti + li / 2, yi, str(li), ha="center", va="center", color="white", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlim(0, 16); ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel(lab)
handles = [Patch(color=WIN_C, label="KOA significantly better"),
           Patch(color=TIE_C, label="No significant difference"),
           Patch(color=LOSS_C, label="KOA significantly worse")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(w_pad=2.0)
save(fig, "fig_wtl")

# ============================================================ Fig 3
# IGD distributions over 30 runs on six representative problems
REP = ["DTLZ2", "DTLZ7", "WFG6", "WFG9", "MaF4", "LSMOP9"]
fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.6))
for k, (ax, prob) in enumerate(zip(axes.flat, REP)):
    sub = main[main.problem == prob]
    med = sub.groupby("algorithm")["IGD"].median().reindex(MAIN)
    algs = list(med.sort_values().index)            # best (lowest IGD) on the left
    data = [sub[sub.algorithm == a]["IGD"].values for a in algs]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                    medianprops=dict(color=INK, linewidth=1.0),
                    whiskerprops=dict(color=INK2, linewidth=0.7),
                    capprops=dict(color=INK2, linewidth=0.7),
                    flierprops=dict(marker="o", markersize=2.2, markerfacecolor=INK2,
                                    markeredgewidth=0, alpha=0.6))
    for patch, med_line, a in zip(bp["boxes"], bp["medians"], algs):
        if a == "KOA":
            patch.set_facecolor("#cfe0f6"); patch.set_edgecolor(KOA_C); patch.set_linewidth(1.3)
            med_line.set_color(KOA_C); med_line.set_linewidth(1.8)
        else:
            patch.set_facecolor(BASE_C); patch.set_edgecolor(INK2); patch.set_linewidth(0.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(1, len(algs) + 1))
    ax.set_xticklabels(algs, rotation=55, ha="right", fontsize=7)
    for t in ax.get_xticklabels():
        if t.get_text() == "KOA":
            t.set_fontweight("bold"); t.set_color(INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, which="major"); ax.set_axisbelow(True)
    ax.set_xlabel(f"({'abcdef'[k]}) {prob}", labelpad=3)
    if k % 3 == 0:
        ax.set_ylabel("IGD (log scale)")
fig.tight_layout(h_pad=1.2, w_pad=1.2)
save(fig, "fig_igd_boxplots")

# ============================================================ Fig 4
# Feature selection: average Friedman ranks over 13 datasets (4 algorithms)
fs = pd.read_csv(os.path.join(RES, "feature_selection_tables", "per_run_metrics.csv"))
fs = fs[fs.algorithm.isin(["KOA-II", "NSGA-II", "GDE3", "SMS-EMOA"])].copy()
fs["algorithm"] = fs.algorithm.map(rename)
FS_ALGS = ["KOA", "NSGA-II", "GDE3", "SMS-EMOA"]
n_ds = fs.dataset.nunique()
cd_fs = nemenyi_cd(4, n_ds)
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.9), sharey=False)
for ax, (metric, higher, lab) in zip(axes, [
        ("HV_train", True, "(a) Average rank, HV (training)"),
        ("HV_test", True, "(b) Average rank, HV (test)"),
        ("IGD", False, "(c) Average rank, IGD")]):
    m = fs.groupby(["dataset", "algorithm"])[metric].mean().unstack()[FS_ALGS]
    rank_panel(ax, m.rank(axis=1, ascending=not higher).mean(), lab, cd_fs)
    ax.set_xlim(0, 4.6)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_fs_ranks")

# ============================================================ Fig 5
# Feature selection: test-set fronts of the median-HV_test run, two datasets
FS_DIR = os.path.join(RES, "feature_selection")
DS = ["Digits", "Madelon"]
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7))
for k, (ax, ds) in enumerate(zip(axes, DS)):
    for a in FS_ALGS:
        file_alg = "KOA-II" if a == "KOA" else a
        runs = fs[(fs.dataset == ds) & (fs.algorithm == a)].sort_values("HV_test")
        seed = int(runs.iloc[len(runs) // 2]["seed"])      # median run by HV_test
        z = np.load(os.path.join(FS_DIR, ds, file_alg, f"run_{seed:02d}.npz"))
        F = z["F_test"]
        F = F[np.argsort(F[:, 0])]
        ax.scatter(F[:, 0], F[:, 1], s=22 if a == "KOA" else 16, marker=FS_MARKERS[a],
                   color=FS_COLORS[a], edgecolor="white", linewidth=0.6,
                   zorder=4 if a == "KOA" else 3, label=a)
    ax.set_xlabel(f"({'ab'[k]}) Ratio of selected features, {ds}")
    if k == 0:
        ax.set_ylabel("Test classification error")
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
handles = [Line2D([0], [0], color=FS_COLORS[a], marker=FS_MARKERS[a], markersize=5,
                  markeredgecolor="white", linewidth=0, label=a) for a in FS_ALGS]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(w_pad=2.0)
save(fig, "fig_fs_fronts")

# ============================================================ Fig 6
# Ablation: average Friedman ranks of KOA and its six variants
ABL = ["KOA", "KOA-Uniform", "KOA-PopGuide", "KOA-FixedAlloc", "KOA-NoAdapt",
       "KOA-DEonly", "KOA-SBXonly"]
abl = raw[raw.algorithm.isin(ABL)]
cd_abl = nemenyi_cd(len(ABL), abl.problem.nunique())
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5))
rank_panel(axes[0], avg_ranks(abl, ABL, "HV", True), "(a) Average Friedman rank, HV", cd_abl)
rank_panel(axes[1], avg_ranks(abl, ABL, "IGD", False), "(b) Average Friedman rank, IGD", cd_abl)
fig.tight_layout(w_pad=2.5)
save(fig, "fig_ablation_ranks")

print("all figures written to", OUT)
