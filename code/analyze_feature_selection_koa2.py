"""
Statistics and LaTeX tables for the feature-selection experiment.

Per run it computes
    HV_train : hypervolume of the training front, reference point (1.1, 1.1)
    IGD      : IGD of the training front w.r.t. the reference front of that dataset
               (non-dominated union of all final fronts of all algorithms and runs)
    HV_test  : hypervolume of the same solutions re-evaluated on the test split
Then, KOA vs every baseline, per dataset: Wilcoxon signed-rank (paired by seed),
Holm-corrected over all (dataset x baseline) comparisons of that metric, alpha = 0.05;
average Friedman ranks and the Friedman test over datasets.

Usage (from code/):  python analyze_feature_selection.py
Outputs: results/feature_selection_tables/
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

from fs_problem import nondominated, dataset_summary
from config import ALGORITHMS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "feature_selection")
TABLES = os.path.join(HERE, "results", "feature_selection_tables")
REF_POINT = np.array([1.1, 1.1])
METRICS = {"HV_train": "max", "IGD": "min", "HV_test": "max"}
ORDER = [a for a in ALGORITHMS if a not in ("KOA", "KOA-II") and not a.startswith("KOA-II-")] + ["KOA-II"]     # KOA in the last column


# ------------------------------------------------------------- indicators ----
def hv2d(F, ref=REF_POINT):
    F = np.asarray(F)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[nondominated(F)]
    F = F[np.argsort(F[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in F:
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return hv


def igd(F, R):
    d = np.sqrt(((R[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1).mean()


# ---------------------------------------------------------------- loading ----
def _load_runs_all():
    recs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "*", "run_*.npz"))):
        ds, alg = path.split(os.sep)[-3:-1]
        seed = int(os.path.basename(path)[4:-4])
        z = np.load(path)
        recs.append(dict(dataset=ds, algorithm=alg.replace("_", "/"), seed=seed,
                         F_train=z["F_train"], F_test=z["F_test"],
                         n_evals=int(z["n_evals"]), runtime=float(z["runtime"])))
    if not recs:
        raise SystemExit(f"No results found in {RESULTS}")
    return recs


def compute_metrics(recs):
    rows = []
    for ds in sorted({r["dataset"] for r in recs}):
        sub = [r for r in recs if r["dataset"] == ds]
        U = np.vstack([r["F_train"] for r in sub])
        R = np.unique(U[nondominated(U)], axis=0)          # reference front
        for r in sub:
            rows.append(dict(dataset=ds, algorithm=r["algorithm"], seed=r["seed"],
                             HV_train=hv2d(r["F_train"]), IGD=igd(r["F_train"], R),
                             HV_test=hv2d(r["F_test"]),
                             min_test_error=r["F_test"][:, 1].min(),
                             n_evals=r["n_evals"], runtime=r["runtime"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- stats ----
def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(df, metric, ref_alg, alpha):
    """Return DataFrame of KOA-vs-baseline symbols (+ / - / ≈) per dataset."""
    better = METRICS[metric]
    base = [a for a in algorithms(df) if a != ref_alg]
    tests = []
    for ds in sorted(df.dataset.unique()):
        k = df[(df.dataset == ds) & (df.algorithm == ref_alg)].set_index("seed")[metric]
        for a in base:
            b = df[(df.dataset == ds) & (df.algorithm == a)].set_index("seed")[metric]
            common = k.index.intersection(b.index)          # pair by seed
            x, y = k.loc[common].to_numpy(), b.loc[common].to_numpy()
            if len(common) < 2 or np.allclose(x, y):
                p = 1.0
            else:
                p = wilcoxon(x, y, zero_method="wilcox").pvalue
            koa_better = (np.mean(x) > np.mean(y)) if better == "max" else (np.mean(x) < np.mean(y))
            tests.append(dict(dataset=ds, baseline=a, p=p, koa_better=koa_better,
                              n_pairs=len(common)))
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t.p)
    t["symbol"] = np.where(t.p_holm >= alpha, "approx",
                           np.where(t.koa_better, "+", "-"))
    return t


def friedman(df, metric):
    better = METRICS[metric]
    means = df.pivot_table(index="dataset", columns="algorithm", values=metric, aggfunc="mean")
    means = means[[a for a in algorithms(df)]]
    vals = means.to_numpy()
    ranks = np.array([rankdata(-r if better == "max" else r) for r in vals])
    avg = pd.Series(ranks.mean(axis=0), index=means.columns)
    stat = friedmanchisquare(*vals.T) if vals.shape[0] >= 2 and vals.shape[1] >= 3 else None
    return avg, stat


def algorithms(df):
    present = list(df.algorithm.unique())
    return [a for a in ORDER if a in present] + sorted(a for a in present if a not in ORDER)


# ------------------------------------------------------------------- LaTeX ----
SYM = {"+": "^{+}", "-": "^{-}", "approx": "^{\\approx}"}


def latex_metric_table(df, metric, tests, ref_alg, caption, label):
    algs = algorithms(df)
    better = METRICS[metric]
    g = df.groupby(["dataset", "algorithm"])[metric]
    mean, std = g.mean(), g.std()
    lines = [
        "\\begin{table*}[!htbp]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.25}",
        "\\begin{tabular}{@{}l" + "c" * len(algs) + "@{}}", "\\hline",
        "\\textbf{Dataset} & " + " & ".join(f"\\textbf{{{a}}}" for a in algs) + " \\\\", "\\hline",
    ]
    for ds in sorted(df.dataset.unique()):
        m = {a: mean.get((ds, a), np.nan) for a in algs}
        best = (max if better == "max" else min)(v for v in m.values() if not np.isnan(v))
        cells = []
        for a in algs:
            if np.isnan(m[a]):
                cells.append("--"); continue
            num = f"{m[a]:.4f}"
            num = f"\\mathbf{{{num}}}" if np.isclose(m[a], best) else num
            sym = ""
            if a != ref_alg:
                s = tests[(tests.dataset == ds) & (tests.baseline == a)].symbol
                sym = SYM[s.iloc[0]] if len(s) else ""
            cells.append(f"${num}_{{\\pm{std.get((ds, a), 0):.4f}}}{sym}$")
        lines.append(f"{ds} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def latex_wtl_table(all_tests, ref_alg, ranks, stats):
    metrics = list(all_tests)
    base = [a for a in ORDER if a in ranks[metrics[0]].index and a != ref_alg] + \
           [a for a in ranks[metrics[0]].index if a not in ORDER and a != ref_alg]
    hdr = " & ".join(f"\\textbf{{{m.replace('_', ' ')}}}" for m in metrics)
    lines = [
        "\\begin{table}[!htbp]",
        f"\\caption{{Feature selection: W/T/L counts of {ref_alg} against each baseline "
        "(Holm-corrected Wilcoxon signed-rank test, $\\alpha=0.05$) and average Friedman "
        "ranks (lower is better).}",
        "\\label{tab:fs_wtl}", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{@{}l" + "c" * (2 * len(metrics)) + "@{}}", "\\hline",
        f" & \\multicolumn{{{len(metrics)}}}{{c}}{{\\textbf{{W/T/L}}}} & "
        f"\\multicolumn{{{len(metrics)}}}{{c}}{{\\textbf{{Friedman rank}}}} \\\\",
        f"\\textbf{{Algorithm}} & {hdr} & {hdr} \\\\", "\\hline",
    ]
    for a in base + [ref_alg]:
        wtl = []
        for m in metrics:
            if a == ref_alg:
                wtl.append("--"); continue
            s = all_tests[m][all_tests[m].baseline == a].symbol
            wtl.append(f"{(s == '+').sum()}/{(s == 'approx').sum()}/{(s == '-').sum()}")
        rk = [f"{ranks[m][a]:.2f}" for m in metrics]
        lines.append(f"{a} & " + " & ".join(wtl + rk) + " \\\\")
    lines.append("\\hline")
    chi = " & ".join(f"{stats[m].statistic:.2f}" if stats[m] else "--" for m in metrics)
    pv = " & ".join(f"{stats[m].pvalue:.1e}" if stats[m] else "--" for m in metrics)
    pad = " & " * len(metrics)
    lines += [f"Friedman $\\chi^2$ {pad}& {chi} \\\\", f"Friedman $p$ {pad}& {pv} \\\\",
              "\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_dataset_table(names):
    lines = ["\\begin{table}[!htbp]",
             "\\caption{Datasets used in the feature-selection experiment.}",
             "\\label{tab:fs_datasets}", "\\centering", "\\small",
             "\\begin{tabular}{@{}lccc@{}}", "\\hline",
             "\\textbf{Dataset} & \\textbf{Features} & \\textbf{Instances} & \\textbf{Classes} \\\\",
             "\\hline"]
    for n, d, s, c in dataset_summary(names):
        lines.append(f"{n} & {d} & {s} & {c} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="KOA-II")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(TABLES, exist_ok=True)

    df = compute_metrics(load_runs())
    df.to_csv(os.path.join(TABLES, "per_run_metrics.csv"), index=False)
    if a.reference not in set(df.algorithm):
        raise SystemExit(f"Reference algorithm {a.reference} has no results yet.")

    # warn about unequal run counts / budgets
    counts = df.groupby(["dataset", "algorithm"]).seed.count().unstack()
    if counts.isna().any().any() or counts.stack().nunique() > 1:
        print("WARNING: unequal numbers of runs:\n", counts, "\n")
    ev = df.groupby("algorithm").n_evals.agg(["min", "max"])
    print("Evaluations used per algorithm:\n", ev, "\n")

    tests, ranks, stats, tex = {}, {}, {}, []
    captions = {
        "HV_train": "Mean HV (higher is better) with std as subscript of the training "
                    "fronts (selected-feature ratio vs.\\ inner-CV error), reference point $(1.1,1.1)$.",
        "IGD": "Mean IGD (lower is better) with std as subscript, w.r.t.\\ the non-dominated "
               "union of all final training fronts of each dataset.",
        "HV_test": "Mean test HV (higher is better) with std as subscript: final solutions "
                   "re-evaluated on the held-out test split.",
    }
    n_tests = None
    for m in METRICS:
        tests[m] = compare(df, m, a.reference, a.alpha)
        ranks[m], stats[m] = friedman(df, m)
        n_tests = len(tests[m])
        tests[m].to_csv(os.path.join(TABLES, f"tests_{m}.csv"), index=False)
        cap = (captions[m] + f" $^+$: {a.reference} significantly better; $^-$: significantly "
               f"worse; $^\\approx$: equivalent (Holm-corrected Wilcoxon signed-rank test over "
               f"the ${n_tests}$ comparisons, $\\alpha={a.alpha}$; {counts.max().max()} paired runs).")
        tex.append(latex_metric_table(df, m, tests[m], a.reference, cap, f"tab:fs_{m.lower()}"))
    tex.append(latex_wtl_table(tests, a.reference, ranks, stats))
    tex.append(latex_dataset_table(sorted(df.dataset.unique())))

    out = os.path.join(TABLES, "fs_tables.tex")
    with open(out, "w") as f:
        f.write("% Generated by analyze_feature_selection.py - requires no extra packages\n\n" + "\n".join(tex))

    for m in METRICS:
        s = tests[m].symbol
        print(f"{m:9s} W/T/L for {a.reference}: {(s == '+').sum()}/{(s == 'approx').sum()}/"
              f"{(s == '-').sum()}   Friedman ranks: "
              + ", ".join(f"{k}={v:.2f}" for k, v in ranks[m].items()))
    print(f"\nTables written to {out}\nPer-run metrics: {os.path.join(TABLES, 'per_run_metrics.csv')}")



def load_runs(*args, **kwargs):
    """KOA-II analysis: same as the original loader, but KOA v3 rows removed."""
    runs = _load_runs_all(*args, **kwargs)
    if hasattr(runs, "columns") and "algorithm" in runs.columns:
        return runs[runs["algorithm"] != "KOA"].reset_index(drop=True)
    if isinstance(runs, list):
        return [r for r in runs if (r.get("algorithm") if isinstance(r, dict) else None) != "KOA"]
    return runs


if __name__ == "__main__":
    main()
