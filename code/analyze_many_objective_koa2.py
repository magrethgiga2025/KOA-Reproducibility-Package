"""
Statistics for the many-objective experiment (Section 5.4 of the manuscript).

Input : results/many_objective_raw.csv  (DTLZ2, DTLZ7, WFG4, MaF4; M = 3, 5, 8, 10; 9 algorithms; 30 seeds)
Output: results/many_objective_summary.csv   mean/std/median of IGD and HV per (problem, M, algorithm)
        results/many_objective_tests.csv     KOA vs each baseline: Wilcoxon signed-rank (paired by seed),
                                             Holm correction over the 128 comparisons of each indicator,
                                             direction of a significant difference taken from the means
        results/many_objective_friedman.csv  average Friedman ranks over the 16 (problem, M) cases
        results/many_objective_table.tex     body of the IGD table (Table "tab:manyobj")

Usage (from code/):  python analyze_many_objective_koa2.py [results_dir]
Naming: "KOA-II" in the code and data is the algorithm called KOA in the manuscript.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
KOA = "KOA-II"
BASE = ["NSGA-II", "NSGA-III", "MOEA/D", "RVEA", "AGE-MOEA2", "CMOPSO", "SMS-EMOA", "GDE3"]
ALGS = BASE + [KOA]
PROBS, MS = ["DTLZ2", "DTLZ7", "WFG4", "MaF4"], [3, 5, 8, 10]
LOWER = {"IGD": True, "HV": False}


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def a12(x, y):
    """Vargha-Delaney A12: probability that a draw from x is larger than a draw from y."""
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))


def main():
    m = pd.read_csv(os.path.join(RES, "many_objective_raw.csv"))
    m = m[(m.status == "ok") & m.algorithm.isin(ALGS)]
    m.groupby(["problem", "M", "algorithm"])[["IGD", "HV"]].agg(["mean", "std", "median", "count"]) \
     .to_csv(os.path.join(RES, "many_objective_summary.csv"))
    rows, fr = [], []
    for ind, low in LOWER.items():
        t = []
        for p in PROBS:
            for M in MS:
                d = m[(m.problem == p) & (m.M == M)].pivot(index="seed", columns="algorithm", values=ind)
                for b in BASE:
                    x = d[[KOA, b]].dropna()
                    pv = 1.0 if np.allclose(x[KOA], x[b]) else wilcoxon(x[KOA], x[b]).pvalue
                    A = a12(x[KOA], x[b]) if not low else a12(x[b], x[KOA])   # P(KOA better)
                    t.append(dict(Metric=ind, Problem=p, M=M, Baseline=b, N_pairs=len(x), p_value=pv,
                                  KOA_mean=x[KOA].mean(), Baseline_mean=x[b].mean(), A12_KOA_better=A))
        t = pd.DataFrame(t); t["p_holm"] = holm(t.p_value)
        better = np.where(low, t.KOA_mean < t.Baseline_mean, t.KOA_mean > t.Baseline_mean)
        t["Outcome"] = np.where(t.p_holm >= 0.05, "tie", np.where(better, "win", "loss"))
        rows.append(t)
        mean = m.groupby(["problem", "M", "algorithm"])[ind].mean().unstack()[ALGS]
        rk = mean.rank(axis=1, ascending=low)
        st = friedmanchisquare(*[mean[a] for a in ALGS])
        for a in ALGS:
            fr.append(dict(Metric=ind, Algorithm=a, Avg_Rank=rk[a].mean(), Friedman_stat=st.statistic,
                           Friedman_p=st.pvalue, N_cases=len(mean),
                           **{f"Avg_Rank_M{M}": rk.xs(M, level="M")[a].mean() for M in MS}))
        print(ind, "W/T/L of KOA:", t.Outcome.value_counts().reindex(["win", "tie", "loss"]).fillna(0).astype(int).tolist(),
              f"Friedman p = {st.pvalue:.2e}")
    tests = pd.concat(rows); tests.to_csv(os.path.join(RES, "many_objective_tests.csv"), index=False)
    pd.DataFrame(fr).to_csv(os.path.join(RES, "many_objective_friedman.csv"), index=False)

    # LaTeX body of the IGD table
    T = tests[tests.Metric == "IGD"]
    mean = m.groupby(["problem", "M", "algorithm"]).IGD.mean().unstack()[ALGS]
    sym = {"win": "$^{+}$", "tie": "$^{\\approx}$", "loss": "$^{-}$"}
    f = lambda x: f"{x:.4f}" if x < 1 else f"{x:.3f}" if x < 10 else f"{x:.2f}" if x < 100 else f"{x:.1f}" if x < 1000 else f"{x:.0f}"
    L = []
    for p in PROBS:
        for M in MS:
            r = mean.loc[(p, M)]; best = r.idxmin(); cells = []
            for a in ALGS:
                s = f(r[a]); s = f"\\textbf{{{s}}}" if a == best else s
                if a != KOA:
                    s += sym[T[(T.Problem == p) & (T.M == M) & (T.Baseline == a)].Outcome.iloc[0]]
                cells.append(s)
            L.append((f"{p} & {M}" if M == 3 else f" & {M}") + " & " + " & ".join(cells) + " \\\\")
        L.append("\\hline")
    w = T.groupby("Baseline").Outcome.value_counts().unstack(fill_value=0)
    L.append("\\multicolumn{2}{@{}l}{W/T/L} & " + " & ".join(
        f"{w.loc[b].get('win', 0)}/{w.loc[b].get('tie', 0)}/{w.loc[b].get('loss', 0)}" for b in BASE) + " & -- \\\\")
    open(os.path.join(RES, "many_objective_table.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
