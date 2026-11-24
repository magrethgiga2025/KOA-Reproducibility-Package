"""
Ablation of KOA (label KOA-II): the full algorithm against its six one-component variants
(Section 5.6 of the manuscript).

Input : results/raw_runs.csv (labels KOA-II and KOA-II-*)
Output: results/ablation_koa2_ranking.csv  average Friedman ranks of the 7 configurations, Friedman test
        results/ablation_koa2_tests.csv    full KOA vs each variant per problem: Wilcoxon signed-rank
                                           (paired by seed), Holm correction over the 96 comparisons
                                           (16 problems x 6 variants) of each indicator; a significant
                                           difference is a win or a loss according to the mean
        results/ablation_koa2_wtl.csv      W/T/L of the full KOA against each variant

Usage (from code/):  python analyze_ablation_koa2.py
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

FULL = "KOA-II"
VARIANTS = ["KOA-II-Uniform", "KOA-II-PopGuide", "KOA-II-FixedAlloc", "KOA-II-NoAdapt", "KOA-II-DEonly", "KOA-II-SBXonly"]
ALGS = [FULL] + VARIANTS


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    d = pd.read_csv("results/raw_runs.csv")
    d = d[d.algorithm.isin(ALGS)]
    print("status:", d.status.value_counts().to_dict())
    d = d[d.status == "ok"]
    rank_rows, wtl_rows, tests = [], [], []
    for metric, higher in [("HV", True), ("IGD", False)]:
        means = d.groupby(["problem", "algorithm"])[metric].mean().unstack()[ALGS]
        ranks = means.rank(axis=1, ascending=not higher)
        stat, pf = friedmanchisquare(*[means[a] for a in ALGS])
        for a in ALGS:
            rank_rows.append(dict(Metric=metric, Variant=a, Avg_Rank=round(ranks[a].mean(), 3),
                                  Friedman_stat=stat, Friedman_p=pf, N_problems=len(means)))
        rows = []
        for v in VARIANTS:
            for prob in means.index:
                a = d[(d.problem == prob) & (d.algorithm == FULL)].set_index("seed")[metric]
                b = d[(d.problem == prob) & (d.algorithm == v)].set_index("seed")[metric]
                a, b = a.align(b, join="inner")
                p = 1.0 if np.allclose(a, b) else wilcoxon(a, b).pvalue
                rows.append(dict(Metric=metric, Variant=v, Problem=prob, N_pairs=len(a), p=p,
                                 full_mean=a.mean(), variant_mean=b.mean(),
                                 full_better=(a.mean() > b.mean()) if higher else (a.mean() < b.mean())))
        t = pd.DataFrame(rows); t["p_holm"] = holm(t.p)          # 96 comparisons per indicator
        t["Outcome"] = np.where(t.p_holm >= 0.05, "tie", np.where(t.full_better, "win", "loss"))
        tests.append(t)
        for v in VARIANTS:
            o = t[t.Variant == v].Outcome
            wtl_rows.append(dict(Metric=metric, Variant=v, W=(o == "win").sum(), T=(o == "tie").sum(), L=(o == "loss").sum()))
    R = pd.DataFrame(rank_rows); WTL = pd.DataFrame(wtl_rows)
    R.to_csv("results/ablation_koa2_ranking.csv", index=False)
    WTL.to_csv("results/ablation_koa2_wtl.csv", index=False)
    pd.concat(tests).to_csv("results/ablation_koa2_tests.csv", index=False)
    print(R.pivot(index="Variant", columns="Metric", values="Avg_Rank").reindex(ALGS).to_string())
    print(WTL.pivot(index="Variant", columns="Metric", values=["W", "T", "L"]).reindex(VARIANTS).to_string())


if __name__ == "__main__":
    main()
