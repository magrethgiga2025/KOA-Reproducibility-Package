"""
Statistics for the one-at-a-time parameter sensitivity of KOA (Section 5.7 of the manuscript).

Input : results/sensitivity_koa2.csv  (13 configurations x 6 problems x 30 seeds)
        parameters (code name -> manuscript symbol): H -> H, archive_ratio -> N_A/N,
        beta -> beta, q_lr -> lambda;  defaults H=5, archive_ratio=1.0, beta=0.1, q_lr=0.3
Output: results/sensitivity_summary.csv   mean and median HV/IGD per (param, value, problem),
                                          with the change of the median IGD relative to the default (%)
        results/sensitivity_tests.csv     each alternative vs the default: Wilcoxon signed-rank (paired by seed),
                                          Holm correction over the 72 comparisons of each indicator
        results/sensitivity_friedman.csv  average Friedman ranks of the 13 configurations

Usage (from code/):  python analyze_sensitivity_koa2.py [results_dir]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def load():
    s = pd.read_csv(os.path.join(RES, "sensitivity_koa2.csv"))
    s = s[s.status == "ok"].copy()
    s["value"] = s.value.astype(str).str.replace(r"\.0$", "", regex=True)   # "10" and "10.0" are the same setting
    s = s.drop_duplicates(["problem", "param", "value", "seed"], keep="last")
    s["config"] = np.where(s.param == "default", "default", s.param + "=" + s.value)
    return s


def main():
    s = load()
    D = s[s.param == "default"]
    print(len(s), "unique successful runs (expected 2340)")
    g = s.groupby(["param", "value", "problem"])
    summ = g[["HV", "IGD"]].agg(["mean", "median", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    dmed = D.groupby("problem").IGD.median()
    summ["IGD_median_change_pct"] = [(r.IGD_median / dmed[r.problem] - 1) * 100 for r in summ.itertuples()]
    summ.to_csv(os.path.join(RES, "sensitivity_summary.csv"), index=False)

    out = []
    for ind, low in [("IGD", True), ("HV", False)]:
        rows = []
        for (par, v), grp in s[s.param != "default"].groupby(["param", "value"]):
            for p in sorted(s.problem.unique()):
                a = grp[grp.problem == p].set_index("seed")[ind]
                b = D[D.problem == p].set_index("seed")[ind]
                a, b = a.align(b, join="inner")
                pv = 1.0 if np.allclose(a, b) else wilcoxon(a, b).pvalue
                rows.append(dict(Metric=ind, param=par, value=v, problem=p, N_pairs=len(a), p_value=pv,
                                 mean=a.mean(), default_mean=b.mean(),
                                 mean_change_pct=(a.mean() / b.mean() - 1) * 100))
        t = pd.DataFrame(rows); t["p_holm"] = holm(t.p_value)
        better = np.where(low, t["mean"] < t.default_mean, t["mean"] > t.default_mean)
        t["Outcome"] = np.where(t.p_holm >= 0.05, "no significant difference",
                                np.where(better, "better than default", "worse than default"))
        out.append(t)
        print(ind, (t.p_holm < 0.05).sum(), "of", len(t), "comparisons significant after Holm")
    pd.concat(out).to_csv(os.path.join(RES, "sensitivity_tests.csv"), index=False)

    fr = []
    for ind, low in [("IGD", True), ("HV", False)]:
        mean = s.groupby(["problem", "config"])[ind].mean().unstack()
        rk = mean.rank(axis=1, ascending=low).mean()
        st = friedmanchisquare(*[mean[c] for c in mean.columns])
        for c in mean.columns:
            fr.append(dict(Metric=ind, config=c, Avg_Rank=rk[c], Friedman_stat=st.statistic, Friedman_p=st.pvalue))
    pd.DataFrame(fr).sort_values(["Metric", "Avg_Rank"]).to_csv(os.path.join(RES, "sensitivity_friedman.csv"), index=False)


if __name__ == "__main__":
    main()
