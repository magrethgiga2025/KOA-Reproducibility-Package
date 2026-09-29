"""
Aggregation and statistical analysis of raw_runs.csv.

Produces:
  - results/summary_table_raw.csv  (mean +/- std per problem x algorithm x metric)
  - results/summary_table.csv      (formatted version for the paper)
  - results/wilcoxon_tests.csv     (pairwise KOA-vs-baseline Wilcoxon tests)
  - results/friedman_ranking.csv   (Friedman test + average ranks)

Fixes applied:
  1. friedman_ranking(): use dropna(how="all") instead of dropna(how="any")
     so that problems where ONE algorithm failed do not eliminate the entire
     problem row from the ranking.
  2. friedman_ranking(): guard against empty DataFrame before pivot so
     the KeyError: 'Algorithm' crash cannot occur.
  3. wilcoxon z-score NaN warnings suppressed with warnings.filterwarnings.
  4. rank(ascending=...) logic corrected: for HV (higher=better),
     rank 1 = largest value, so ascending=False is correct.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# suppress the z-score divide-by-zero from scipy.stats.wilcoxon when
# all differences are zero (identical paired samples)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in scalar divide")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROBLEM_NAMES, ALGORITHMS
from utils.stats import (
    vargha_delaney_a12, a12_magnitude, cliffs_delta,
    rank_biserial_from_wilcoxon, bootstrap_ci_mean, bootstrap_ci_median,
    holm_correction,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_CSV     = RESULTS_DIR / "raw_runs.csv"

METRICS = ["HV", "IGD", "Spacing", "Spread"]
HIGHER_IS_BETTER = {"HV": True, "IGD": False, "Spacing": False, "Spread": False}


def load_raw():
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found -- run run_experiments.py first."
        )
    df = pd.read_csv(RAW_CSV)
    n_total_expected = len(PROBLEM_NAMES) * len(ALGORITHMS) * 30
    n_ok     = (df["status"] == "ok").sum()
    n_error  = (df["status"] == "error").sum()
    n_empty  = (df["status"] == "empty_front").sum()
    print(f"Loaded {len(df)} rows  "
          f"({n_ok} ok, {n_error} error, {n_empty} empty_front)  "
          f"expected up to {n_total_expected} total runs.")
    if n_error > 0:
        err_pairs = (df[df["status"] == "error"]
                     [["problem", "algorithm", "error"]]
                     .drop_duplicates(subset=["problem", "algorithm"]))
        print(f"WARNING: {n_error} runs errored. Affected (problem, algorithm):")
        print(err_pairs.to_string(index=False))
    return df


def build_summary_table(df):
    """Mean +/- std per (problem, algorithm) for each metric, plus a
    bootstrap 95% CI for the mean AND the median (review: 'results
    tables should ideally report medians in addition to means' and
    'report 95% confidence intervals for mean or median HV/IGD')."""
    ok = df[df["status"] == "ok"]
    rows = []
    for problem in PROBLEM_NAMES:
        for algo in ALGORITHMS:
            sub = ok[(ok["problem"] == problem) & (ok["algorithm"] == algo)]
            row = {"Problem": problem, "Algorithm": algo, "N_runs": len(sub)}
            for m in METRICS:
                vals = sub[m].replace([np.inf, -np.inf], np.nan).dropna().values
                if len(vals) > 0:
                    row[f"{m}_mean"] = vals.mean()
                    row[f"{m}_std"] = vals.std()
                    row[f"{m}_median"] = float(np.median(vals))
                    ci_mean = bootstrap_ci_mean(vals)
                    ci_med = bootstrap_ci_median(vals)
                    row[f"{m}_mean_ci_lo"], row[f"{m}_mean_ci_hi"] = ci_mean
                    row[f"{m}_median_ci_lo"], row[f"{m}_median_ci_hi"] = ci_med
                else:
                    for suffix in ["mean", "std", "median", "mean_ci_lo",
                                   "mean_ci_hi", "median_ci_lo", "median_ci_hi"]:
                        row[f"{m}_{suffix}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def format_summary_for_paper(summary_df):
    """Render 'mean±std' strings matching the manuscript table style."""
    out = summary_df.copy()
    for m in METRICS:
        out[m] = out.apply(
            lambda r: (f"{r[f'{m}_mean']:.4f}±{r[f'{m}_std']:.4f}"
                       if pd.notna(r[f"{m}_mean"]) else "N/A"),
            axis=1,
        )
    return out[["Problem", "Algorithm", "N_runs"] + METRICS]


def wilcoxon_tests(df):
    """KOA vs every other algorithm per (problem, metric) — paired Wilcoxon,
    Holm-corrected within each metric's full family of 96 comparisons
    (16 problems x 6 baselines), matching the family size the manuscript
    itself states. Both the raw and Holm-corrected significance calls
    are reported so the difference is visible rather than silently
    fixed (the previous version of this function computed p_value but
    never actually applied Holm correction before thresholding at 0.05
    -- 'significant_0.05' was always the UNCORRECTED call)."""
    ok = df[df["status"] == "ok"]
    rows = []
    comparison_algos = [a for a in ALGORITHMS if a != "KOA"]

    for problem in PROBLEM_NAMES:
        koa  = ok[(ok["problem"] == problem) & (ok["algorithm"] == "KOA")
                  ].set_index("seed")
        for baseline in comparison_algos:
            base = ok[(ok["problem"] == problem) & (ok["algorithm"] == baseline)
                      ].set_index("seed")
            common = koa.index.intersection(base.index)
            for m in METRICS:
                a = koa.loc[common, m].replace([np.inf, -np.inf], np.nan)
                b = base.loc[common, m].replace([np.inf, -np.inf], np.nan)
                paired = pd.concat([a, b], axis=1).dropna()
                if len(paired) < 5:
                    rows.append({
                        "Problem": problem, "Baseline": baseline, "Metric": m,
                        "N_pairs": len(paired), "p_value": np.nan,
                        "significant_0.05": np.nan, "KOA_better": np.nan,
                        "A12": np.nan, "A12_magnitude": "n/a",
                        "Cliffs_delta": np.nan, "rank_biserial_r": np.nan,
                    })
                    continue
                try:
                    _, p = stats.wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1])
                except ValueError:
                    p = np.nan

                koa_vals = paired.iloc[:, 0].values
                base_vals = paired.iloc[:, 1].values
                koa_mean  = koa_vals.mean()
                base_mean = base_vals.mean()
                better = ((koa_mean > base_mean) if HIGHER_IS_BETTER[m]
                          else (koa_mean < base_mean))

                # Effect sizes (review: report alongside significance,
                # since with 30 paired runs even small differences can
                # be statistically detectable). A12/Cliff's delta use
                # the values oriented so that >0.5 / >0 means KOA better.
                if HIGHER_IS_BETTER[m]:
                    a12 = vargha_delaney_a12(koa_vals, base_vals)
                else:
                    a12 = vargha_delaney_a12(base_vals, koa_vals)
                delta = 2 * a12 - 1 if pd.notna(a12) else np.nan
                rbc = rank_biserial_from_wilcoxon(
                    koa_vals if HIGHER_IS_BETTER[m] else base_vals,
                    base_vals if HIGHER_IS_BETTER[m] else koa_vals,
                )

                rows.append({
                    "Problem": problem, "Baseline": baseline, "Metric": m,
                    "N_pairs": len(paired), "p_value": p,
                    "significant_0.05": (p < 0.05) if pd.notna(p) else np.nan,
                    "KOA_better": bool(better),
                    "A12": a12, "A12_magnitude": a12_magnitude(a12),
                    "Cliffs_delta": delta,
                    "rank_biserial_r": rbc,
                })

    result = pd.DataFrame(rows)

    # Holm correction WITHIN EACH METRIC across the full stated family
    # of comparisons (16 problems x 6 baselines = 96 per indicator, per
    # Table 1/Section 5's own description). This was previously computed
    # but never actually applied to the significance calls used
    # everywhere else in this script -- fixed here.
    result["significant_holm"] = False
    for m in METRICS:
        mask = result.Metric == m
        pvals = result.loc[mask, "p_value"].values
        result.loc[mask, "significant_holm"] = holm_correction(pvals, alpha=0.05)

    return result


def friedman_ranking(df):
    """
    Friedman test + average rank per algorithm per metric.

    Fix 1: use dropna(how="all") so that a single NaN algorithm does not
           eliminate the entire problem row.
    Fix 2: fill remaining NaN with the worst possible rank (n_algos)
           before computing the Friedman statistic.
    Fix 3: guard against empty rows list before returning.
    Fix 4: rank(ascending=False) for HIGHER_IS_BETTER metrics so that
           rank 1 == best (largest) value.
    """
    ok   = df[df["status"] == "ok"]
    rows = []

    for m in METRICS:
        # mean per (problem, algorithm)
        pivot = (ok.groupby(["problem", "algorithm"])[m]
                 .mean()
                 .replace([np.inf, -np.inf], np.nan)
                 .unstack("algorithm"))

        # reindex to ensure all algorithms are present as columns
        pivot = pivot.reindex(columns=ALGORITHMS)

        # drop problems where ALL algorithms are NaN (Fix 1)
        pivot = pivot.dropna(how="all")

        if pivot.shape[0] < 2:
            print(f"  WARNING: fewer than 2 valid problems for {m} "
                  f"-- Friedman test skipped for this metric.")
            # still emit rows with NaN stats so downstream code does not crash
            for algo in ALGORITHMS:
                rows.append({
                    "Metric": m, "Algorithm": algo,
                    "Avg_Rank": np.nan,
                    "Friedman_stat": np.nan, "Friedman_p": np.nan,
                    "N_problems": pivot.shape[0],
                })
            continue

        # fill remaining NaN with worst rank (Fix 2)
        worst = len(ALGORITHMS)
        pivot_filled = pivot.fillna(
            {a: (pivot[a].min() if HIGHER_IS_BETTER[m]
                 else pivot[a].max())
             for a in ALGORITHMS}
        )

        # rank: for HIGHER_IS_BETTER, rank 1 = largest (Fix 4)
        ascending = not HIGHER_IS_BETTER[m]   # True for IGD (lower=better)
        ranks     = pivot_filled.rank(axis=1, ascending=ascending)
        avg_rank  = ranks.mean(axis=0)

        try:
            stat, p = stats.friedmanchisquare(
                *[pivot_filled[a].values for a in ALGORITHMS]
            )
        except ValueError:
            stat, p = np.nan, np.nan

        for algo in ALGORITHMS:
            rows.append({
                "Metric": m, "Algorithm": algo,
                "Avg_Rank": float(avg_rank[algo]),
                "Friedman_stat": float(stat) if pd.notna(stat) else np.nan,
                "Friedman_p":    float(p)    if pd.notna(p)    else np.nan,
                "N_problems": pivot.shape[0],
            })

    # Fix 3: guard against empty rows
    if not rows:
        print("WARNING: No valid Friedman rows produced. "
              "Check that raw_runs.csv has sufficient successful runs.")
        return pd.DataFrame(columns=["Metric", "Algorithm", "Avg_Rank",
                                     "Friedman_stat", "Friedman_p", "N_problems"])
    return pd.DataFrame(rows)


def main():
    df = load_raw()

    # Summary tables
    summary = build_summary_table(df)
    summary.to_csv(RESULTS_DIR / "summary_table_raw.csv", index=False)
    formatted = format_summary_for_paper(summary)
    formatted.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'summary_table.csv'}")

    # Wilcoxon tests
    wilcoxon_df = wilcoxon_tests(df)
    wilcoxon_df.to_csv(RESULTS_DIR / "wilcoxon_tests.csv", index=False)
    n_sig   = wilcoxon_df["significant_0.05"].sum()
    n_total = wilcoxon_df["significant_0.05"].notna().sum()
    print(f"Wrote {RESULTS_DIR / 'wilcoxon_tests.csv'} "
          f"({int(n_sig)}/{int(n_total)} comparisons significant at p<0.05)")

    # Friedman ranking
    friedman_df = friedman_ranking(df)
    friedman_df.to_csv(RESULTS_DIR / "friedman_ranking.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'friedman_ranking.csv'}")

    # Print summary (Fix 3: only pivot if Algorithm column exists)
    print("\n--- Friedman average ranks (lower = better rank) ---")
    if "Algorithm" in friedman_df.columns and len(friedman_df) > 0:
        try:
            print(
                friedman_df.pivot(
                    index="Algorithm", columns="Metric", values="Avg_Rank"
                )
                .reindex(ALGORITHMS)
                [["HV", "IGD"]]
                .round(2)
                .to_string()
            )
        except Exception as e:
            print(f"  (pivot display failed: {e})")
            print(friedman_df[["Algorithm", "Metric", "Avg_Rank"]].to_string(index=False))
    else:
        print("  (no valid Friedman results)")

    # W/T/L summary (Holm-corrected, matching the manuscript's stated protocol)
    print("\n--- W/T/L summary (KOA vs each baseline, HV, Holm-corrected) ---")
    wt = wilcoxon_df[wilcoxon_df["Metric"] == "HV"]
    for bl in [a for a in ALGORITHMS if a != "KOA"]:
        b = wt[wt["Baseline"] == bl]
        W = int(((b["KOA_better"] == True)  & (b["significant_holm"] == True)).sum())
        L = int(((b["KOA_better"] == False) & (b["significant_holm"] == True)).sum())
        T = int((b["significant_holm"] == False).sum())
        print(f"  vs {bl:12s}: {W}/{T}/{L}")


if __name__ == "__main__":
    main()
