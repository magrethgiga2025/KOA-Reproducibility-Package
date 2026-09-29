"""
analyze_ablation.py
=====================
Turns results/ablation_final_raw.csv (and, if present,
results/factorial_ablation_raw.csv) into a proper ablation table:
Friedman rank of each variant ACROSS PROBLEMS, plus Holm-corrected
paired Wilcoxon W/T/L of each variant against full KOA on each problem
-- the same methodology the manuscript's Table 5/6 already use for the
main comparison, not a raw mean across problems.

This exists because a raw per-problem mean IGD (or HV) is dominated by
whichever problems happen to have the largest numerical scale (MaF4,
DTLZ1), which is exactly the kind of scale-sensitivity mistake the
review criticizes elsewhere in the paper -- averaging across problems
with wildly different IGD magnitudes is not a valid way to compare
ablation variants, and produced a misleading picture when tried
informally (adaptive geometry survival looked WORSE than several fixed-
rho variants on raw mean IGD, which does not survive switching to
rank-based comparison -- see the printed output when you run this).

Outputs:
  results/ablation_ranking.csv        (Friedman rank per variant, HV & IGD)
  results/ablation_wtl.csv            (Holm-corrected W/T/L, full KOA vs
                                        each other variant, per problem)
  results/factorial_ranking.csv       (if factorial_ablation_raw.csv exists)

Usage: python analyze_ablation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.stats import vargha_delaney_a12, a12_magnitude, holm_correction

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ABLATION_CSV = RESULTS_DIR / "ablation_final_raw.csv"
FACTORIAL_CSV = RESULTS_DIR / "factorial_ablation_raw.csv"

METRICS = ["HV", "IGD"]
HIGHER_IS_BETTER = {"HV": True, "IGD": False}


def friedman_rank_variants(df, group_col="Variant", value_cols=("HV", "IGD"),
                            problem_col="Problem"):
    """Friedman rank of each value in `group_col`, across the problems in
    `problem_col`, for each metric in `value_cols`. Mirrors
    analyze_results.friedman_ranking() but generalised to any set of
    variant labels (ablation variants, factorial cells, ...)."""
    ok = df[df.status == "ok"] if "status" in df.columns else df
    variants = sorted(ok[group_col].unique())
    rows = []
    for m in value_cols:
        pivot = (ok.groupby([problem_col, group_col])[m]
                 .mean().unstack(group_col).reindex(columns=variants))
        pivot = pivot.dropna(how="all")
        if pivot.shape[0] < 2:
            for v in variants:
                rows.append({"Metric": m, group_col: v, "Avg_Rank": np.nan,
                             "Friedman_stat": np.nan, "Friedman_p": np.nan,
                             "N_problems": pivot.shape[0]})
            continue
        worst_fill = {v: (pivot[v].min() if HIGHER_IS_BETTER[m] else pivot[v].max())
                      for v in variants}
        pivot_filled = pivot.fillna(worst_fill)
        ascending = not HIGHER_IS_BETTER[m]
        ranks = pivot_filled.rank(axis=1, ascending=ascending)
        avg_rank = ranks.mean(axis=0)
        try:
            stat, p = stats.friedmanchisquare(*[pivot_filled[v].values for v in variants])
        except ValueError:
            stat, p = np.nan, np.nan
        for v in variants:
            rows.append({"Metric": m, group_col: v, "Avg_Rank": float(avg_rank[v]),
                         "Friedman_stat": float(stat) if pd.notna(stat) else np.nan,
                         "Friedman_p": float(p) if pd.notna(p) else np.nan,
                         "N_problems": pivot.shape[0]})
    return pd.DataFrame(rows)


def holm_wtl_vs_reference(df, reference_label, group_col="Variant",
                           value_cols=("HV", "IGD"), problem_col="Problem",
                           seed_col="seed"):
    """Holm-corrected paired Wilcoxon of every OTHER value in `group_col`
    against `reference_label` (full KOA), per problem, with A12 effect
    size -- same protocol as analyze_results.wilcoxon_tests(), applied
    to ablation variants instead of baseline algorithms."""
    ok = df[df.status == "ok"] if "status" in df.columns else df
    others = [v for v in sorted(ok[group_col].unique()) if v != reference_label]
    problems = sorted(ok[problem_col].unique())

    raw_rows = []
    for m in value_cols:
        for problem in problems:
            ref = (ok[(ok[problem_col] == problem) & (ok[group_col] == reference_label)]
                   .set_index(seed_col)[m])
            for other in others:
                oth = (ok[(ok[problem_col] == problem) & (ok[group_col] == other)]
                       .set_index(seed_col)[m])
                common = ref.index.intersection(oth.index)
                a = ref.loc[common].replace([np.inf, -np.inf], np.nan)
                b = oth.loc[common].replace([np.inf, -np.inf], np.nan)
                paired = pd.concat([a, b], axis=1).dropna()
                if len(paired) < 5:
                    raw_rows.append({"Metric": m, "Problem": problem, "Variant": other,
                                     "N_pairs": len(paired), "p_value": np.nan})
                    continue
                try:
                    _, p = stats.wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1])
                except ValueError:
                    p = np.nan
                ref_vals, oth_vals = paired.iloc[:, 0].values, paired.iloc[:, 1].values
                ref_mean, oth_mean = ref_vals.mean(), oth_vals.mean()
                koa_better = ((ref_mean > oth_mean) if HIGHER_IS_BETTER[m]
                              else (ref_mean < oth_mean))
                a12 = (vargha_delaney_a12(ref_vals, oth_vals) if HIGHER_IS_BETTER[m]
                       else vargha_delaney_a12(oth_vals, ref_vals))
                raw_rows.append({"Metric": m, "Problem": problem, "Variant": other,
                                 "N_pairs": len(paired), "p_value": p,
                                 "KOA_better": bool(koa_better),
                                 "A12": a12, "A12_magnitude": a12_magnitude(a12)})

    result = pd.DataFrame(raw_rows)
    # Holm correction within each metric's full family of (problem, variant)
    # comparisons -- mirrors the manuscript's stated protocol for the main
    # comparison table, applied consistently here per the review's request
    # ("Table VII should state whether its W/T/L calls are Holm corrected").
    result["significant_holm"] = False
    for m in value_cols:
        mask = result.Metric == m
        pvals = result.loc[mask, "p_value"].values
        sig = holm_correction(pvals, alpha=0.05)
        result.loc[mask, "significant_holm"] = sig
    return result


def wtl_summary(holm_df, group_col="Variant"):
    rows = []
    for m in holm_df.Metric.unique():
        for v in holm_df[holm_df.Metric == m][group_col].unique():
            sub = holm_df[(holm_df.Metric == m) & (holm_df[group_col] == v)]
            W = int(((sub.KOA_better == True) & (sub.significant_holm == True)).sum())
            L = int(((sub.KOA_better == False) & (sub.significant_holm == True)).sum())
            T = int((sub.significant_holm == False).sum())
            rows.append({"Metric": m, group_col: v, "W": W, "T": T, "L": L})
    return pd.DataFrame(rows)


def main():
    if not ABLATION_CSV.exists():
        print(f"{ABLATION_CSV} not found -- run run_ablation_final.py first.")
        return

    df = pd.read_csv(ABLATION_CSV)
    print(f"Loaded {len(df)} ablation rows, variants={sorted(df.Variant.unique())}")

    rank_df = friedman_rank_variants(df)
    rank_df.to_csv(RESULTS_DIR / "ablation_ranking.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'ablation_ranking.csv'}")
    print("\n=== ABLATION FRIEDMAN RANK (lower = better; 'KOA' = full algorithm) ===")
    print(rank_df.pivot(index="Variant", columns="Metric", values="Avg_Rank")
          .sort_values("IGD").round(3).to_string())

    holm_df = holm_wtl_vs_reference(df, reference_label="KOA")
    holm_df.to_csv(RESULTS_DIR / "ablation_holm_tests.csv", index=False)
    wtl = wtl_summary(holm_df)
    wtl.to_csv(RESULTS_DIR / "ablation_wtl.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'ablation_wtl.csv'}")
    print("\n=== FULL KOA vs EACH VARIANT: Holm-corrected W/T/L "
          "(win = full KOA significantly better) ===")
    print(wtl.pivot(index="Variant", columns="Metric", values=["W", "T", "L"])
          .to_string())

    if FACTORIAL_CSV.exists():
        fdf = pd.read_csv(FACTORIAL_CSV)
        frank = friedman_rank_variants(fdf, group_col="cell", problem_col="problem")
        frank.to_csv(RESULTS_DIR / "factorial_ranking.csv", index=False)
        print(f"\nWrote {RESULTS_DIR / 'factorial_ranking.csv'}")
        print("\n=== FACTORIAL CELL FRIEDMAN RANK (S/A/G = mechanism ON, "
              "s/a/g = OFF; e.g. 'SAG' = full KOA, 'sag' = all three off) ===")
        print(frank.pivot(index="cell", columns="Metric", values="Avg_Rank")
              .sort_values("IGD").round(3).to_string())
    else:
        print(f"\n({FACTORIAL_CSV.name} not found -- skipping factorial "
              "ranking; run run_factorial_ablation.py to generate it.)")


if __name__ == "__main__":
    main()
