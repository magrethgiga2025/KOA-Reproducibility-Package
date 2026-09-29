"""
utils/stats.py

Effect sizes and confidence intervals for paired algorithm comparisons,
answering the review's "report effect sizes in addition to significance
... Vargha-Delaney A12, Cliff's delta, or rank-biserial correlation"
and "report 95% confidence intervals for mean or median HV/IGD."
"""
import numpy as np
from scipy import stats


def vargha_delaney_a12(x, y):
    """
    Vargha-Delaney A12 effect size for two independent (or paired,
    treated pointwise) samples x and y: the probability that a randomly
    drawn value from x is greater than a randomly drawn value from y
    (plus half the probability of a tie).

    A12 = 0.5   -> no difference
    A12 > 0.5   -> x tends to be larger than y
    A12 < 0.5   -> x tends to be smaller than y

    Common qualitative thresholds (Vargha & Delaney, 2000): |A12-0.5| <
    0.06 negligible, < 0.14 small, < 0.21 medium, otherwise large.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return np.nan
    # rank-sum formulation, equivalent to counting pairwise comparisons
    # but O((m+n)log(m+n)) instead of O(mn)
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)
    r1 = ranks[:m].sum()
    a12 = (r1 / m - (m + 1) / 2) / n
    return float(a12)


def a12_magnitude(a12):
    """Qualitative label for |A12 - 0.5| per Vargha & Delaney (2000)."""
    if a12 is None or np.isnan(a12):
        return "n/a"
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    elif d < 0.14:
        return "small"
    elif d < 0.21:
        return "medium"
    return "large"


def cliffs_delta(x, y):
    """
    Cliff's delta, linearly related to A12 by delta = 2*A12 - 1.
    delta in [-1, 1]; 0 = no difference, +1 = every x > every y,
    -1 = every x < every y. Reported here since the review explicitly
    names it as an acceptable alternative to A12.
    """
    a12 = vargha_delaney_a12(x, y)
    if np.isnan(a12):
        return np.nan
    return float(2 * a12 - 1)


def rank_biserial_from_wilcoxon(x, y):
    """
    Matched-pairs rank-biserial correlation for a paired Wilcoxon
    signed-rank test: r = (W+ - W-) / (W+ + W-), where W+/W- are the
    sums of positive/negative signed ranks of the paired differences
    d = x - y. r in [-1, 1]; the third effect-size measure the review
    names as acceptable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[d != 0]  # ties contribute zero rank and are dropped, standard convention
    if len(d) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    total = w_pos + w_neg
    if total == 0:
        return np.nan
    return float((w_pos - w_neg) / total)


def bootstrap_ci_mean(x, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI for the mean of x. Non-parametric --
    appropriate given the review's own observation that HV/IGD
    distributions are often skewed or heavy-tailed, so a normal-
    approximation CI would understate uncertainty on exactly the
    problems (DTLZ1, MaF4) where it matters most."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def bootstrap_ci_median(x, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI for the median of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    lo = np.percentile(boot_meds, 100 * alpha / 2)
    hi = np.percentile(boot_meds, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def holm_correction(pvalues, alpha=0.05):
    """
    Holm-Bonferroni step-down correction for a family of p-values.
    Self-contained (no statsmodels dependency) implementation of the
    same procedure the manuscript already names ("Holm correction
    applied across all 96 comparisons per indicator").

    Returns a boolean array, same order and length as `pvalues`, True
    where the corresponding null hypothesis is rejected at level alpha.
    NaN p-values are never rejected (treated as untestable, not as
    non-significant by omission -- callers should track these
    separately if the distinction matters).
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    reject = np.zeros(n, dtype=bool)
    valid_idx = np.where(~np.isnan(pvalues))[0]
    if len(valid_idx) == 0:
        return reject
    order = valid_idx[np.argsort(pvalues[valid_idx])]
    m = len(order)
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvalues[idx] <= threshold:
            reject[idx] = True
        else:
            break  # Holm step-down: stop at first non-rejection
    return reject
