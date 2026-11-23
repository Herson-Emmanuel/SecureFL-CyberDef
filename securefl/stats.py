"""Statistics across random seeds (Major Issue 2).

Reporting convention used in the revised tables:
  mean +- sample standard deviation (ddof=1) over S seeds, and a 95% CI from the
  Student t distribution with S-1 degrees of freedom.

Significance (SecureFL vs each baseline, same seeds => paired design):
  * paired two-sided t-test on per-seed differences
  * Wilcoxon signed-rank test (exact). NOTE: with S = 5 seeds the smallest
    attainable two-sided p-value is 2/2^5 = 0.0625, so it can never reach 0.05;
    use S >= 6 (min p = 0.03125) -- we recommend S = 10 -- if you want to report it.
  * Mann-Whitney U (Wilcoxon rank-sum, the test the reviewer named) treats runs as
    independent; with 5 vs 5 the smallest two-sided p is 0.0079.
  * Holm-Bonferroni correction across the baselines compared.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def summarize(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return dict(n=n, mean=m, std=sd, ci_low=m - h, ci_high=m + h)


def paired_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = dict(mean_diff=float(d.mean()))
    out["t_p"] = float(stats.ttest_rel(a, b).pvalue) if np.any(d != d[0]) or d[0] != 0 else 1.0
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                                                 method="exact").pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    out["mannwhitney_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    out["cohens_dz"] = float(d.mean() / sd) if sd > 0 else float("inf")
    return out


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_ci(y, p, thr, metric_fn, B=1000, seed=0, conf=0.95):
    """Test-set bootstrap CI for a single run (complements seed variance)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(y[idx], p[idx], thr))
    lo, hi = np.percentile(vals, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return float(lo), float(hi)


def fmt_pm(x, digits=4):
    s = summarize(x)
    return f"{s['mean']:.{digits}f} $\\pm$ {s['std']:.{digits}f}"
