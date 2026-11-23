"""Renyi-DP accountant for the Poisson-subsampled Gaussian mechanism.

Self-contained (numpy/scipy only) re-implementation of the analysis in
Mironov, Talwar & Zhang (2019), "Renyi Differential Privacy of the Sampled
Gaussian Mechanism" -- the same analysis used by Opacus' RDPAccountant and
TF-Privacy. It lets the paper report *how* epsilon was computed and lets the
privacy-utility sweep pick sigma for a target epsilon without Opacus.
"""
import math
import numpy as np
from scipy.special import gammaln

DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def _log_add(a, b):
    m = max(a, b)
    if m == -np.inf:
        return m
    return m + math.log1p(math.exp(-abs(a - b)))


def _log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _log_a_int(q, sigma, alpha):
    log_a = -np.inf
    for i in range(int(alpha) + 1):
        log_coef = _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q)
        s = log_coef + (i * i - i) / (2 * sigma ** 2)
        log_a = _log_add(log_a, s)
    return float(log_a)


def _log_erfc(x):
    from scipy.special import log_ndtr
    return math.log(2) + log_ndtr(-x * 2 ** 0.5)


def _log_sub(a, b):
    if b == -np.inf:
        return a
    if a == b:
        return -np.inf
    return a + math.log1p(-math.exp(b - a))


def _log_a_frac(q, sigma, alpha):
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0
    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5
    while True:
        coef = gammaln(alpha + 1) - gammaln(i + 1) - gammaln(alpha - i + 1)
        log_coef = coef
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma ** 2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma ** 2) + log_e1
        log_a0 = _log_add(log_a0, log_s0)
        log_a1 = _log_add(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30:
            break
    return _log_add(log_a0, log_a1)


def rdp_sampled_gaussian(q, sigma, alpha):
    if q == 0:
        return 0.0
    if sigma == 0:
        return np.inf
    if q == 1.0:
        return alpha / (2 * sigma ** 2)
    if float(alpha).is_integer():
        return _log_a_int(q, sigma, int(alpha)) / (alpha - 1)
    return _log_a_frac(q, sigma, alpha) / (alpha - 1)


def epsilon(q, sigma, steps, delta, orders=DEFAULT_ORDERS, conversion="balle2020"):
    """Return (epsilon, best_order).

    conversion='balle2020' : eps = rdp - (log(delta)+log(a))/(a-1) + log((a-1)/a)
                             (Balle et al. 2020; used by Opacus >= 1.0)
    conversion='mironov'   : eps = rdp + log(1/delta)/(a-1)   (classic)
    """
    best = (np.inf, None)
    for a in orders:
        r = steps * rdp_sampled_gaussian(q, sigma, a)
        if conversion == "balle2020":
            e = r - (math.log(delta) + math.log(a)) / (a - 1) + math.log((a - 1) / a)
        else:
            e = r + math.log(1 / delta) / (a - 1)
        if e < best[0]:
            best = (e, a)
    return best


def sigma_for_epsilon(target_eps, q, steps, delta, lo=0.3, hi=50.0, tol=1e-3, **kw):
    """Smallest noise multiplier achieving epsilon <= target (bisection)."""
    if epsilon(q, hi, steps, delta, **kw)[0] > target_eps:
        raise ValueError("target epsilon unreachable with sigma <= %g" % hi)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta, **kw)[0] > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
