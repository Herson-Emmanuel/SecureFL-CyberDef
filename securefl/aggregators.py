"""Server-side aggregation rules operating on flattened client updates.

All functions take ``U`` with shape (M, d) -- one row per client update -- and
return a single aggregated update of shape (d,). They are pure numpy so they
are framework-independent, unit-testable, and their cost can be benchmarked in
isolation (scripts/bench_aggregators.py).

Rules implemented (references in the revised manuscript):
  fedavg            McMahan et al. 2017               O(M d)
  median            Yin et al. 2018 (coordinate-wise) O(M d)  (quickselect; O(M d log M) with sorting)
  trimmed_mean      Yin et al. 2018                   O(M d log M)
  krum, multi_krum  Blanchard et al. 2017             O(M^2 d)
  bulyan            El Mhamdi et al. 2018             O(M^2 d)
  geomed (RFA)      Pillutla et al. 2022              O(T M d)
  centered_clip     Karimireddy et al. 2021           O(T M d)
"""
from __future__ import annotations

import numpy as np


class AggregationError(ValueError):
    pass


def fedavg(U, weights=None, **_):
    U = np.asarray(U, dtype=np.float64)
    if weights is None:
        return U.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    return (w[:, None] * U).sum(axis=0) / w.sum()


def median(U, **_):
    """Coordinate-wise median, Eq. (6)-(7) of the paper.

    np.median uses introselect (partition), i.e. expected O(M) per coordinate.
    """
    return np.median(np.asarray(U, dtype=np.float64), axis=0)


def trimmed_mean(U, f=1, beta=None, **_):
    """Coordinate-wise trimmed mean: drop the k largest and k smallest values.

    k = f by default (number of assumed Byzantine clients); if ``beta`` is given
    k = floor(beta * M). Requires M > 2k.
    """
    U = np.asarray(U, dtype=np.float64)
    M = U.shape[0]
    k = int(np.floor(beta * M)) if beta is not None else int(f)
    if M <= 2 * k:
        raise AggregationError(f"trimmed_mean needs M > 2k (M={M}, k={k})")
    S = np.sort(U, axis=0)
    return S[k:M - k].mean(axis=0)


def _pairwise_sq_dists(U):
    sq = (U * U).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * U @ U.T
    np.maximum(D, 0.0, out=D)
    return D


def krum_scores(U, f):
    U = np.asarray(U, dtype=np.float64)
    M = U.shape[0]
    if M < 2 * f + 3:
        raise AggregationError(f"Krum needs M >= 2f+3 (M={M}, f={f})")
    D = _pairwise_sq_dists(U)
    n_neigh = M - f - 2
    scores = np.empty(M)
    for i in range(M):
        d = np.delete(D[i], i)
        scores[i] = np.partition(d, n_neigh - 1)[:n_neigh].sum()
    return scores


def krum(U, f=1, **_):
    scores = krum_scores(U, f)
    return np.asarray(U, dtype=np.float64)[int(np.argmin(scores))].copy()


def multi_krum(U, f=1, m=None, **_):
    """Average of the m updates with the lowest Krum scores (default m = M - f)."""
    U = np.asarray(U, dtype=np.float64)
    M = U.shape[0]
    m = M - f if m is None else int(m)
    scores = krum_scores(U, f)
    idx = np.argsort(scores)[:m]
    return U[idx].mean(axis=0)


def bulyan(U, f=1, **_):
    """Bulyan (El Mhamdi et al. 2018): iterative Krum selection of theta = M - 2f
    updates followed by coordinate-wise trimmed mean keeping beta = theta - 2f
    values closest to the median. Requires M >= 4f + 3.
    """
    U = np.asarray(U, dtype=np.float64)
    M = U.shape[0]
    if M < 4 * f + 3:
        raise AggregationError(f"Bulyan needs M >= 4f+3 (M={M}, f={f})")
    theta = M - 2 * f
    remaining = list(range(M))
    selected = []
    while len(selected) < theta:
        sub = U[remaining]
        # Krum on the remaining set; guard for small sets near the end
        if len(remaining) >= 2 * f + 3:
            j = int(np.argmin(krum_scores(sub, f)))
        else:
            j = int(np.argmin(_pairwise_sq_dists(sub).sum(axis=1)))
        selected.append(remaining.pop(j))
    S = U[selected]                       # (theta, d)
    beta = theta - 2 * f
    med = np.median(S, axis=0)
    order = np.argsort(np.abs(S - med), axis=0)[:beta]
    closest = np.take_along_axis(S, order, axis=0)
    return closest.mean(axis=0)


def geomed(U, iters=10, eps=1e-8, weights=None, **_):
    """Geometric median via smoothed Weiszfeld (RFA, Pillutla et al. 2022)."""
    U = np.asarray(U, dtype=np.float64)
    w = np.ones(U.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    z = (w[:, None] * U).sum(0) / w.sum()
    for _ in range(iters):
        d = np.linalg.norm(U - z, axis=1)
        beta = w / np.maximum(d, eps)
        z = (beta[:, None] * U).sum(0) / beta.sum()
    return z


def centered_clip(U, tau=10.0, iters=3, v0=None, **_):
    """Centered clipping (Karimireddy et al. 2021); v0 = previous aggregate."""
    U = np.asarray(U, dtype=np.float64)
    v = np.zeros(U.shape[1]) if v0 is None else np.asarray(v0, dtype=np.float64)
    for _ in range(iters):
        diff = U - v
        n = np.linalg.norm(diff, axis=1, keepdims=True)
        scale = np.minimum(1.0, tau / np.maximum(n, 1e-12))
        v = v + (diff * scale).mean(axis=0)
    return v


AGGREGATORS = {
    "fedavg": fedavg,
    "median": median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
    "bulyan": bulyan,
    "geomed": geomed,
    "centered_clip": centered_clip,
}

# Big-O of the server step, for the complexity table (M clients, d params, T iters)
COMPLEXITY = {
    "fedavg": "O(Md)",
    "median": "O(Md) expected (O(Md log M) with full sort)",
    "trimmed_mean": "O(Md log M)",
    "krum": "O(M^2 d)",
    "multi_krum": "O(M^2 d)",
    "bulyan": "O(M^2 d)",
    "geomed": "O(TMd)",
    "centered_clip": "O(TMd)",
}


def min_clients(rule: str, f: int) -> int:
    return {"krum": 2 * f + 3, "multi_krum": 2 * f + 3, "bulyan": 4 * f + 3,
            "trimmed_mean": 2 * f + 1, "median": 2 * f + 1}.get(rule, 1)


def aggregate(rule: str, U, **kw):
    try:
        fn = AGGREGATORS[rule]
    except KeyError as e:
        raise AggregationError(f"unknown aggregator {rule!r}") from e
    return fn(U, **kw)
