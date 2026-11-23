#!/usr/bin/env python3
"""Wall-clock cost of the server aggregation step vs. number of clients M
(Minor 16: complexity and scalability). Uses d = 21,249 (the DP-compatible
network) and f = floor(0.2 M) Byzantine clients.

  python scripts/bench_aggregators.py --out tables --fig figures
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from securefl.aggregators import AggregationError, aggregate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=21249)
ap.add_argument("--Ms", nargs="+", type=int, default=[5, 10, 20, 50, 100])
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", default="tables")
ap.add_argument("--fig", default="figures")
a = ap.parse_args()
rules = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "bulyan"]
rng = np.random.default_rng(0)
res = {r: [] for r in rules}
for M in a.Ms:
    U = rng.normal(size=(M, a.d))
    f = max(1, int(0.2 * M))
    for r in rules:
        try:
            aggregate(r, U, f=f)
            ts = []
            for _ in range(a.reps):
                t = time.perf_counter()
                aggregate(r, U, f=f)
                ts.append((time.perf_counter() - t) * 1e3)
            res[r].append(float(np.median(ts)))
        except AggregationError:
            res[r].append(float("nan"))
    print(M, {r: round(res[r][-1], 2) for r in rules})

os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "agg_timing.tex"), "w") as fh:
    for i, M in enumerate(a.Ms):
        fh.write(f"{M} & {max(1, int(0.2 * M))} & " +
                 " & ".join("n/a" if np.isnan(res[r][i]) else f"{res[r][i]:.2f}" for r in rules) + " \\\\\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
os.makedirs(a.fig, exist_ok=True)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for r, mk in zip(rules, "os^Dvx"):
    ax.plot(a.Ms, res[r], marker=mk, ms=4, lw=1.2, label=r.replace("_", " "))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of clients"); ax.set_ylabel("Aggregation time (ms)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7, ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(a.fig, "aggregation_time_vs_clients.png"), dpi=300)
print("wrote", os.path.join(a.out, "agg_timing.tex"))
