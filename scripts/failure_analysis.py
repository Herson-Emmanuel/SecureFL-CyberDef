#!/usr/bin/env python3
"""Per-attack-category recall at the calibrated threshold (Minor 19: failure cases).

  python scripts/failure_analysis.py --results results --group main --out tables
Writes tables/failure_by_category.tex : category | n | recall of each model (mean +- sd over seeds)
"""
import argparse, glob, json, os
import numpy as np

MODELS = [("fedavg_signflip", "FedAvg"), ("median_signflip", "Median"), ("krum_signflip", "Krum"),
          ("securefl_dpsgd_median", "SecureFL-Median"), ("dpsgd_multi_krum_signflip", "SecureFL-MK")]
ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results"); ap.add_argument("--group", default="main")
ap.add_argument("--out", default="tables")
a = ap.parse_args()
per = {}
for n, _ in MODELS:
    for f in glob.glob(os.path.join(a.results, a.group, n, "seed*.json")):
        L = json.load(open(f)); z = np.load(f.replace(".json", "_pred.npz"), allow_pickle=True)
        thr = L["test"]["best_f1"]["threshold"]
        cat, y, p = z["cat_test"].astype(str), z["y_test"], z["p_test"]
        for c in np.unique(cat[y == 1]):
            m = (cat == c)
            per.setdefault(c, {}).setdefault(n, []).append(float((p[m] >= thr).mean()))
            per[c]["_n"] = int(m.sum())
rows = []
for c in sorted(per, key=lambda c: -per[c]["_n"]):
    cells = [c.replace("_", "\\_").replace("&", "\\&"), f"{per[c]['_n']:,}"]
    for n, _ in MODELS:
        v = per[c].get(n, [])
        cells.append(f"{np.mean(v):.3f} $\\pm$ {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}" if v else "n/a")
    rows.append(" & ".join(cells) + " \\\\")
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, "failure_by_category.tex"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
