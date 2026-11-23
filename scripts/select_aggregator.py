#!/usr/bin/env python3
"""Choose the recommended aggregator for SecureFL-CyberDef on VALIDATION data only.

For every DP-SGD aggregator in the main group (and in the attack matrix), report the
validation AUC-ROC of the selected checkpoint (best round), mean +- sd over seeds.
The test set is not read. The aggregator with the highest mean is the recommended one.

  python scripts/select_aggregator.py --results results
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); a = ap.parse_args()


def val_best(run_dir):
    v = []
    for f in glob.glob(os.path.join(run_dir, "seed*.json")):
        L = json.load(open(f))
        aucs = [r["val_auc"] for r in L["rounds"] if r.get("val_auc") == r.get("val_auc")]  # drop NaN
        v.append(max(aucs) if aucs else float("nan"))
    return np.array(v)


print("Main setting (sign-flip), DP-SGD local training, validation AUC-ROC:")
main = {"median": "securefl_dpsgd_median", "multi_krum": "dpsgd_multi_krum_signflip",
        "krum": "dpsgd_krum_signflip", "trimmed_mean": "dpsgd_trimmed_mean_signflip", "fedavg": "dpsgd_fedavg_signflip"}
for agg, n in main.items():
    v = val_best(os.path.join(a.results, "main", n))
    if len(v):
        print(f"  {agg:13s} {v.mean():.5f} +- {v.std(ddof=1):.5f}  (n={len(v)})")
print("\nAll seven attacks (attacks group), mean over attacks of validation AUC-ROC; worst attack in brackets:")
for agg in ["fedavg", "median", "trimmed_mean", "krum", "multi_krum"]:
    per = []
    for d in glob.glob(os.path.join(a.results, "attacks", f"{agg}__*")):
        v = val_best(d)
        if len(v):
            per.append((v.mean(), os.path.basename(d).split("__")[1]))
    if per:
        m = np.mean([p[0] for p in per]); w = min(per)
        print(f"  {agg:13s} mean {m:.5f}   worst {w[0]:.5f} ({w[1]})")
