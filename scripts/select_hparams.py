#!/usr/bin/env python3
"""Pick hyperparameters from the tuning runs using VALIDATION AUC-ROC only
(best round, as in checkpoint selection). The test set is never read.
Writes configs/hparams.json, which run_suite.py then applies to every method.

  python scripts/select_hparams.py --results results
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
a = ap.parse_args()
rows = {"dp": [], "nodp": []}
for f in sorted(glob.glob(os.path.join(a.results, "tune", "*", "seed*.json"))):
    L = json.load(open(f)); c = L["config"]
    best = max(r["val_auc"] for r in L["rounds"])
    kind = "dp" if c["dp"] else "nodp"
    hp = {"lr": c["lr"], "local_epochs": c["local_epochs"]}
    if c["dp"]:
        hp["max_grad_norm"] = c["max_grad_norm"]
    rows[kind].append((best, hp, os.path.basename(os.path.dirname(f))))
out = {}
for kind in ["nodp", "dp"]:
    if not rows[kind]:
        continue
    print(f"\n{kind.upper()}  (validation AUC-ROC, best round)")
    for best, hp, name in sorted(rows[kind], key=lambda r: -r[0]):
        print(f"  {best:.5f}  {name}")
    out[kind] = max(rows[kind], key=lambda r: r[0])[1]
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "configs"), exist_ok=True)
path = os.path.join(os.path.dirname(__file__), "..", "configs", "hparams.json")
json.dump(out, open(path, "w"), indent=1)
print("\nselected ->", out, "\nwritten to", path)
