#!/usr/bin/env python3
"""Run the revision experiment grid.

Examples
  python scripts/run_suite.py --groups main --data-path data/cicids2017
  python scripts/run_suite.py --groups main attacks byzfrac privacy ablation \
         --data-path data/cicids2017 --path2018 data/cicids2018 --seeds 42 43 44 45 46
  python scripts/run_suite.py --groups main --only securefl --seeds 42 --rounds 2   # smoke test

Runs are resumable: an existing results/<group>/<name>/seed<k>.json is skipped.
Use --shard i/n to split the grid across n GPUs/machines.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.suite import build  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--groups", nargs="+", default=["main"])
p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
p.add_argument("--data-path", default="data/cicids2017")
p.add_argument("--path2018", default="data/cicids2018")
p.add_argument("--max-rows-2018", type=int, default=3_000_000,
               help="stratified sample size for CSE-CIC-IDS2018 (0 = use all ~16M rows)")
p.add_argument("--only", default=None, help="substring filter on run name")
p.add_argument("--rounds", type=int, default=None)
p.add_argument("--out-dir", default="results")
p.add_argument("--shard", default=None, help="i/n")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()

import json  # noqa: E402
HP = os.path.join(os.path.dirname(__file__), "..", "configs", "hparams.json")
hp = json.load(open(HP)) if os.path.exists(HP) else None
if hp and "tune" not in a.groups:
    print("using tuned hyperparameters from configs/hparams.json:", hp)
elif "tune" not in a.groups:
    print("NOTE: configs/hparams.json not found -> paper defaults (lr=1e-3, E=1, C=1.0)")
jobs = []
for c in build(a.groups, a.path2018, a.max_rows_2018 or None):
    if a.only and a.only not in c["name"]:
        continue
    for s in a.seeds:
        j = {"data_path": a.data_path, **c, "seed": s, "out_dir": a.out_dir}
        if hp and c["group"] != "tune":
            for k, v in hp["dp" if c.get("dp", True) else "nodp"].items():
                j.setdefault(k, v)
        if a.rounds:
            j["rounds"] = a.rounds
        jobs.append(j)
if a.shard:
    i, n = map(int, a.shard.split("/"))
    jobs = jobs[i::n]
print(f"{len(jobs)} runs")
if not a.dry_run:
    from securefl.train import run  # noqa: E402  (imports torch)
for j in jobs:
    if a.dry_run:
        print(j["group"], j["name"], j["seed"])
        continue
    try:
        run(j)
    except SystemExit as e:
        print("SKIPPED", j["name"], e)
