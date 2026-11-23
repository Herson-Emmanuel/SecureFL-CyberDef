#!/usr/bin/env python3
"""Load, clean and cache a dataset once, then print what the paper reports
(record count, features, class balance, split sizes, client partition).
Run this before the grid: it builds the cache every later run reuses.

  python scripts/check_data.py --dataset cicids2017 --path data/cicids2017
  python scripts/check_data.py --dataset cicids2018 --path data/cicids2018 --max-rows 3000000
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from securefl.data import describe_partition, load_dataset, partition_profile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["cicids2017", "cicids2018"])
ap.add_argument("--path", required=True)
ap.add_argument("--max-rows", type=int, default=0)
ap.add_argument("--attempted", default="benign", choices=["benign", "drop", "keep"])
a = ap.parse_args()
t = time.time()
d = load_dataset(a.dataset, a.path, 42, attempted=a.attempted, max_rows=a.max_rows or None)
print(f"\nloaded in {time.time() - t:.0f}s")
print(f"records (cleaned): {d['n_records']:,}   features: {len(d['features'])}")
for k in ["pool", "val", "test"]:
    y = d[f"y_{k}"]
    print(f"{k:>5}: {len(y):,}  benign {int((y == 0).sum()):,}  attack {int(y.sum()):,}")
yp = d["y_pool"]
print(f"pos_weight (benign/attack in pool): {(yp == 0).sum() / yp.sum():.4f}")
print("\nclient partition (M=5):")
for r in describe_partition(yp, partition_profile(yp, 5, 42)):
    print(f"  {r['client']}: {r['samples']:,}  benign {r['benign']:,}  attack {r['attack']:,}  rate {r['attack_rate']:.4f}")
print("\nfeatures:", ", ".join(d["features"]))
