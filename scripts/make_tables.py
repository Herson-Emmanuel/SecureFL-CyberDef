#!/usr/bin/env python3
"""Aggregate results/*/*/seed*.json into LaTeX row fragments for the manuscript.

The revised manuscript \\input{}s these files (folder ``tables/`` next to the .tex);
when a file is missing the manuscript shows a highlighted [TO BE FILLED] row instead.

  python scripts/make_tables.py --results results --out ../manuscript/tables
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from securefl.stats import holm, paired_tests, summarize  # noqa: E402

LABEL = {
    "centralized": "Centralized baseline",
    "fedavg_clean": "FedAvg (no attack)",
    "fedavg_signflip": "FedAvg",
    "median_signflip": "Coordinate-wise median",
    "trimmed_mean_signflip": "Trimmed mean ($k{=}1$)",
    "krum_signflip": "Krum",
    "multi_krum_signflip": "Multi-Krum ($m{=}M{-}f$)",
    "median_dpcompat_signflip": "Median, DP-compatible net (no DP)",
    "dpsgd_fedavg_signflip": "DP-SGD + FedAvg",
    "dpsgd_trimmed_mean_signflip": "DP-SGD + trimmed mean",
    "dpsgd_krum_signflip": "DP-SGD + Krum",
    "dpsgd_multi_krum_signflip": "\\textbf{SecureFL-MK (DP-SGD + Multi-Krum)}",
    "securefl_dpsgd_median": "\\textbf{SecureFL-Median (DP-SGD + median)}",
}
AGG_LABEL = {"fedavg": "FedAvg", "median": "Median", "trimmed_mean": "Trim.\\ mean", "krum": "Krum",
             "multi_krum": "Multi-Krum", "bulyan": "Bulyan"}
ATK_LABEL = {"sign_flip": "Sign-flip ($\\lambda{=}5$)", "gaussian": "Gaussian noise", "label_flip": "Label flip",
             "targeted_flip": "Targeted flip (attack$\\to$benign)", "model_replacement": "Model replacement",
             "alie": "ALIE (omniscient)", "ipm": "IPM (omniscient)", "none": "No attack"}
REF = "securefl_dpsgd_median"


def load(results):
    runs = defaultdict(lambda: defaultdict(dict))  # group -> name -> seed -> log
    for f in glob.glob(os.path.join(results, "*", "*", "seed*.json")):
        g, n = f.split(os.sep)[-3:-1]
        log = json.load(open(f))
        runs[g][n][log["config"]["seed"]] = log
    return runs


def col(logs, op, key):
    return np.array([logs[s]["test"][op][key] for s in sorted(logs)], dtype=float)


def pm(x, d=4):
    s = summarize(x)
    return f"{s['mean']:.{d}f} $\\pm$ {s['std']:.{d}f}"


def pm_int(x):
    s = summarize(x)
    return f"{s['mean']:,.0f} $\\pm$ {s['std']:,.0f}"


def main_rows(G, op):
    rows = []
    for n, lab in LABEL.items():
        if n not in G:
            continue
        L = G[n]
        eps = [L[s].get("epsilon_at_best") for s in L]
        e = pm(np.array(eps, float), 2) if all(v is not None for v in eps) else "--"
        rows.append(" & ".join([lab, pm(col(L, op, "precision")), pm(col(L, op, "recall")),
                                pm(col(L, op, "f1")), pm(col(L, op, "auc_roc")), pm(col(L, op, "fpr")),
                                pm_int(col(L, op, "fn")), e]) + " \\\\")
    return "\n".join(rows) + "\n"


def pvalue_rows(G, metrics=("f1", "auc_roc", "fpr")):
    if REF not in G:
        return ""
    ref = G[REF]
    names = [n for n in LABEL if n in G and n not in (REF, "centralized", "fedavg_clean")]
    rows = []
    res = {m: [] for m in metrics}
    for n in names:
        common = sorted(set(ref) & set(G[n]))
        for m in metrics:
            a = np.array([ref[s]["test"]["best_f1"][m] for s in common])
            b = np.array([G[n][s]["test"]["best_f1"][m] for s in common])
            res[m].append(paired_tests(a, b))
    for m in metrics:
        adj = holm([r["t_p"] for r in res[m]])
        for r, h in zip(res[m], adj):
            r["t_p_holm"] = h
    for i, n in enumerate(names):
        cells = [LABEL[n]]
        for m in metrics:
            r = res[m][i]
            cells.append(f"{r['mean_diff']:+.4f} & {r['t_p_holm']:.2e} & {r['wilcoxon_p']:.4f}")
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def matrix_rows(G, row_keys, col_keys, name_fn, metric="f1", op="best_f1", row_label=None):
    rows = []
    for rk in row_keys:
        cells = [row_label(rk) if row_label else rk]
        any_ = False
        for ck in col_keys:
            n = name_fn(rk, ck)
            if n in G:
                cells.append(pm(col(G[n], op, metric), 3))
                any_ = True
            else:
                cells.append("n/a")
        if any_:
            rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def privacy_rows(G):
    rows = []
    order = sorted([n for n in G if n.startswith("eps_")], key=lambda s: float(s[4:]))
    order += sorted([n for n in G if n.startswith("sigma_")], key=lambda s: -float(s[6:]))
    order += [n for n in G if n == "nodp_noBN"]
    for n in order:
        L = G[n]
        s0 = sorted(L)[0]
        sigma = L[s0]["config"].get("noise_multiplier") if L[s0]["config"]["dp"] else None
        eps = [L[s].get("epsilon_at_best") for s in L]
        rows.append(" & ".join([
            f"{sigma:.3f}" if sigma is not None else "no DP",
            pm(np.array(eps, float), 2) if sigma is not None else "$\\infty$",
            pm(col(L, "best_f1", "f1")), pm(col(L, "best_f1", "auc_roc")), pm(col(L, "best_f1", "fpr")),
            pm(col(L, "fpr_target", "recall"))]) + " \\\\")
    return "\n".join(rows) + "\n"


def hardware(runs):
    for g in runs.values():
        for n in g.values():
            for log in n.values():
                return json.dumps(log["hardware"], indent=1)
    return "{}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = load(a.results)
    W = lambda fn, s: open(os.path.join(a.out, fn), "w").write(s)
    if "main" in R:
        W("main_bestf1.tex", main_rows(R["main"], "best_f1"))
        W("main_fpr.tex", main_rows(R["main"], "fpr_target"))
        W("main_pvalues.tex", pvalue_rows(R["main"]))
    if "dataset" in R:
        W("dataset2018.tex", main_rows(R["dataset"], "best_f1"))
    if "attacks" in R:
        atks = ["sign_flip", "gaussian", "label_flip", "targeted_flip", "model_replacement", "alie", "ipm"]
        aggs = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum"]
        W("attacks_f1.tex", matrix_rows(R["attacks"], atks, aggs, lambda r, c: f"{c}__{r}",
                                        row_label=lambda r: ATK_LABEL[r]))
        W("attacks_recall_lowfpr.tex", matrix_rows(R["attacks"], atks, aggs, lambda r, c: f"{c}__{r}",
                                                   metric="recall", op="fpr_target",
                                                   row_label=lambda r: ATK_LABEL[r]))
    if "byzfrac" in R:
        aggs = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "bulyan"]
        for atk in ["sign_flip", "alie"]:
            W(f"byzfrac_{atk}.tex", matrix_rows(
                R["byzfrac"], [0, 1, 2, 3, 4], aggs,
                lambda f, c: f"{c}__{'none' if f == 0 else atk}__f{f}",
                row_label=lambda f: f"{f} ({10 * f}\\%)"))
    if "privacy" in R:
        W("privacy.tex", privacy_rows(R["privacy"]))
    if "privacy_mk" in R:
        W("privacy_mk.tex", privacy_rows(R["privacy_mk"]))
    if "byzfrac" in R:
        rows = []
        for atk in ["sign_flip", "alie"]:
            for f in [2, 3]:
                G = R["byzfrac"]
                cells = [ATK_LABEL[atk], f"{f} ({10 * f}\\%)"]
                for n in [f"median__{atk}__f{f}", f"multi_krum__{atk}__f{f}", f"multi_krum__{atk}__f{f}__fassumed1",
                          f"trimmed_mean__{atk}__f{f}", f"trimmed_mean__{atk}__f{f}__fassumed1"]:
                    cells.append(pm(col(G[n], "best_f1", "f1"), 3) if n in G else "n/a")
                rows.append(" & ".join(cells) + " \\\\")
        W("byzfrac_misspec.tex", "\n".join(rows) + "\n")
    if "ablation" in R:
        AB = [("median_noBN_noDP", "Median, no clipping, no noise"),
              ("median_clip_only", "Median, clipping only ($\\sigma{=}0$)"),
              ("median_clip_noise", "Median, clipping + noise ($\\sigma{=}1$) = SecureFL"),
              ("fedavg_clip_only", "FedAvg, clipping only ($\\sigma{=}0$)"),
              ("median_clip_noise_clean", "SecureFL, no attack")]
        rows = []
        for n, lab in AB:
            if n in R["ablation"]:
                L = R["ablation"][n]
                rows.append(" & ".join([lab, pm(col(L, "best_f1", "f1")), pm(col(L, "best_f1", "auc_roc")),
                                        pm(col(L, "best_f1", "fpr")), pm(col(L, "fpr_target", "recall"))]) + " \\\\")
        W("ablation.tex", "\n".join(rows) + "\n")
    W("hardware.json", hardware(R))
    print("wrote", sorted(os.listdir(a.out)))
