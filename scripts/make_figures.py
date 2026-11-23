#!/usr/bin/env python3
"""Regenerate every data-driven figure of the revised manuscript at 300 dpi
from the saved runs (Mandatory requirement 3: >= 300 dpi, no stretching,
text >= 8 pt). Figures are sized for the final column width so they are never
rescaled non-proportionally: 3.5 in (single column) or 7.0 in (full width).

  python scripts/make_figures.py --results results --out ../manuscript/figures
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "font.family": "serif", "savefig.dpi": 300})
W1, W2 = 3.5, 7.0
MAIN = [("fedavg_clean", "FedAvg (no attack)"), ("fedavg_signflip", "FedAvg"),
        ("median_signflip", "Median"), ("trimmed_mean_signflip", "Trimmed mean"),
        ("krum_signflip", "Krum"), ("multi_krum_signflip", "Multi-Krum"),
        ("securefl_dpsgd_median", "SecureFL-Median"), ("dpsgd_multi_krum_signflip", "SecureFL-MK")]


def logs(results, group, name):
    out = {}
    for f in sorted(glob.glob(os.path.join(results, group, name, "seed*.json"))):
        L = json.load(open(f))
        z = np.load(f.replace(".json", "_pred.npz"))
        out[L["config"]["seed"]] = (L, z)
    return out


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=300)
    plt.close(fig)
    print("  ", name)


def roc_pr(results, out, seed):
    sel = [("centralized", "Centralized"), ("fedavg_clean", "FedAvg (no attack)"), ("fedavg_signflip", "FedAvg"),
           ("median_signflip", "Median"), ("krum_signflip", "Krum"),
           ("securefl_dpsgd_median", "SecureFL-Median"), ("dpsgd_multi_krum_signflip", "SecureFL-MK")]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.7))
    for n, lab in sel:
        R = logs(results, "main", n)
        if seed not in R:
            continue
        z = R[seed][1]
        fpr, tpr, _ = roc_curve(z["y_test"], z["p_test"])
        pr, rc, _ = precision_recall_curve(z["y_test"], z["p_test"])
        lw = 1.6 if n.startswith(("securefl", "dpsgd")) else 1.0
        axs[0].plot(fpr, tpr, lw=lw, label=lab)
        axs[1].plot(fpr, tpr, lw=lw)
        axs[2].plot(rc, pr, lw=lw)
    axs[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC")
    axs[1].set(xlabel="False positive rate", xlim=(0, 0.01), ylim=(0.0, 1.0), title="(b) ROC, FPR $\\leq$ 0.01")
    axs[1].set_xticks([0, 0.005, 0.01]); axs[1].set_xticklabels(["0", "0.005", "0.01"])
    axs[2].set(xlabel="Recall", ylabel="Precision", title="(c) Precision\u2013recall")
    for a in axs:
        a.grid(alpha=.3)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.5)
    fig.savefig(os.path.join(out, "roc_pr_curves.png"), dpi=300)
    plt.close(fig)
    print("   roc_pr_curves.png")


def threshold_sensitivity(results, out, seed):
    from securefl.metrics import metrics_at
    sel = [("fedavg_signflip", "(a) FedAvg (attacked)"), ("median_signflip", "(b) Median"),
           ("securefl_dpsgd_median", "(c) SecureFL-Median"), ("dpsgd_multi_krum_signflip", "(d) SecureFL-MK")]
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.3), sharey=True)
    ts = np.logspace(-6, 0, 80)
    for a, (n, lab) in zip(axs, sel):
        R = logs(results, "main", n)
        if seed not in R:
            continue
        L, z = R[seed]
        m = [metrics_at(z["y_test"], z["p_test"], t) for t in ts]
        for k, st in [("f1", "-"), ("precision", "--"), ("recall", ":")]:
            a.plot(ts, [x[k] for x in m], st, lw=1.2, label={"f1": "F1-score"}.get(k, k.capitalize()))
        a.axvline(max(L["test"]["best_f1"]["threshold"], 1e-6), color="k", lw=.8, alpha=.7)
        a.axvline(0.5, color="r", lw=.8, alpha=.7)
        a.set(xscale="log", xlim=(1e-6, 1), xlabel="Decision threshold")
        a.set_title(lab, fontsize=9)
        a.grid(alpha=.3)
    axs[0].set_ylabel("Score")
    axs[0].legend(frameon=False, fontsize=7, loc="center left")
    save(fig, out, "threshold_sensitivity.png")


def training_curves(results, out):
    sel = [("fedavg_clean", "FedAvg (no attack)"), ("fedavg_signflip", "FedAvg"), ("median_signflip", "Median"),
           ("multi_krum_signflip", "Multi-Krum"), ("securefl_dpsgd_median", "SecureFL-Median"),
           ("dpsgd_multi_krum_signflip", "SecureFL-MK")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    for n, lab in sel:
        R = logs(results, "main", n)
        if not R:
            continue
        def _pad(k):  # diverged runs stop early: pad with NaN up to the longest run
            rows = [[r[k] for r in L["rounds"]] for L, _ in R.values()]
            T = max(len(x) for x in rows)
            return np.array([x + [np.nan] * (T - len(x)) for x in rows], dtype=float)
        for a, Y in zip(axs, [_pad("val_auc"), _pad("val_f1_default")]):
            x = np.arange(1, Y.shape[1] + 1)
            m = np.nanmean(Y, 0)
            s = np.nanstd(Y, 0, ddof=1) if len(Y) > 1 else 0 * m
            lw = 1.6 if n.startswith(("securefl", "dpsgd")) else 1.0
            a.plot(x, m, lw=lw, label=lab)
            a.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), alpha=.12)
    axs[0].set(xlabel="Communication round", ylabel="Validation AUC-ROC", ylim=(0, 1.02), title="(a) Ranking")
    axs[1].set(xlabel="Communication round", ylabel="Validation F1 at threshold 0.5", ylim=(0, 1.02),
               title="(b) Default threshold")
    for a in axs:
        a.grid(alpha=.3)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=300)
    plt.close(fig)
    print("   training_curves.png")


def _privacy_points(results, group):
    pts = []
    for d in glob.glob(os.path.join(results, group, "*")):
        R = logs(results, group, os.path.basename(d))
        if not R or not R[next(iter(R))][0]["config"]["dp"]:
            continue
        e = [L["epsilon_at_best"] for L, _ in R.values()]
        f = [L["test"]["best_f1"]["f1"] for L, _ in R.values()]
        u = [L["test"]["best_f1"]["auc_roc"] for L, _ in R.values()]
        pts.append((np.mean(e), np.mean(f), np.std(f, ddof=1) if len(f) > 1 else 0, np.mean(u),
                    np.std(u, ddof=1) if len(u) > 1 else 0))
    return np.array(sorted(pts))


def privacy_utility(results, out):
    series = [(g, lab) for g, lab in [("privacy_mk", "SecureFL-MK"), ("privacy", "SecureFL-Median")]
              if len(_privacy_points(results, g))]
    if not series:
        return
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), sharex=True)
    for (g, lab), mk in zip(series, "os"):
        P = _privacy_points(results, g)
        axs[0].errorbar(P[:, 0], P[:, 1], yerr=P[:, 2], marker=mk, ms=3, lw=1.2, capsize=2, label=lab)
        axs[1].errorbar(P[:, 0], P[:, 3], yerr=P[:, 4], marker=mk, ms=3, lw=1.2, capsize=2, label=lab)
    axs[0].set(ylabel="Test F1-score", title="(a) F1-score")
    axs[1].set(ylabel="Test AUC-ROC", title="(b) AUC-ROC")
    for a in axs:
        a.set(xscale="log", xlabel="Privacy budget $\\varepsilon$ ($\\delta=10^{-5}$)")
        a.grid(alpha=.3, which="both")
    axs[0].legend(frameon=False)
    save(fig, out, "privacy_utility_curve.png")


def byzfrac(results, out):
    NAMES = {"fedavg": "FedAvg", "median": "Median (SecureFL-Median)", "trimmed_mean": "Trimmed mean",
             "krum": "Krum", "multi_krum": "Multi-Krum (SecureFL-MK)", "bulyan": "Bulyan"}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
    for a, atk, ttl in zip(axs, ["sign_flip", "alie"], ["(a) Sign-flip", "(b) ALIE"]):
        for agg, mk in zip(NAMES, "osv^Dx"):
            xs, ms, ss = [], [], []
            for f in range(5):
                R = logs(results, "byzfrac", f"{agg}__{'none' if f == 0 else atk}__f{f}")
                if R:
                    v = [L["test"]["best_f1"]["f1"] for L, _ in R.values()]
                    xs.append(10 * f); ms.append(np.mean(v)); ss.append(np.std(v, ddof=1) if len(v) > 1 else 0)
            if xs:
                a.errorbar(xs, ms, yerr=ss, marker=mk, ms=3.5, lw=1.1, capsize=2, label=NAMES[agg])
        xs, ms, ss = [], [], []
        for f in [2, 3]:
            R = logs(results, "byzfrac", f"multi_krum__{atk}__f{f}__fassumed1")
            if R:
                v = [L["test"]["best_f1"]["f1"] for L, _ in R.values()]
                xs.append(10 * f); ms.append(np.mean(v)); ss.append(np.std(v, ddof=1) if len(v) > 1 else 0)
        if xs:
            a.errorbar(xs, ms, yerr=ss, marker="*", ms=6, lw=1.1, ls="--", color="k", capsize=2,
                       label="Multi-Krum, f assumed = 1")
        a.set(xlabel="Byzantine clients (%), M = 10", title=ttl, xticks=[0, 10, 20, 30, 40])
        a.grid(alpha=.3)
    axs[0].set_ylabel("Test F1-score")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(out, "byzantine_fraction_sweep.png"), dpi=300)
    plt.close(fig)
    print("   byzantine_fraction_sweep.png")


def operational(results, out):
    """Fig. operational_comparison.png: F1, missed attacks, FPR (mean +- sd over seeds)."""
    names, F, FN, FP = [], [], [], []
    for n, lab in [("centralized", "Centralized")] + MAIN:
        R = logs(results, "main", n)
        if not R:
            continue
        g = lambda k: [L["test"]["best_f1"][k] for L, _ in R.values()]
        names.append(lab); F.append(g("f1")); FN.append(g("fn")); FP.append(g("fpr"))
    if not names:
        return
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6))
    x = np.arange(len(names))
    for a, V, t in zip(axs, [F, FN, FP], ["(a) F1-score", "(b) Missed attacks (FN)", "(c) False positive rate"]):
        m = [np.mean(v) for v in V]; sd = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in V]
        a.bar(x, m, yerr=sd, capsize=2, color="0.55", edgecolor="k", lw=.5)
        a.set_xticks(x); a.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        a.set_title(t); a.grid(axis="y", alpha=.3)
    save(fig, out, "operational_comparison.png")


def confusion(results, out, seed):
    sel = [("centralized", "Centralized"), ("fedavg_clean", "FedAvg (no attack)"), ("fedavg_signflip", "FedAvg"),
           ("median_signflip", "Median"), ("securefl_dpsgd_median", "SecureFL-Median"),
           ("dpsgd_multi_krum_signflip", "SecureFL-MK")]
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.4))
    axs = axs.ravel()
    for a, (n, lab) in zip(axs, sel):
        R = logs(results, "main", n)
        a.set_title(lab, fontsize=8); a.set_xticks([0, 1]); a.set_yticks([0, 1])
        a.set_xticklabels(["Benign", "Attack"], fontsize=8); a.set_yticklabels(["Benign", "Attack"], fontsize=8)
        if seed not in R:
            continue
        b = R[seed][0]["test"]["best_f1"]
        Mx = np.array([[b["tn"], b["fp"]], [b["fn"], b["tp"]]])
        a.imshow(Mx / Mx.sum(1, keepdims=True), cmap="Greys", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                v = Mx[i, j] / Mx[i].sum()
                a.text(j, i, f"{Mx[i, j]:,}", ha="center", va="center", fontsize=8,
                       color="white" if v > .5 else "black")
    for a in axs:
        a.set_xlabel("Predicted class", fontsize=8); a.set_ylabel("True class", fontsize=8)
    save(fig, out, "confusion_matrices.png")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--seed", type=int, default=42, help="seed used for single-run curves")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    operational(a.results, a.out)
    confusion(a.results, a.out, a.seed)
    roc_pr(a.results, a.out, a.seed)
    threshold_sensitivity(a.results, a.out, a.seed)
    training_curves(a.results, a.out)
    privacy_utility(a.results, a.out)
    byzfrac(a.results, a.out)
