"""Experiment grid for the revision. Each entry -> one row of a manuscript table.

Group        Manuscript item                      Reviewer item
-----------  -----------------------------------  -------------------------------
main         Tables 4 and 5 (+ p-values)          Major 1, Major 2
attacks      Table 6 (attack x aggregator)        Major 3, Minor 17, Minor 19
byzfrac      Fig. "Byzantine fraction" + Table 7  Major 1 (high Byzantine share)
privacy      Table 8 + privacy-utility figure     Major 5
dataset      Table 9 (CSE-CIC-IDS2018)            Major 6
ablation     Table 10 (why DP-SGD helps median)   Minor 2
"""

ROBUST = ["median", "trimmed_mean", "krum", "multi_krum"]
ALL_AGG = ["fedavg"] + ROBUST


def main_group():
    g = [
        dict(name="centralized", mode="centralized", dp=False, attack="none"),
        dict(name="fedavg_clean", aggregator="fedavg", dp=False, attack="none"),
        dict(name="fedavg_signflip", aggregator="fedavg", dp=False),
        dict(name="median_signflip", aggregator="median", dp=False),
        dict(name="trimmed_mean_signflip", aggregator="trimmed_mean", dp=False),
        dict(name="krum_signflip", aggregator="krum", dp=False),
        dict(name="multi_krum_signflip", aggregator="multi_krum", dp=False),
        dict(name="median_dpcompat_signflip", aggregator="median", dp=False, batchnorm=False),
        dict(name="dpsgd_fedavg_signflip", aggregator="fedavg", dp=True),
        dict(name="dpsgd_trimmed_mean_signflip", aggregator="trimmed_mean", dp=True),
        dict(name="dpsgd_krum_signflip", aggregator="krum", dp=True),
        dict(name="dpsgd_multi_krum_signflip", aggregator="multi_krum", dp=True),
        dict(name="securefl_dpsgd_median", aggregator="median", dp=True),   # proposed
    ]
    return [dict(group="main", **c) for c in g]


def attacks_group():
    out = []
    for atk in ["sign_flip", "gaussian", "label_flip", "targeted_flip", "model_replacement", "alie", "ipm"]:
        for agg in ALL_AGG:
            out.append(dict(group="attacks", name=f"{agg}__{atk}", aggregator=agg, attack=atk, dp=True))
    return out


def byzfrac_group():
    out = []
    for f in [0, 1, 2, 3, 4]:
        for atk in ["sign_flip", "alie"]:
            if f == 0 and atk == "alie":
                continue
            for agg in ALL_AGG + ["bulyan"]:
                need = {"krum": 2 * max(f, 1) + 3, "multi_krum": 2 * max(f, 1) + 3,
                        "bulyan": 4 * max(f, 1) + 3, "trimmed_mean": 2 * max(f, 1) + 1}.get(agg, 1)
                if need > 10:
                    continue
                out.append(dict(group="byzfrac", name=f"{agg}__{atk if f else 'none'}__f{f}", clients=10,
                                aggregator=agg, attack=atk if f else "none", byz_fraction=f / 10,
                                trim_f=max(f, 1), dp=True))
    return out


def privacy_group():
    out = [dict(group="privacy", name="nodp_noBN", aggregator="median", dp=False, batchnorm=False)]
    for eps in [0.5, 1, 2, 5, 10, 25, 50]:
        out.append(dict(group="privacy", name=f"eps_{eps}", aggregator="median", dp=True, target_epsilon=eps))
    for s in [0.5, 1.0, 1.5, 2.0]:          # also report raw sigma values incl. the paper's sigma = 1.0
        out.append(dict(group="privacy", name=f"sigma_{s}", aggregator="median", dp=True, noise_multiplier=s))
    return out


def dataset_group(path2018, max_rows=None):
    base = [c for c in main_group() if c["name"] in {
        "centralized", "fedavg_clean", "fedavg_signflip", "median_signflip", "trimmed_mean_signflip",
        "krum_signflip", "multi_krum_signflip", "securefl_dpsgd_median"}]
    return [{**c, "group": "dataset", "dataset": "cicids2018", "data_path": path2018, "max_rows": max_rows}
            for c in base]


def ablation_group():
    return [
        dict(group="ablation", name="median_noBN_noDP", aggregator="median", dp=False, batchnorm=False),
        dict(group="ablation", name="median_clip_only", aggregator="median", dp=True, noise_multiplier=0.0),
        dict(group="ablation", name="median_clip_noise", aggregator="median", dp=True),
        dict(group="ablation", name="fedavg_clip_only", aggregator="fedavg", dp=True, noise_multiplier=0.0),
        dict(group="ablation", name="median_clip_noise_clean", aggregator="median", dp=True, attack="none"),
    ]


def tune_group():
    """Validation-only tuning on attack-free FedAvg (so no method, and no attack, is
    tuned for). Non-DP: learning rate x local epochs. DP: learning rate x clipping norm
    with one local epoch, so the privacy budget stays fixed."""
    out = []
    for lr in [5e-4, 1e-3, 3e-3]:
        for E in [1, 2]:
            out.append(dict(group="tune", name=f"nodp_lr{lr:g}_E{E}", aggregator="fedavg", attack="none",
                            dp=False, lr=lr, local_epochs=E))
        for C in [0.5, 1.0, 2.0]:
            out.append(dict(group="tune", name=f"dp_lr{lr:g}_C{C:g}", aggregator="fedavg", attack="none",
                            dp=True, lr=lr, max_grad_norm=C))
    return out


def extra_group(path2018, max_rows=None):
    """Additions for the pluggable-aggregator framing (DP-SGD + Multi-Krum as the
    recommended configuration):
      * CSE-CIC-IDS2018 with DP-SGD + Krum / Multi-Krum / trimmed mean
      * privacy-utility sweep for DP-SGD + Multi-Krum
      * misspecified f: Multi-Krum and trimmed mean told f=1 while the true f is 2 or 3
    """
    out = []
    for agg in ["multi_krum", "krum", "trimmed_mean"]:
        out.append(dict(group="dataset", name=f"dpsgd_{agg}_signflip", aggregator=agg, dp=True,
                        dataset="cicids2018", data_path=path2018, max_rows=max_rows))
    for eps in [0.5, 1, 2, 5, 10, 25, 50]:
        out.append(dict(group="privacy_mk", name=f"eps_{eps}", aggregator="multi_krum", dp=True, target_epsilon=eps))
    for s_ in [0.5, 1.0, 1.5, 2.0]:
        out.append(dict(group="privacy_mk", name=f"sigma_{s_}", aggregator="multi_krum", dp=True, noise_multiplier=s_))
    for f in [2, 3]:
        for atk in ["sign_flip", "alie"]:
            for agg in ["multi_krum", "trimmed_mean"]:
                out.append(dict(group="byzfrac", name=f"{agg}__{atk}__f{f}__fassumed1", clients=10, aggregator=agg,
                                attack=atk, byz_fraction=f / 10, trim_f=1, dp=True))
    return out


def build(groups, path2018="data/cicids2018", max_rows_2018=None):
    table = {"tune": tune_group, "main": main_group, "attacks": attacks_group, "byzfrac": byzfrac_group,
             "privacy": privacy_group, "ablation": ablation_group,
             "dataset": lambda: dataset_group(path2018, max_rows_2018),
             "extra": lambda: extra_group(path2018, max_rows_2018)}
    out = []
    for g in groups:
        out += table[g]()
    return out
