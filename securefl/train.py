"""Federated training loop for SecureFL-CyberDef and all baselines.

One call to ``run(cfg)`` = one (configuration, seed) cell of a results table.
It writes  results/<group>/<name>/seed<k>.json   (metrics, eps, timings, hardware)
and        results/<group>/<name>/seed<k>_pred.npz (val/test scores, per-round logs)
"""
from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import accountant as acc
from .aggregators import aggregate, min_clients
from .attacks import DATA_LEVEL, apply_update_attack, poison_labels
from .data import (choose_byzantine, describe_partition, load_dataset,
                   partition_dirichlet, partition_profile)
from .metrics import best_f1_threshold, evaluate, metrics_at
from .models import MLP, flatten_state, load_flat_state, n_params, state_keys

DEFAULTS = dict(
    group="main", name="securefl_dpsgd_median", dataset="cicids2017", data_path="data/cicids2017",
    attempted="benign", max_rows=None,
    mode="federated",            # federated | centralized
    clients=5, partition="profile", dirichlet_alpha=0.5,
    rounds=20, local_epochs=1, batch_size=4096, lr=1e-3, weight_decay=1e-5,
    pos_weight="global",         # 'global' (n_benign/n_attack of the pool, 3.0586) or a float
    batchnorm=None,              # None -> True unless dp
    aggregator="median", trim_f=None, multi_krum_m=None,
    byz_fraction=0.2, byz_select="attack_dominant_first",
    attack="sign_flip", attack_strength=5.0, attack_noise=0.01, boost=None,
    gaussian_std=1.0, alie_z=None, ipm_eps=0.5,
    dp=True, noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5, target_epsilon=None,
    select_by="val_auc",         # checkpoint selection: val_auc | val_f1_cal | last
    fpr_target=1e-3, seed=42, out_dir="results", device="auto", num_workers=0,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def hardware_info():
    info = dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
                cpu=platform.processor() or platform.machine(), cpu_count=os.cpu_count())
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    try:
        import opacus
        info["opacus"] = opacus.__version__
    except ImportError:
        pass
    return info


@torch.no_grad()
def predict(model, X, device, bs=65536):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(torch.as_tensor(X[i:i + bs], device=device))).cpu().numpy())
    return np.concatenate(out)


def _loader(X, y, bs, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g, drop_last=False)


def local_train(model, X, y, cfg, device, seed, engine=None):
    """One client's local training. Returns the trained *plain* nn.Module.

    DP-SGD (Opacus): Poisson sampling with rate q = B/n; for every example the
    per-sample gradient is clipped to L2 norm C; the clipped gradients are
    summed, Gaussian noise N(0, sigma^2 C^2 I) is added to the SUM, the result
    is divided by the expected batch size, and only then passed to Adam
    (DP-Adam). Noise is therefore added to gradients (not to the transmitted
    update); Adam's moment estimates are post-processing of the privatised
    gradients, so the guarantee is unchanged.
    """
    model = model.to(device)
    model.train()                 # the global model was left in eval mode by predict()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["_pos_weight"], device=device))
    loader = _loader(X, y, cfg["batch_size"], True, seed)
    if cfg["dp"]:
        model, opt, loader = engine.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=cfg["noise_multiplier"], max_grad_norm=cfg["max_grad_norm"],
            poisson_sampling=True)
    model.train()
    steps = 0
    for _ in range(cfg["local_epochs"]):
        for xb, yb in loader:
            if len(xb) == 0:          # Poisson sampling can yield an empty batch
                continue
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
            steps += 1
    plain = model._module if hasattr(model, "_module") else model
    return plain, steps


def run(user_cfg):
    cfg = {**DEFAULTS, **user_cfg}
    out_dir = os.path.join(cfg["out_dir"], cfg["group"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"seed{cfg['seed']}")
    if os.path.exists(stem + ".json"):
        print("skip (exists):", stem)
        return json.load(open(stem + ".json"))

    set_seed(cfg["seed"])
    device = torch.device("cuda" if (cfg["device"] == "auto" and torch.cuda.is_available())
                          else ("cpu" if cfg["device"] == "auto" else cfg["device"]))
    d = load_dataset(cfg["dataset"], cfg["data_path"], cfg["seed"], attempted=cfg["attempted"],
                     max_rows=cfg["max_rows"])
    Xp, yp = d["X_pool"], d["y_pool"]
    pw = (yp == 0).sum() / max((yp == 1).sum(), 1) if cfg["pos_weight"] == "global" else float(cfg["pos_weight"])
    cfg["_pos_weight"] = float(pw)
    bn = (not cfg["dp"]) if cfg["batchnorm"] is None else bool(cfg["batchnorm"])
    make_model = lambda: MLP(d_in=Xp.shape[1], batchnorm=bn)
    gmodel = make_model().to(device)
    keys = state_keys(gmodel)
    # attacks act on trainable parameters only (Eq. 13); BatchNorm running statistics are
    # buffers, not gradient-based updates, and are aggregated by the same rule unmodified
    _pn = {n for n, _ in gmodel.named_parameters()}
    pmask = np.concatenate([np.full(gmodel.state_dict()[k].numel(), k in _pn) for k in keys])
    log = dict(config={k: v for k, v in cfg.items() if not k.startswith("_")}, hardware=hardware_info(),
               n_params=n_params(gmodel), n_records=d["n_records"], n_features=int(Xp.shape[1]),
               pos_weight=cfg["_pos_weight"], rounds=[])

    t_start = time.time()
    if cfg["mode"] == "centralized":
        best, best_score = None, -1
        for r in range(cfg["rounds"]):
            gmodel, _ = local_train(gmodel, Xp, yp, {**cfg, "dp": False}, device, cfg["seed"] + r)
            pv = predict(gmodel, d["X_val"], device)
            m = metrics_at(d["y_val"], pv, 0.5)
            log["rounds"].append(dict(round=r + 1, val_auc=m["auc_roc"], val_f1_default=m["f1"]))
            if m["auc_roc"] > best_score:
                best_score, best = m["auc_roc"], copy.deepcopy(gmodel.state_dict())
        gmodel.load_state_dict(best)
    else:
        M = cfg["clients"]
        parts = (partition_profile(yp, M, cfg["seed"]) if cfg["partition"] == "profile"
                 else partition_dirichlet(yp, M, cfg["seed"], cfg["dirichlet_alpha"]))
        log["partition"] = describe_partition(yp, parts)
        f = int(round(cfg["byz_fraction"] * M)) if cfg["attack"] != "none" else 0
        byz = choose_byzantine(parts, yp, f, cfg["byz_select"], cfg["seed"])
        log["byzantine_clients"] = byz
        f_assumed = cfg["trim_f"] if cfg["trim_f"] is not None else max(f, 1)
        if M < min_clients(cfg["aggregator"], f_assumed):
            raise SystemExit(f"{cfg['aggregator']} needs >= {min_clients(cfg['aggregator'], f_assumed)} "
                             f"clients for f={f_assumed}; got M={M}")
        sizes = np.array([len(p) for p in parts], dtype=float)

        # DP: one persistent PrivacyEngine (and RDP accountant) per client
        engines = [None] * M
        if cfg["dp"]:
            from opacus import PrivacyEngine
            if cfg["target_epsilon"] is not None:
                q = cfg["batch_size"] / sizes.min()
                steps = cfg["rounds"] * cfg["local_epochs"] * math.ceil(1 / q)
                cfg["noise_multiplier"] = acc.sigma_for_epsilon(cfg["target_epsilon"], q, steps, cfg["delta"])
                log["config"]["noise_multiplier"] = cfg["noise_multiplier"]
            engines = [PrivacyEngine(accountant="rdp") for _ in range(M)]
        steps_done = np.zeros(M, dtype=int)

        # round 0 = the initial global model, a valid fallback if training diverges at once
        pv0 = predict(gmodel, d["X_val"], device)
        m0 = metrics_at(d["y_val"], pv0, 0.5)
        best, best_round = flatten_state(gmodel, keys), 0
        best_score = m0["auc_roc"] if np.isfinite(m0["auc_roc"]) else -np.inf
        log["diverged_round"] = None
        for r in range(cfg["rounds"]):
            t_r = time.time()
            w = flatten_state(gmodel, keys)
            U = np.zeros((M, len(w)))
            for i, idx in enumerate(parts):
                yi = yp[idx]
                if i in byz and cfg["attack"] in DATA_LEVEL:
                    yi = poison_labels(yi, cfg["attack"])
                local = copy.deepcopy(gmodel)
                local, st = local_train(local, Xp[idx], yi, cfg, device,
                                        cfg["seed"] * 1000 + r * 31 + i, engines[i])
                steps_done[i] += st
                U[i] = flatten_state(local, keys) - w
            if byz:
                U[:, pmask] = apply_update_attack(cfg["attack"], U[:, pmask], byz,
                                                  np.random.default_rng(cfg["seed"] + r), M,
                                                  {**cfg, "boost": cfg["boost"] or float(M)})
            t_a = time.perf_counter()
            agg = aggregate(cfg["aggregator"], U, f=f_assumed, weights=sizes, m=cfg["multi_krum_m"])
            agg_ms = (time.perf_counter() - t_a) * 1e3
            load_flat_state(gmodel, w + agg, keys)
            pv = predict(gmodel, d["X_val"], device)
            if not (np.all(np.isfinite(w + agg)) and np.all(np.isfinite(pv))):
                # weights/scores overflowed (e.g. FedAvg under sign-flip): the model has diverged and
                # cannot recover; keep the best validation checkpoint from earlier rounds and stop.
                log["diverged_round"] = r + 1
                log["rounds"].append(dict(round=r + 1, val_auc=float("nan"), val_f1_default=float("nan"),
                                          agg_ms=(time.perf_counter() - t_a) * 1e3,
                                          round_s=time.time() - t_r, diverged=True))
                print(f"[{cfg['name']} s{cfg['seed']}] DIVERGED at round {r+1}; "
                      f"using best checkpoint (round {best_round})", flush=True)
                break
            m = metrics_at(d["y_val"], pv, 0.5)
            rec = dict(round=r + 1, val_auc=m["auc_roc"], val_f1_default=m["f1"], agg_ms=agg_ms,
                       round_s=time.time() - t_r)
            if cfg["select_by"] == "val_f1_cal":
                rec["val_f1_cal"] = metrics_at(d["y_val"], pv, best_f1_threshold(d["y_val"], pv))["f1"]
            if cfg["dp"]:
                eps_opacus = [engines[i].get_epsilon(cfg["delta"]) for i in range(M) if i not in byz]
                q = cfg["batch_size"] / sizes
                eps_ours = [acc.epsilon(q[i], cfg["noise_multiplier"], int(steps_done[i]), cfg["delta"])[0]
                            for i in range(M) if i not in byz]
                rec.update(eps_max=float(max(eps_opacus)), eps_ours_max=float(max(eps_ours)))
            log["rounds"].append(rec)
            score = {"val_auc": m["auc_roc"], "val_f1_cal": rec.get("val_f1_cal", 0), "last": r}[cfg["select_by"]]
            if score > best_score:
                best_score, best, best_round = score, w + agg, r + 1
            print(f"[{cfg['name']} s{cfg['seed']}] round {r+1}/{cfg['rounds']} val_auc={m['auc_roc']:.5f}"
                  + (f" eps={rec['eps_max']:.3f}" if cfg["dp"] else ""), flush=True)
        load_flat_state(gmodel, best, keys)
        log["best_round"] = best_round
        if cfg["dp"]:
            br = log["rounds"][best_round - 1] if best_round > 0 else {}
            log["epsilon_at_best"] = br.get("eps_max", 0.0)
            log["epsilon_ours_at_best"] = br.get("eps_ours_max", 0.0)
            log["accountant"] = ("RDP (Mironov et al. 2017/2019) via Opacus RDPAccountant; Poisson sampling "
                                 "q=B/n_i; per-client record-level (epsilon, delta); eps = max over honest clients")

    log["train_seconds"] = time.time() - t_start
    pv = predict(gmodel, d["X_val"], device)
    pt = predict(gmodel, d["X_test"], device)
    log["test"] = evaluate(d["y_val"], pv, d["y_test"], pt, cfg["fpr_target"])
    json.dump(log, open(stem + ".json", "w"), indent=1, default=float)
    np.savez_compressed(stem + "_pred.npz", y_val=d["y_val"].astype(np.int8), p_val=pv.astype(np.float32),
                        y_test=d["y_test"].astype(np.int8), p_test=pt.astype(np.float32),
                        cat_test=d["cat_test"])
    b = log["test"]["best_f1"]
    print(f"DONE {cfg['group']}/{cfg['name']} seed={cfg['seed']} F1={b['f1']:.6f} AUC={b['auc_roc']:.6f} "
          f"FPR={b['fpr']:.6f} FN={b['fn']}")
    return log
