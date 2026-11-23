"""Byzantine behaviours.

Two families:

* Update-level attacks transform the update a Byzantine client would have sent
  (functions below, numpy). They differ in what the attacker must *observe*:

    sign_flip      -lambda*u_i + xi       BLIND: needs only its own honest update
                                           (Eq. (9) of the original paper)
    gaussian       N(0, s^2 I)            BLIND: needs nothing
    model_replace  gamma*(x_mal - w_t)    BLIND: own malicious model only
                                           (Bagdasaryan et al. 2020; gamma = M/eta)
    ipm            -eps*mean(honest)      OMNISCIENT (Xie et al. 2020)
    alie           mean - z*std(honest)   OMNISCIENT (Baruch et al. 2019)

* Data-level attacks change the Byzantine client's local labels before
  training (see ``poison_labels``): ``label_flip`` (all labels inverted) and
  ``targeted_flip`` (attack -> benign, the evasion goal an IDS adversary has).
  The resulting update is submitted *without* any scaling, which makes the
  attack stealthy for distance-based defences.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

BLIND = {"none", "sign_flip", "gaussian", "model_replacement", "label_flip", "targeted_flip"}
OMNISCIENT = {"alie", "ipm"}
DATA_LEVEL = {"label_flip", "targeted_flip", "model_replacement"}


def poison_labels(y, mode, rng=None):
    y = np.asarray(y).copy()
    if mode == "label_flip":
        return 1 - y
    if mode in ("targeted_flip", "model_replacement"):
        y[y == 1] = 0          # attacker teaches the model that attacks are benign
        return y
    return y


def sign_flip(u, lam=5.0, noise_std=0.01, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return -lam * u + rng.normal(0.0, noise_std, size=u.shape)


def gaussian(u, std=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(0.0, std, size=u.shape)


def model_replacement(u_mal, gamma):
    """u_mal = x_mal - w_t is the update of a model trained on poisoned labels.
    Scaling by gamma = M / eta (eta = server learning rate, 1 here) makes the
    FedAvg result approximately equal to x_mal (Bagdasaryan et al. 2020)."""
    return gamma * u_mal


def alie_z(M, f):
    s = int(np.floor(M / 2 + 1)) - f
    return float(norm.ppf((M - s) / M))


def alie(U_honest, M, f, z=None):
    mu = U_honest.mean(axis=0)
    sd = U_honest.std(axis=0)
    z = alie_z(M, f) if z is None else z
    return mu - z * sd


def ipm(U_honest, eps=0.5):
    return -eps * U_honest.mean(axis=0)


def apply_update_attack(attack, U, byz_idx, rng, M, cfg):
    """Return a copy of U (M, d) in which rows ``byz_idx`` are replaced by the
    Byzantine updates. ``U[byz_idx]`` must contain what each Byzantine client
    computed locally (honest training for sign_flip, poisoned training for
    model_replacement / label flips)."""
    U = np.array(U, dtype=np.float64, copy=True)
    byz_idx = list(byz_idx)
    if attack in ("none", "label_flip", "targeted_flip") or not byz_idx:
        return U
    honest = np.array([i for i in range(M) if i not in byz_idx])
    f = len(byz_idx)
    if attack == "sign_flip":
        for i in byz_idx:
            U[i] = sign_flip(U[i], cfg.get("attack_strength", 5.0), cfg.get("attack_noise", 0.01), rng)
    elif attack == "gaussian":
        for i in byz_idx:
            U[i] = gaussian(U[i], cfg.get("gaussian_std", 1.0), rng)
    elif attack == "model_replacement":
        gamma = cfg.get("boost", float(M))
        for i in byz_idx:
            U[i] = model_replacement(U[i], gamma)
    elif attack == "alie":
        v = alie(U[honest], M, f, cfg.get("alie_z"))
        for i in byz_idx:
            U[i] = v
    elif attack == "ipm":
        v = ipm(U[honest], cfg.get("ipm_eps", 0.5))
        for i in byz_idx:
            U[i] = v
    else:
        raise ValueError(f"unknown attack {attack!r}")
    return U
