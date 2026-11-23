"""Dataset loading, preprocessing and non-IID partitioning.

Supported datasets
  cicids2017   -- CICIDS2017 (the "improved" re-labelled release of Engelen et al.
                  2021 or the original MachineLearningCSV files)
  cicids2018   -- CSE-CIC-IDS2018 (Processed Traffic Data for ML Algorithms CSVs)

The pipeline mirrors Section IV-B of the manuscript:
  drop identifier columns -> coerce numeric -> drop NaN/inf rows -> binary label
  -> stratified 80/20 train/test -> 10% of train held out for validation
  -> z-score with statistics from the federated training pool only.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ID_COLUMNS = {
    "id", "flow id", "flow_id", "src ip", "source ip", "src_ip", "dst ip", "destination ip",
    "dst_ip", "timestamp", "src port", "source port", "attempted category", "label", "class",
}

BENIGN_NAMES = {"benign", "normal"}


def _norm(c):
    return str(c).strip().lower()


def _count_rows(f):
    n = 0
    with open(f, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            n += block.count(b"\n")
    return max(n - 1, 0)


def load_clean(path, attempted="benign", max_rows=None, seed=0, chunksize=500_000, verbose=True):
    """Memory-bounded CSV loader for CICIDS2017 and CSE-CIC-IDS2018.

    Reads each file in chunks, keeps only the columns shared by all files (some
    CSE-CIC-IDS2018 days carry extra Flow ID / Src IP / Dst IP / Src Port columns),
    drops repeated header rows, converts features to float32 chunk by chunk and,
    if ``max_rows`` is set, keeps a class-stratified random fraction of every chunk
    (same benign/attack ratio as the full data). Peak RAM is about one chunk plus
    the kept float32 rows, so the full 16 M-row CSE-CIC-IDS2018 fits in 32 GB.
    """
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no CSV files under {path}")
    heads = [[str(c).strip() for c in pd.read_csv(f, nrows=0, encoding="latin-1").columns] for f in files]
    common = set.intersection(*[set(h) for h in heads])
    order = [c for c in heads[0] if c in common]
    label_col = next(c for c in order if _norm(c) in ("label", "class"))
    feats = [c for c in order if _norm(c) not in ID_COLUMNS]
    frac = 1.0
    if max_rows:
        total = sum(_count_rows(f) for f in files)
        frac = min(1.0, max_rows / total)
        if verbose:
            print(f"[data] {total:,} rows in {len(files)} files; keeping fraction {frac:.4f}")
    rng = np.random.default_rng(seed)
    Xs, labs = [], []
    for f, h in zip(files, heads):
        reader = pd.read_csv(f, usecols=lambda c: str(c).strip() in common, chunksize=chunksize,
                             dtype=str, encoding="latin-1")
        n_f = 0
        for ch in reader:
            ch.columns = [str(c).strip() for c in ch.columns]
            lab = ch[label_col].astype(str).str.strip()
            ok = lab.str.lower() != "label"                      # repeated header rows (2018)
            ch, lab = ch[ok], lab[ok]
            if frac < 1.0:                                       # stratified per chunk
                is_ben = lab.str.lower().isin(BENIGN_NAMES).values
                keep = np.zeros(len(lab), bool)
                for m in (is_ben, ~is_ben):
                    idx = np.where(m)[0]
                    k = int(round(frac * len(idx)))
                    if k:
                        keep[rng.choice(idx, k, replace=False)] = True
                ch, lab = ch[keep], lab[keep]
            X = np.array(ch[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64), copy=True)
            X[~np.isfinite(X)] = np.nan
            good = ~np.isnan(X).any(axis=1)
            Xs.append(X[good].astype(np.float32))
            labs.append(lab.values[good].astype(str))
            n_f += int(good.sum())
        if verbose:
            print(f"[data] {os.path.basename(f)}: {n_f:,} rows kept")
    X = np.concatenate(Xs)
    lab = pd.Series(np.concatenate(labs))
    if attempted != "keep":
        att = lab.str.contains("attempted", case=False).values
        if attempted == "drop":
            X, lab = X[~att], lab[~att].reset_index(drop=True)
        elif attempted == "benign":
            lab = lab.where(~att, "BENIGN")
    y = (~lab.str.lower().isin(BENIGN_NAMES)).astype(np.int64).values
    if verbose:
        print(f"[data] total {len(y):,} records, {X.shape[1]} features, attack share {y.mean():.4f}")
    return dict(X=X, y=y, features=feats, cat=lab.values.astype(str))


def split(X, y, seed, test_size=0.2, val_size=0.1, cat=None):
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    ifl, iva = train_test_split(itr, test_size=val_size, stratify=y[itr], random_state=seed)
    Xfl, Xva, Xte = X[ifl], X[iva], X[ite]
    yfl, yva, yte = y[ifl], y[iva], y[ite]
    mu = Xfl.mean(axis=0)
    sd = Xfl.std(axis=0)
    sd[sd == 0] = 1.0
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    return dict(X_pool=z(Xfl), y_pool=yfl, X_val=z(Xva), y_val=yva, X_test=z(Xte), y_test=yte,
                mu=mu, sd=sd, cat_test=None if cat is None else cat[ite])


def load_dataset(name, path, seed, cache_dir="cache", attempted="benign", max_rows=None):
    """Load + clean + split, caching the cleaned arrays (the split depends on seed).

    By default the train/test split is FIXED across seeds (split_seed=0) so that the
    seeds only vary model initialisation, client sampling order and DP noise --
    the usual protocol for seed-variance reporting. Set env SECUREFL_RESPLIT=1 to
    re-split per seed instead.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(json.dumps([name, os.path.abspath(path), attempted, max_rows]).encode()).hexdigest()[:10]
    cpath = os.path.join(cache_dir, f"{name}_{key}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        X, y, feats, cat = z["X"], z["y"], list(z["features"]), z["cat"]
    else:
        c = load_clean(path, attempted=attempted, max_rows=max_rows)
        X, y, feats, cat = c["X"], c["y"], c["features"], c["cat"]
        np.savez_compressed(cpath, X=X, y=y, features=np.array(feats, dtype=object), cat=cat)
    split_seed = seed if os.environ.get("SECUREFL_RESPLIT") == "1" else 0
    d = split(X, y, split_seed, cat=cat)
    d["features"] = feats
    d["n_records"] = int(len(y))
    return d


# ----------------------------------------------------------------------------- partitions
PROFILE_5 = [0.0, 0.232, 0.75, 0.25, 0.0]   # Table 2 of the original submission (attack share 0.246)


def scaled_profile(attack_share):
    """Client attack rates (0, a, 3a, a, 0) for dataset attack share a.

    Keeps the structure of the original Table 2 on any dataset: two benign-only
    clients, one attack-dominant client (3x the average) and two mixed clients at
    the average rate. With a = 0.246 this is (0, .246, .739, .246, 0), i.e. the
    original profile. Requires a <= 1/3 (otherwise the dominant rate is capped at 1).
    """
    a = float(attack_share)
    return [0.0, a, min(1.0, 3 * a), a, 0.0]


def partition_profile(y, M, seed, rates=None):
    """Equal-size clients whose attack rates follow ``rates`` (cycled if M > 5).

    Default rates = scaled_profile(mean(y)), so every dataset gets two benign-only
    clients, one attack-dominant client and two mixed clients. Attack samples are
    assigned to the highest-rate clients first; a client whose target cannot be met
    gets what is left.
    """
    rng = np.random.default_rng(seed)
    rates = rates or scaled_profile(np.mean(y))
    rates = [rates[i % len(rates)] for i in range(M)]
    ben = rng.permutation(np.where(y == 0)[0])
    att = rng.permutation(np.where(y == 1)[0])
    n = len(y) // M
    n_att = [0] * M
    pool = len(att)
    for i in sorted(range(M), key=lambda k: -rates[k]):
        take = min(int(round(rates[i] * n)), pool)
        n_att[i] = take
        pool -= take
    idx, a0, b0 = [], 0, 0
    for i in range(M):
        na = n_att[i]
        nb = n - na
        part = np.concatenate([att[a0:a0 + na], ben[b0:b0 + nb]])
        a0 += na
        b0 += nb
        idx.append(rng.permutation(part))
    return idx


def partition_dirichlet(y, M, seed, alpha=0.5):
    rng = np.random.default_rng(seed)
    idx = [[] for _ in range(M)]
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet(alpha * np.ones(M))
        cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
        for i, part in enumerate(np.split(ids, cuts)):
            idx[i].extend(part.tolist())
    return [rng.permutation(np.array(i, dtype=int)) for i in idx]


def describe_partition(y, parts):
    rows = []
    for i, p in enumerate(parts):
        yi = y[p]
        rows.append(dict(client=f"C{i}", samples=int(len(p)), benign=int((yi == 0).sum()),
                         attack=int((yi == 1).sum()), attack_rate=float(yi.mean()) if len(p) else 0.0))
    return rows


def choose_byzantine(parts, y, f, mode="attack_dominant_first", seed=0):
    if f == 0:
        return []
    rates = [y[p].mean() for p in parts]
    if mode == "attack_dominant_first":
        return [int(i) for i in np.argsort(rates)[::-1][:f]]
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(len(parts), f, replace=False))
