# SecureFL-CyberDef — revision experiments

Code for every new experiment requested in the review of manuscript PAPER024139-1.
Running it produces LaTeX table rows (`tables/*.tex`) and 300-dpi figures
(`figures/*.png`). Copy those two folders next to `SecureFL-CyberDef_revised.tex`
and recompile: the `[TO BE FILLED]` rows in the manuscript are replaced automatically.

> **Status.** The numpy parts (aggregators, attacks, statistics, RDP accountant,
> partitioning, metrics) are unit-tested (`python tests/test_core.py`). The PyTorch/Opacus
> training loop (`securefl/train.py`) was written to match the paper's description
> but could not be executed where it was written (no GPU, no dataset). **Run the smoke
> test below first** and compare with your original code before launching the full grid.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, opacus, numpy, scipy, scikit-learn, pandas, matplotlib
python tests/test_core.py                # all lines should print "ok"
```

## 2. Data

| Dataset | Put CSV files in | Source |
|---|---|---|
| CICIDS2017 (improved, Engelen et al. 2021) | `data/cicids2017/` | https://intrusion-detection.distrinet-research.be/WTMC2021/ |
| CSE-CIC-IDS2018 (processed CSVs) | `data/cicids2018/` | https://www.unb.ca/cic/datasets/ids-2018.html |

`attempted="benign"` (default) maps the "- Attempted" labels of the improved release to
benign. **Check this matches your original pipeline** — your cleaned set had 2,099,971
records and 84 features; the first run prints `n_records` and `n_features` into the JSON.

### Check and cache each dataset once (do this first)

```bash
python scripts/check_data.py --dataset cicids2017 --path data/cicids2017
python scripts/check_data.py --dataset cicids2018 --path data/cicids2018 --max-rows 3000000
```

The loader reads CSVs in chunks, so even the full CSE-CIC-IDS2018 (~16 M rows) fits in
32 GB RAM. By default the `dataset` group uses a class-stratified 3,000,000-row sample
(`--max-rows-2018`, same benign/attack ratio as the full data) so that a run costs about
as much as on CICIDS2017; pass `--max-rows-2018 0` to use everything. Report the size you
used in Section 4.1.

## 3. Smoke test (≈ minutes)

```bash
python scripts/run_suite.py --groups main --only securefl --seeds 42 --rounds 2
```

## 4. Full grid

Each run = one (configuration, seed). Results are resumable and can be sharded
across GPUs with `--shard i/n`.

| Priority | Group | Runs (×5 seeds) | Fills | Reviewer item |
|---|---|---|---|---|
| 1 | `main` | 13 × 5 = 65 | Tables 4, 5, p-value table, Figs. ROC/PR, threshold, training | Major 1, 2 |
| 1 | `privacy` | 12 × 5 = 60 | Privacy–utility table + figure | Major 5 |
| 1 | `attacks` | 35 × 5 = 175 | Attack × aggregator table | Major 3, Minor 19 |
| 2 | `dataset` | 8 × 5 = 40 | CSE-CIC-IDS2018 table | Major 6 |
| 2 | `ablation` | 5 × 5 = 25 | Mechanism ablation table | Minor 2 |
| 3 | `byzfrac` | 44 × 5 = 220 | Byzantine-fraction figure (M = 10) | Major 1 (high Byzantine share) |

```bash
python scripts/run_suite.py --groups main privacy attacks --data-path data/cicids2017
python scripts/run_suite.py --groups dataset --path2018 data/cicids2018
python scripts/run_suite.py --groups ablation byzfrac --data-path data/cicids2017
python scripts/bench_aggregators.py --out tables --fig figures
python scripts/make_tables.py  --results results --out tables
python scripts/make_figures.py --results results --out figures
```

Seeds 42–46 are fixed; the train/test split is fixed (split seed 0) so seeds vary
initialisation, data order, client sampling and DP noise. Set `SECUREFL_RESPLIT=1`
to re-split per seed as well.

## 5. What each attack assumes (Minor 17)

| Attack | Knowledge needed by the Byzantine client |
|---|---|
| sign-flip (paper Eq. 9) | only its **own** honest update → blind / non-omniscient |
| Gaussian | none |
| label flip, targeted flip | only its own data (data poisoning) |
| model replacement (γ = M) | its own poisoned model (boosted, Bagdasaryan et al. 2020) |
| ALIE, IPM | mean / std of **honest** updates → omniscient, colluding (upper bound) |

## 6. Privacy accounting (Major 4, Minor 15)

* DP-SGD via Opacus `PrivacyEngine(accountant="rdp")`, Poisson sampling q = B/nᵢ,
  per-sample clipping C, Gaussian noise σC added to the **sum of clipped per-example
  gradients**, then divided by the expected batch size and passed to **Adam**
  (DP-Adam; the Adam moments are post-processing, so the guarantee is unchanged).
* ε is record-level, per client, w.r.t. that client's local data; we report the max
  over honest clients. `securefl/accountant.py` re-computes it independently.
* **Check before resubmitting:** with σ = 1.0, C = 1.0, B = 4096, nᵢ ≈ 302,396 and
  19 rounds × 1 local epoch, the RDP accountant gives **ε ≈ 3.35** (δ = 1e-5), not
  26.75. The value 26.75 corresponds to, e.g., ≈ 10,000 samples per client at
  B = 4096. Please verify which setting your original run used.

```bash
python -c "from securefl.accountant import epsilon; print(epsilon(4096/302396, 1.0, 19*74, 1e-5))"
```

## 7. Statistics (Major 2)

`securefl/stats.py`: mean ± sd (ddof=1), 95 % t-interval, paired t-test with Holm
correction, exact Wilcoxon signed-rank and Mann–Whitney U. With 5 seeds the exact
two-sided Wilcoxon signed-rank p-value cannot go below 0.0625; the paired t-test (and
Mann–Whitney, min p = 0.0079) are therefore the primary tests. Use 10 seeds if you want
Wilcoxon to be able to reach p < 0.05.
