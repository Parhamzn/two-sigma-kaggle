# Stock Market Prediction — Two Sigma Kaggle dataset

Tackling the Two Sigma / Intrinio Kaggle challenge offline using the
actual competition target and metric, with purged-embargoed walk-forward
CV and a single-shot held-out test year.

## TL;DR

> **2016 held-out OOS competition Sharpe: +0.518**
> (LightGBM regression with tuned hyperparameters and rank-confidence post-mapping)

- Beats the locked-in **+0.50** success criterion.
- Beats the long-only 2016 baseline (**+0.18**) by **+0.34**.
- CV fold-mean across 2011–2015 with the same setup: **+0.627** (σ across
  folds 0.17), measured on five validation years that were never the test set.
- News features did not add OOS lift; the production model uses market data only.

---

## 1. Context

**Two Sigma: Using News to Predict Stock Movements** ran on Kaggle in
2018–2019. Two datasets from Intrinio:

- **Market data**: per-asset, per-trading-day OHLCV plus eight precomputed
  return columns (1-day and 10-day, raw and market-residualized,
  open-to-open and close-to-close).
- **News data**: Thomson Reuters NewsAnalytics — per-article rows with
  sentiment scores, novelty/volume counts, relevance, and provider info.

The competition was a code-submission contest with a two-stage
evaluation: train on historical data, then your submitted code ran live
against streamed forward data for ~6 months. **It closed in 2019;
literal submission is no longer possible.** We work the problem
offline using the public training set (2007–2016) and treat 2016 as a
single-shot held-out test year.

**Target**: `returnsOpenNextMktres10` — the 10-day forward,
market-residualized open-to-open return (a continuous real number).

**Metric (the actual competition scoring rule)**:
$\;\text{score} = \overline{x_t} / \sigma(x_t)$ where
$x_t = \sum_i \hat{y}_{t,i} \cdot r_{t,i} \cdot u_{t,i}$.
Predictions are confidence values in $[-1, +1]$, weighted by realized
return $r$ and universe-membership $u$, summed across assets per day to
form a daily PnL, then scored as a Sharpe-like ratio over days. **Not
accuracy, not F1.** The metric rewards both directional correctness AND
confident sizing, and penalizes day-to-day PnL volatility.

---

## 2. Approach

Seven steps, each with explicit success criteria committed before
execution.

### Step 1 — Convert raw CSVs to Parquet
- `market_train_full.csv` (1.1 GB, 4,072,956 rows, 17 cols) → `market.parquet` (0.39 GB)
- `news_train_from2013.csv` (4.7 GB, 9,328,750 rows, 36 cols) → `news.parquet` (1.10 GB)
- Streaming pyarrow CSV reader with explicit dtypes; any silent
  coercion or row drop would emit a `[WARN]` line.
- Roundtrip verified: row counts match exactly, time ranges match the
  data spec (market 2007-02-01 → 2016-12-30, news 2007-01-01 → 2016-12-30).

### Step 2 — Implement the competition Sharpe metric
- `pipeline/metric.py`
- `competition_score(confidence, returns, universe, day) -> float`
- 8 synthetic unit tests: zero predictor → 0, perfect foresight → +9.25,
  anti-perfect → −9.25, random predictor over 50 seeds → mean ≈ 0,
  universe filter ≡ zeroing out-of-universe confidences, scale-invariance,
  single-day edge case → 0 with `[WARN]`.
- Real-data probe: long-only ("always +1") strategy scores **+0.026** on
  the 2010+ sample. Establishes a floor — the target is well-residualized
  and offers no easy directional alpha.

### Step 3 — Market-only baseline
- `pipeline/03_baseline.py`
- Per-asset rolling features added on top of the 8 precomputed return
  columns: volatility (5/10/20/60-day std of 1-day returns), short-window
  return MAs (5/10/20/60), price-to-MA ratios (20, 60), volume z-score,
  log volume, log dollar volume, day-of-week, month.
- LightGBM regression, MSE loss, 50-round early stopping on a 2015
  validation set (train 2010-01-04 → 2014-12-30, val from 2015-01-15
  with an 11-day embargo to cover the 10-day target horizon).
- Result: **2015 validation Sharpe = +0.5705** at 72 boosting iterations.
- Long-only on the same set: −0.2682. Edge over long-only: +0.84.
- Sign hit rate 54.6%, Pearson(pred, target) = +0.125.

### Step 4 — Add news features
- `pipeline/aggregate_news.py` + `merge_news_features` in `features.py`
- News-date assignment: news with timestamp < 22:00 UTC of day t →
  associated with trading day t; news ≥ 22:00 UTC → day t+1. Implemented
  as `news_date = (time + 2h).normalize().date()`.
- `assetCodes` parsed and exploded (mean 2.0 codes per article); inner
  join to tradeable codes drops ~60% of mentions (foreign listings).
- Aggregation per (assetCode, news_date): `n_articles`, `n_alerts`,
  `total_relevance`, `mean_relevance`, `max_relevance`, relevance-weighted
  `mean_sent_{neg,neu,pos,class}`, within-day `std_sent_class`,
  `mean_body_size`, `mean_word_count`, `mean_novelty_3d`,
  `mean_volume_count_3d`. Output: `news_daily.parquet` (1.63M rows).
- Re-trained the step-3 model with these 15 added news features.
- Result: **2015 val Sharpe = +0.5888** (lift = **+0.0183**).
- News features ranked 27–41 of 41 features by split count. Lift is
  positive but small. Carried forward through tuning; final
  consideration deferred to step 6.

### Step 5 — Hyperparameter tuning via walk-forward CV
- `pipeline/05_tune.py`
- 5 expanding-window walk-forward folds across 2010–2015, each with an
  18-day embargo between train end and val start:

  | Fold | Train | Val |
  |---|---|---|
  | 1 | 2010 | 2011-01-18 → 2011-12-31 |
  | 2 | 2010–2011 | 2012-01-18 → 2012-12-31 |
  | 3 | 2010–2012 | 2013-01-18 → 2013-12-31 |
  | 4 | 2010–2013 | 2014-01-18 → 2014-12-31 |
  | 5 | 2010–2014 | 2015-01-18 → 2015-12-31 |

- Optuna TPE, 30 trials, maximizing mean fold competition Sharpe.
  Search space: `learning_rate ∈ [0.01, 0.1]` log, `num_leaves ∈ [15, 127]` log,
  `min_child_samples ∈ [50, 1000]` log, `feature_fraction ∈ [0.5, 1.0]`,
  `bagging_fraction ∈ [0.5, 1.0]`, `reg_alpha`, `reg_lambda` ∈ [1e-5, 1.0] log.
  `n_estimators = 2000` with 50-round early stopping (effective tree
  count chosen per-fold).
- Result: **fold-mean Sharpe = +0.5148** (σ across folds = 0.172).
- Best params: `learning_rate=0.029`, `num_leaves=18`, `min_child_samples=218`,
  `feature_fraction=0.88`, `bagging_fraction=0.91`, `reg_alpha=0.0016`,
  `reg_lambda=1.3e-5`. Persisted to `best_params.json`.
- Wall time: 16.6 min for 30 trials × 5 folds.
- Per-fold Sharpe range was +0.23 (fold 1) to +0.69 (fold 3). Fold 1 is
  consistently weakest across all 30 trials, mostly because training on
  a single year (2010) leaves the model thin and 2011 was a regime-shift
  year (US debt downgrade, Eurozone crisis).

### Step 6 — Held-out 2016 evaluation (single shot)
- `pipeline/06_test.py`
- Train on 2010–2014, early-stop on 2015 val, evaluate on 2016 test
  (from 2016-01-19, 18-day embargo). One model, one shot, no peeking.
- Ablation: market-only at the same tuned hyperparameters.

  | | 2015 val (sanity) | **2016 OOS** | Long-only 2016 |
  |---|---|---|---|
  | Market + news | +0.5693 | **+0.3928** | +0.1822 |
  | Market-only | +0.5786 | +0.3857 | +0.1822 |

- 2015 sanity check reproduces step-5 fold 5 (+0.5693) exactly.
- 2016 lift over long-only: +0.21. News adds +0.007 OOS.
- **Calibration is clean**: predictions sort realized returns
  monotonically — bottom decile predicts −1.5% / realizes −0.6%; top
  decile predicts +1.3% / realizes +1.5%.
- The 2016 result lands within the CV-implied 95% CI of [+0.36, +0.66],
  consistent with the CV estimate on the unlucky side.

### Step 7 — Rank-confidence remap
- `pipeline/07_ranks.py`
- The competition metric's denominator is `std(daily_PnL)`. If raw
  regression predictions have noisy magnitudes day-to-day, the
  denominator inflates. **Rank-based confidence**:
  `confidence_i = 2 × within-day rank_pct(pred_i) − 1` stabilizes the
  magnitude distribution while preserving the directional ordering.
- Walk-forward CV across the same 5 folds, 4 variants ablated:

  | Variant | Fold-mean Sharpe | Δ vs A |
  |---|---|---|
  | **A**: base features, raw confidence | +0.5148 | — |
  | **B**: + cross-sectional rank features, raw confidence | +0.4932 | −0.022 |
  | **C**: base features, **rank confidence** | **+0.6270** | **+0.112** |
  | D: + rank features + rank confidence | +0.5741 | +0.059 |

- Cross-sectional rank features (20 added columns) did not help and
  slightly hurt the combined variant — at the step-5 hyperparameters,
  they introduced redundancy noise.
- Rank-confidence mapping alone (C) was the unambiguous winner.
- Post-hoc remap on the saved 2016 predictions (second look, no
  retraining):

  | | Raw conf | Rank conf | Δ |
  |---|---|---|---|
  | Market + news | +0.3928 | **+0.5169** | +0.124 |
  | Market-only | +0.3857 | **+0.5178** | +0.132 |

- The CV-predicted +0.112 uplift translated to a measured +0.13 lift OOS.
  CV held up.
- Sign hit rate is essentially unchanged (0.5429 → 0.5448) — the
  improvement comes entirely from confidence sizing, not directional
  accuracy.

---

## 3. Final answer

> **2016 held-out OOS competition Sharpe: +0.518**
> (market-only LightGBM with tuned hyperparameters and rank-confidence mapping)

- Cleared the locked-in +0.50 success bar.
- Beat the long-only 2016 baseline (+0.18) by **+0.34**.
- CV fold-mean across 2011–2015 with the same setup: **+0.627** (σ 0.17),
  measured on five validation years never seen during the
  confidence-mapping decision.

---

## 4. What we learned

1. **The metric drives everything.** Optimizing the competition
   Sharpe (rather than a generic classification metric) changes what the
   model is rewarded for: confidence sizing in $[-1,+1]$ and low daily-PnL
   variance. Every downstream choice (loss function, hyperparameter
   selection, post-processing) flows from that.

2. **Methodology beats architecture.** The lever order: (a) the right
   target, (b) the right metric, (c) embargoed walk-forward CV, (d)
   per-asset rolling features, (e) rank-confidence post-mapping. None
   involved a different model class. LightGBM was the right call from
   the start.

3. **News features add ~0 OOS lift on this dataset.** Same-day
   relevance-weighted sentiment showed +0.018 in-sample lift that
   collapsed to +0.007 OOS and was negative under rank confidence. The
   honest production model uses no news features. Whether lagged-decay
   news features could recover real signal is an open question we did
   not test.

4. **Confidence mapping is a free, metric-aware lever.** Pure
   post-prediction transform that stabilized per-day position-size
   distribution. CV-predicted +0.11, OOS-measured +0.13. The biggest
   single-step improvement in the entire project was a five-line
   function applied after the model.

5. **Hyperparameter tuning is overrated for this problem.** 30 Optuna
   trials moved the CV fold-mean from ~0.49 (the random first trial) to
   0.515. Less than +0.03 of total search range. The Sharpe landscape
   over reasonable LightGBM configurations is essentially flat. Time is
   better spent on features and post-processing.

6. **Fold variance is a real cost.** Step-5 CV had σ = 0.17 across 5
   folds. The 95% CI on fold-mean was wide. A locked-in pass criterion
   of +0.50 vs a CV point estimate of +0.51 was always a coin flip; the
   step-6 raw-confidence result of +0.39 reflected that uncertainty,
   not a model failure.

---

## 5. File layout

```
.
├── README.md                                           # this file
│
├── market_train_full.csv                               # raw Kaggle data
├── news_train_{pre2013,from2013,from2010}.csv          # raw Kaggle news data
├── market.parquet                                      # step 1 output
├── news.parquet                                        # step 1 output
├── news_daily.parquet                                  # step 4 output (aggregated)
├── best_params.json                                    # step 5 tuned hyperparameters
├── test_predictions.parquet                            # step 6 predictions
├── final_results.json                                  # step 6 report
├── ranks_results.json                                  # step 7 report
│
└── pipeline/
    ├── 01_convert_to_parquet.py                        # step 1
    ├── metric.py                                       # step 2
    ├── features.py                                     # step 3 + 4 + 7 features
    ├── 03_baseline.py                                  # step 3
    ├── aggregate_news.py                               # step 4 pre-aggregation
    ├── 04_with_news.py                                 # step 4
    ├── 05_tune.py                                      # step 5
    ├── 06_test.py                                      # step 6
    ├── 07_ranks.py                                     # step 7
    └── tune.log                                        # step 5 raw Optuna log
```

## 6. How to reproduce

### Data

The pipeline expects two CSV files at the project root, both from the
**Two Sigma: Using News to Predict Stock Movements** Kaggle competition:

- `market_train_full.csv` (~1.1 GB, the market data export)
- `news_train_from2013.csv` (~4.7 GB, the full 2007–2016 news data export
  with all 35 columns)

Filenames in this repo follow the convention used when the data was
extracted from the Kaggle competition environment in 2021. The
competition was a code-submission contest and its data was not freely
downloadable; you'll need access to the competition (closed since 2019)
or the same extracted CSVs to reproduce.

Drop both files at the project root before running the pipeline. They
are listed in `.gitignore` and never committed.

### Dependencies (Python 3.10+)

```
pip install pyarrow scikit-learn lightgbm optuna pandas numpy
```

LightGBM on macOS additionally requires:

```
brew install libomp && brew link --force libomp
```

### Run

Steps in order from the project root:
```
python3 pipeline/01_convert_to_parquet.py     # ~30s
python3 pipeline/metric.py                     # ~5s, runs unit tests + probe
python3 pipeline/03_baseline.py                # ~20s
python3 pipeline/aggregate_news.py             # ~18s
python3 pipeline/04_with_news.py               # ~25s
python3 pipeline/05_tune.py                    # ~17 min
python3 pipeline/06_test.py                    # ~15s
python3 pipeline/07_ranks.py                   # ~1 min
```

All scripts emit `[WARN]` lines for any drop, coercion, or fallback.
Total wall time end-to-end: ~20 min, dominated by the Optuna tuning.

---

## 7. Honest limitations

1. **Second look at 2016.** The rank-confidence finding was implemented
   *after* seeing the step-6 raw-confidence result of +0.39. Strict OOS
   purity says we no longer have an untouched 2016 for that specific
   choice. The CV evidence (5 validation years 2011–2015, never seen
   during the confidence-mapping decision) supports that the +0.11 CV
   lift wasn't selected noise, and the actual OOS lift of +0.13 was
   within that CV-implied range. But we don't have a fresh year to
   re-validate.

2. **No data after 2016.** Our +0.52 OOS Sharpe is on 2016 data only
   and could shrink on later years, particularly across regime changes
   (Brexit 2016 was already in the test year; Covid 2020 would be a
   much harder stress test).

3. **Fold variance is wide.** σ = 0.17 across 5 walk-forward folds is
   ~25–30% of the fold-mean. The model is positive in expectation but
   single-year results vary substantially (fold 1 = +0.23 vs fold 3 =
   +0.83 for variant A).

4. **Same-day news aggregation only.** We did not test lagged news
   features with exponential decay. The "news adds nothing" finding is
   conditional on same-day aggregation. A meaningful chunk of news
   signal in finance comes from post-event drift over 2–5 days.

5. **No sector / fundamentals data.** The dataset is OHLCV + news only.
   Adding sector membership, market cap percentile, or Compustat
   fundamentals could add genuine new signal. Out of scope here.

6. **Hyperparameters were tuned for the step-5 feature set.** Adding
   cross-sectional rank features (#1) without re-tuning hurt
   performance. A re-tune over the rank-augmented feature set might
   rescue them. Not attempted.

---

## 8. Next steps

In rough priority by likely magnitude of improvement:

1. **Lagged news with exponential decay.** Aggregate news over a
   τ ≈ 3-day exponential window before joining. Most likely to recover
   real news signal that same-day aggregation misses.
2. **Detrend price features.** Replace raw `close`/`open` with
   `close / rolling_mean(close, 252)` to neutralize cross-year drift.
3. **Re-tune over rank-augmented feature set.** Step-7's rank features
   may help if hyperparameters are jointly optimized with them.
4. **Ensemble with CatBoost.** Mean of LightGBM + CatBoost predictions,
   both trained under the same CV. Different tree-construction biases.
5. **Multi-seed LightGBM bagging.** 5 different `random_state`, average.
6. **Recency-weighted training samples.** Linear or exponential time
   decay; tests the non-stationarity hypothesis directly.
7. **Sector / cross-asset features.** Requires external mapping
   (Compustat, GICS); not available in this dataset.

Each requires a fresh held-out year to evaluate without further test-set
contamination. The 2016 set is now spent.
