"""
Step 5: hyperparameter tuning via 5-fold walk-forward CV with competition Sharpe.

CV folds (expanding-window walk-forward, 18-day embargo between train end and val start):
  fold 1: train 2010,        val 2011 (from 2011-01-18)
  fold 2: train 2010-2011,   val 2012
  fold 3: train 2010-2012,   val 2013
  fold 4: train 2010-2013,   val 2014
  fold 5: train 2010-2014,   val 2015           <- mirrors step 3/4

LightGBM early stops on MSE (its native loss); we select hyperparameters that
maximize mean-fold competition-Sharpe across the 5 folds.

Output: best_params.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as papq

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))

from features import (
    build_market_features,
    merge_news_features,
    FEATURE_COLS,
    NEWS_FEATURE_COLS,
    TARGET_COL,
)
from metric import competition_score


ROOT = PIPE.parent

FOLDS = [
    # (train_start, train_end, val_start, val_end)
    ("2010-01-01", "2010-12-31", "2011-01-18", "2011-12-31"),
    ("2010-01-01", "2011-12-31", "2012-01-18", "2012-12-31"),
    ("2010-01-01", "2012-12-31", "2013-01-18", "2013-12-31"),
    ("2010-01-01", "2013-12-31", "2014-01-18", "2014-12-31"),
    ("2010-01-01", "2014-12-31", "2015-01-18", "2015-12-31"),
]

FEAT = FEATURE_COLS + NEWS_FEATURE_COLS

N_TRIALS = 30


def prep_data() -> pd.DataFrame:
    print("Preparing data ...", flush=True)
    t0 = time.time()
    market = papq.read_table(ROOT / "market.parquet").to_pandas()
    news_daily = papq.read_table(ROOT / "news_daily.parquet").to_pandas()
    df = build_market_features(market)
    df = merge_news_features(df, news_daily)
    # Keep 2009 forward (for 60-day lookback warmup); drop 2016+ to be sure
    # 2016 stays untouched until step 6.
    df = df[
        (df["time"] >= pd.Timestamp("2009-01-01", tz="UTC"))
        & (df["time"] < pd.Timestamp("2016-01-01", tz="UTC"))
    ].copy()
    n_tgt_nan = int(df[TARGET_COL].isna().sum())
    if n_tgt_nan > 0:
        print(
            f"[WARN] target_nan_in_pool: expected=0, got={n_tgt_nan}, fallback=dropped",
            flush=True,
        )
        df = df.dropna(subset=[TARGET_COL])
    print(f"  pool: {len(df):,} rows  ({time.time() - t0:.1f}s)", flush=True)
    return df


def run_fold(df, params, ts, te, vs, ve):
    train_mask = (
        (df["time"] >= pd.Timestamp(ts, tz="UTC"))
        & (df["time"] <= pd.Timestamp(te, tz="UTC"))
        & (df["universe"] == 1)
    )
    val_mask = (
        (df["time"] >= pd.Timestamp(vs, tz="UTC"))
        & (df["time"] <= pd.Timestamp(ve, tz="UTC"))
        & (df["universe"] == 1)
    )
    train = df.loc[train_mask]
    val = df.loc[val_mask]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[FEAT],
        train[TARGET_COL],
        eval_set=[(val[FEAT], val[TARGET_COL])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    y_pred = model.predict(val[FEAT])
    score = competition_score(
        confidence=y_pred,
        returns=val[TARGET_COL].to_numpy(),
        universe=val["universe"].to_numpy(),
        day=val["time"].dt.date.to_numpy(),
    )
    return score, int(model.best_iteration_)


def objective(trial, df):
    params = {
        "objective": "regression",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    scores, iters = [], []
    for ts, te, vs, ve in FOLDS:
        s, n = run_fold(df, params, ts, te, vs, ve)
        scores.append(s)
        iters.append(n)
    trial.set_user_attr("fold_scores", scores)
    trial.set_user_attr("fold_iters", iters)
    return float(np.mean(scores))


def main() -> int:
    df = prep_data()

    print("\nFolds:")
    for i, (ts, te, vs, ve) in enumerate(FOLDS, 1):
        print(f"  fold {i}: train [{ts}..{te}]  val [{vs}..{ve}]")

    print(f"\nRunning Optuna ({N_TRIALS} trials) ...\n", flush=True)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t_start = time.time()

    def cb(study, trial):
        scores = trial.user_attrs.get("fold_scores", [])
        iters = trial.user_attrs.get("fold_iters", [])
        elapsed = time.time() - t_start
        sstr = ", ".join(f"{s:+.3f}" for s in scores)
        istr = ", ".join(str(i) for i in iters)
        cur_best = study.best_value
        print(
            f"  trial {trial.number:>3d}: "
            f"mean={trial.value:+.4f}  "
            f"best_so_far={cur_best:+.4f}  "
            f"folds=[{sstr}]  iters=[{istr}]  "
            f"elapsed={elapsed / 60:.1f}min",
            flush=True,
        )

    study.optimize(lambda t: objective(t, df), n_trials=N_TRIALS, callbacks=[cb])
    elapsed = time.time() - t_start

    best = study.best_trial
    fold_scores = best.user_attrs["fold_scores"]
    fold_iters = best.user_attrs["fold_iters"]
    fold_mean = float(np.mean(fold_scores))
    fold_std = float(np.std(fold_scores, ddof=1))

    print(f"\nOptuna done in {elapsed / 60:.1f} min")
    print(f"\n=== Best trial #{best.number} ===")
    print(f"  fold-mean Sharpe : {fold_mean:+.4f}  (std across folds = {fold_std:.4f})")
    for i, (s, n) in enumerate(zip(fold_scores, fold_iters), 1):
        print(f"  fold {i}: Sharpe={s:+.4f}  best_iter={n}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    out = {
        "best_params": best.params,
        "fold_mean_sharpe": fold_mean,
        "fold_std_sharpe": fold_std,
        "fold_scores": fold_scores,
        "fold_iters": fold_iters,
        "mean_iter": int(np.mean(fold_iters)),
        "n_trials": N_TRIALS,
        "feature_cols": FEAT,
    }
    out_path = ROOT / "best_params.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path.name}")

    if fold_mean > 0.4:
        print(f"\nStep 5 OK: fold-mean Sharpe {fold_mean:+.4f} > 0.4 "
              f"(std {fold_std:.3f}).")
        return 0
    print(f"\n[WARN] fold-mean Sharpe {fold_mean:+.4f} below 0.4 pass criterion.")
    return 0  # don't hard-fail; user decides


if __name__ == "__main__":
    sys.exit(main())
