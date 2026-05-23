"""
Step 3 baseline: market-only features -> LightGBM regression -> 2015 validation.

- Train: 2010-01-01 .. 2014-12-31, universe == 1
- Embargo: 11 trading days (target horizon is 10 days)
- Val:   2015-01-15 .. 2015-12-31, universe == 1
- Loss:  MSE (regression_l2), early stopping on val MSE
- Score: competition Sharpe on val predictions (metric.py)

Pass criterion: validation competition-Sharpe > 0 (we want clearly better than
the long-only +0.026 baseline on 2010+, ideally > 0.10 to be interesting).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))

import numpy as np
import pandas as pd
import pyarrow.parquet as papq
import lightgbm as lgb

from features import build_market_features, FEATURE_COLS, TARGET_COL
from metric import competition_score


ROOT = PIPE.parent

TRAIN_START = pd.Timestamp("2010-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2014-12-31", tz="UTC")
VAL_START = pd.Timestamp("2015-01-15", tz="UTC")    # 11-day embargo after train end
VAL_END = pd.Timestamp("2015-12-31", tz="UTC")


def main() -> int:
    print("Loading market.parquet ...")
    t0 = time.time()
    m = papq.read_table(ROOT / "market.parquet").to_pandas()
    print(f"  loaded {len(m):,} rows, {len(m.columns)} cols  ({time.time() - t0:.1f}s)")

    print("Building features ...")
    df = build_market_features(m)
    del m

    # Filter to a slightly larger window than train+val so 60-day lookbacks are warm,
    # then apply the actual masks. (build_market_features already populated columns
    # for the full panel, so we just slice.)
    df = df[df["time"] >= pd.Timestamp("2009-01-01", tz="UTC")].copy()

    train_mask = (
        (df["time"] >= TRAIN_START)
        & (df["time"] <= TRAIN_END)
        & (df["universe"] == 1)
    )
    val_mask = (
        (df["time"] >= VAL_START)
        & (df["time"] <= VAL_END)
        & (df["universe"] == 1)
    )

    train = df.loc[train_mask].copy()
    val = df.loc[val_mask].copy()

    print(
        f"\nSplits:\n"
        f"  train: {len(train):>9,} rows  "
        f"{train['time'].min().date()} .. {train['time'].max().date()}\n"
        f"  val:   {len(val):>9,} rows  "
        f"{val['time'].min().date()} .. {val['time'].max().date()}"
    )

    # ----- NaN audit -----
    nan_train = train[FEATURE_COLS].isna().sum()
    nan_train = nan_train[nan_train > 0]
    if len(nan_train) > 0:
        print(f"\n[WARN] feature_nan_audit: expected=0 NaNs, "
              f"got={dict(nan_train)}, fallback=LightGBM handles NaN natively")
    n_target_nan = int(train[TARGET_COL].isna().sum())
    if n_target_nan > 0:
        print(f"\n[WARN] target_nan: expected=0, got={n_target_nan}, "
              f"fallback=dropping these rows")
        train = train.dropna(subset=[TARGET_COL])

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_val = val[FEATURE_COLS]
    y_val = val[TARGET_COL]

    # ----- LightGBM -----
    print("\nTraining LightGBM ...")
    t0 = time.time()
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    print(
        f"  done in {time.time() - t0:.1f}s, "
        f"best_iter={model.best_iteration_}, "
        f"best_val_l2={model.best_score_['valid_0']['l2']:.6e}"
    )

    # ----- Predict and score -----
    print("\nScoring on 2015 validation ...")
    y_pred = model.predict(X_val)

    val_score, daily_val = competition_score(
        confidence=y_pred,
        returns=y_val.to_numpy(),
        universe=val["universe"].to_numpy(),
        day=val["time"].dt.date.to_numpy(),
        return_daily=True,
    )

    # Long-only reference on same val set
    long_only_score = competition_score(
        confidence=np.ones(len(val)),
        returns=y_val.to_numpy(),
        universe=val["universe"].to_numpy(),
        day=val["time"].dt.date.to_numpy(),
    )

    print(f"\n=== 2015 validation results ===")
    print(f"  model competition-Sharpe : {val_score:+.4f}")
    print(f"  long-only baseline       : {long_only_score:+.4f}")
    print(f"  improvement              : {val_score - long_only_score:+.4f}")
    print(f"  daily PnL mean / std     : {daily_val.mean():+.5f}  /  {daily_val.std(ddof=1):.5f}")
    print(f"  n trading days           : {len(daily_val)}")

    # ----- Sanity check 1: directional hit rate -----
    sign_match = (np.sign(y_pred) == np.sign(y_val)).mean()
    print(f"  sign(pred) == sign(target): {sign_match:.4f}")

    # ----- Sanity check 2: correlation -----
    pearson = np.corrcoef(y_pred, y_val)[0, 1]
    print(f"  pearson(pred, target)    : {pearson:+.4f}")

    # ----- Feature importance -----
    imp = pd.DataFrame(
        {"feature": FEATURE_COLS, "gain": model.feature_importances_}
    ).sort_values("gain", ascending=False)
    print("\nTop 15 features by gain:")
    print(imp.head(15).to_string(index=False))

    # ----- Pass criterion -----
    print()
    if val_score <= 0:
        print(f"[FAIL] validation Sharpe is non-positive ({val_score:+.4f}). "
              f"Baseline did not pass.")
        return 1
    if val_score <= long_only_score:
        print(f"[WARN] baseline did not beat long-only ({val_score:+.4f} <= "
              f"{long_only_score:+.4f}). Signal is weak; consider tuning before news features.")
        return 0
    print(f"Step 3 OK: validation Sharpe {val_score:+.4f} > 0 and beats long-only "
          f"({long_only_score:+.4f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
