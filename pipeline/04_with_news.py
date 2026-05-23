"""
Step 4: add news features and measure lift over the step-3 market-only baseline.

Same train/val split, same LightGBM config as step 3. Only difference is the
feature set: market + news vs market only. Run step 3 first to produce
news_daily.parquet (via aggregate_news.py).
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

from features import (
    build_market_features,
    merge_news_features,
    FEATURE_COLS,
    NEWS_FEATURE_COLS,
    TARGET_COL,
)
from metric import competition_score


ROOT = PIPE.parent

TRAIN_START = pd.Timestamp("2010-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2014-12-31", tz="UTC")
VAL_START = pd.Timestamp("2015-01-15", tz="UTC")
VAL_END = pd.Timestamp("2015-12-31", tz="UTC")


def train_and_score(X_train, y_train, X_val, y_val, val_df, label: str):
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
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    train_time = time.time() - t0

    y_pred = model.predict(X_val)
    score = competition_score(
        confidence=y_pred,
        returns=y_val.to_numpy(),
        universe=val_df["universe"].to_numpy(),
        day=val_df["time"].dt.date.to_numpy(),
    )

    pearson = float(np.corrcoef(y_pred, y_val)[0, 1])
    sign_match = float((np.sign(y_pred) == np.sign(y_val)).mean())

    print(
        f"\n--- {label} ---\n"
        f"  competition Sharpe : {score:+.4f}\n"
        f"  pearson(pred,tgt)  : {pearson:+.4f}\n"
        f"  sign hit-rate      : {sign_match:.4f}\n"
        f"  best_iter          : {model.best_iteration_}\n"
        f"  best_val_l2        : {model.best_score_['valid_0']['l2']:.6e}\n"
        f"  train time         : {train_time:.1f}s"
    )
    return model, score


def main() -> int:
    print("Loading market.parquet and news_daily.parquet ...")
    t0 = time.time()
    market = papq.read_table(ROOT / "market.parquet").to_pandas()
    news_daily = papq.read_table(ROOT / "news_daily.parquet").to_pandas()
    print(f"  market: {len(market):,} rows")
    print(f"  news_daily: {len(news_daily):,} rows")
    print(f"  ({time.time() - t0:.1f}s)")

    print("\nBuilding market features ...")
    df = build_market_features(market)
    del market

    print("Merging news features ...")
    df = merge_news_features(df, news_daily)
    del news_daily

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
        f"{val['time'].min().date()} .. {val['time'].max().date()}\n"
        f"  fraction of train rows with news : {train['has_news'].mean():.3f}\n"
        f"  fraction of val rows with news   : {val['has_news'].mean():.3f}"
    )

    # ----- Drop any rows with NaN target (spec says zero, defensive) -----
    n_tgt_nan = int(train[TARGET_COL].isna().sum())
    if n_tgt_nan > 0:
        print(f"[WARN] train_target_nan: expected=0, got={n_tgt_nan}, fallback=dropped")
        train = train.dropna(subset=[TARGET_COL])

    y_train = train[TARGET_COL]
    y_val = val[TARGET_COL]

    # ---- Baseline: market-only ----
    market_only_model, score_market = train_and_score(
        train[FEATURE_COLS], y_train,
        val[FEATURE_COLS], y_val,
        val, label="market-only (step 3 baseline)",
    )

    # ---- With news ----
    feat_with_news = FEATURE_COLS + NEWS_FEATURE_COLS
    full_model, score_full = train_and_score(
        train[feat_with_news], y_train,
        val[feat_with_news], y_val,
        val, label="market + news",
    )

    delta = score_full - score_market
    print(f"\n=== Lift from news features: {delta:+.4f} "
          f"({score_market:+.4f} -> {score_full:+.4f}) ===")

    # ---- News feature importance, ranked among ALL features ----
    imp = pd.DataFrame({
        "feature": feat_with_news,
        "gain": full_model.feature_importances_,
        "is_news": [c in NEWS_FEATURE_COLS for c in feat_with_news],
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1
    print("\nTop 20 features overall:")
    print(imp.head(20).to_string(index=False))
    print("\nNews features ranked:")
    print(imp[imp["is_news"]].to_string(index=False))

    # ---- Verdict ----
    print()
    if delta >= 0.05:
        print(f"News features add meaningful lift ({delta:+.4f}). Keep them.")
    elif delta >= 0.01:
        print(f"News features add marginal lift ({delta:+.4f}). Worth keeping but not transformative.")
    elif delta > -0.01:
        print(f"News features add ~no lift ({delta:+.4f}). Consider dropping for simpler model.")
    else:
        print(f"[WARN] news features HURT performance ({delta:+.4f}). "
              f"Likely added noise; investigate before keeping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
