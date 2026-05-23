"""
Step 6: held-out 2016 evaluation.

Setup (committed before looking at 2016):
  - Hyperparameters: best_params.json from step 5
  - Train:           2010-01-01 .. 2014-12-31  (universe == 1)
  - Val (early-stop):2015-01-18 .. 2015-12-31  (universe == 1)
  - Test:            2016-01-19 .. 2016-12-31  (universe == 1)
  - Embargo: ~18 calendar days at both boundaries (>= 11 trading days, >= target horizon)

Single shot. The market+news model is the headline. A market-only model trained
with the same hyperparameters is reported as an ablation. No selection between
variants after seeing 2016.

Outputs:
  - final_results.json
  - test_predictions.parquet  (asset, day, prediction, target — for any later analysis)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
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

TRAIN_START = pd.Timestamp("2010-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2014-12-31", tz="UTC")
VAL_START   = pd.Timestamp("2015-01-18", tz="UTC")
VAL_END     = pd.Timestamp("2015-12-31", tz="UTC")
TEST_START  = pd.Timestamp("2016-01-19", tz="UTC")
TEST_END    = pd.Timestamp("2016-12-31", tz="UTC")


def load_tuned_params() -> dict:
    cfg = json.load(open(ROOT / "best_params.json"))
    p = dict(cfg["best_params"])
    p.update(
        objective="regression",
        n_estimators=2000,
        bagging_freq=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    print(f"Loaded tuned params (step 5 fold-mean Sharpe was "
          f"{cfg['fold_mean_sharpe']:+.4f}, std {cfg['fold_std_sharpe']:.4f})")
    return p


def score_set(model, X, df_slice, label: str):
    pred = model.predict(X)
    s, daily = competition_score(
        confidence=pred,
        returns=df_slice[TARGET_COL].to_numpy(),
        universe=df_slice["universe"].to_numpy(),
        day=df_slice["time"].dt.date.to_numpy(),
        return_daily=True,
    )
    lo = competition_score(
        confidence=np.ones(len(df_slice)),
        returns=df_slice[TARGET_COL].to_numpy(),
        universe=df_slice["universe"].to_numpy(),
        day=df_slice["time"].dt.date.to_numpy(),
    )
    pearson = float(np.corrcoef(pred, df_slice[TARGET_COL])[0, 1])
    sign_match = float((np.sign(pred) == np.sign(df_slice[TARGET_COL])).mean())
    print(
        f"  {label:<32} Sharpe={s:+.4f}  long-only={lo:+.4f}  "
        f"pearson={pearson:+.4f}  sign={sign_match:.4f}  "
        f"daily(mean/std)={daily.mean():+.5f}/{daily.std(ddof=1):.5f}  "
        f"n_days={len(daily)}"
    )
    return pred, s, lo, daily


def main() -> int:
    params = load_tuned_params()

    print("\nLoading data + features ...")
    t0 = time.time()
    market = papq.read_table(ROOT / "market.parquet").to_pandas()
    news_daily = papq.read_table(ROOT / "news_daily.parquet").to_pandas()
    df = build_market_features(market)
    df = merge_news_features(df, news_daily)
    df = df.dropna(subset=[TARGET_COL])
    print(f"  ready in {time.time() - t0:.1f}s")
    del market, news_daily

    FEAT = FEATURE_COLS + NEWS_FEATURE_COLS

    def slc(start, end):
        m = (df["time"] >= start) & (df["time"] <= end) & (df["universe"] == 1)
        return df.loc[m].copy()

    train = slc(TRAIN_START, TRAIN_END)
    val = slc(VAL_START, VAL_END)
    test = slc(TEST_START, TEST_END)

    print(
        f"\nSplits:\n"
        f"  train: {len(train):>9,} rows  "
        f"{train['time'].min().date()} .. {train['time'].max().date()}\n"
        f"  val:   {len(val):>9,} rows    "
        f"{val['time'].min().date()} .. {val['time'].max().date()}\n"
        f"  test:  {len(test):>9,} rows   "
        f"{test['time'].min().date()} .. {test['time'].max().date()}"
    )

    # ---- OOD sanity (don't peek at target; just feature distribution) ----
    print(f"\nOOD sanity (close, train vs test):")
    for q in (0.10, 0.50, 0.90, 0.99):
        print(f"  q={q:.2f}  train={train['close'].quantile(q):>9.2f}   "
              f"test={test['close'].quantile(q):>9.2f}")

    # ====================================================================
    # MARKET + NEWS  (the headline model)
    # ====================================================================
    print("\n=== Training market+news model ===")
    t0 = time.time()
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[FEAT], train[TARGET_COL],
        eval_set=[(val[FEAT], val[TARGET_COL])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    print(f"  trained in {time.time() - t0:.1f}s, best_iter={model.best_iteration_}")

    print("\nScores (val sanity, then headline 2016):")
    _, val_score_mn, _, _ = score_set(model, val[FEAT], val, "val 2015 (sanity)")
    test_pred, test_score, test_lo, daily_test = score_set(
        model, test[FEAT], test, "TEST 2016 (held-out)"
    )

    # ---- Per-month 2016 breakdown ----
    test_aug = test.copy()
    test_aug["pred"] = test_pred
    test_aug["month"] = test_aug["time"].dt.to_period("M").astype(str)

    monthly = []
    for month, g in test_aug.groupby("month", sort=True):
        s = competition_score(
            g["pred"].to_numpy(), g[TARGET_COL].to_numpy(),
            g["universe"].to_numpy(), g["time"].dt.date.to_numpy(),
        )
        lo = competition_score(
            np.ones(len(g)), g[TARGET_COL].to_numpy(),
            g["universe"].to_numpy(), g["time"].dt.date.to_numpy(),
        )
        monthly.append({
            "month": month,
            "sharpe": round(s, 4),
            "long_only": round(lo, 4),
            "lift": round(s - lo, 4),
            "n_rows": len(g),
        })
    print(f"\n2016 per-month breakdown:")
    print(pd.DataFrame(monthly).to_string(index=False))

    # ---- Decile calibration ----
    test_aug["pred_decile"] = pd.qcut(
        test_aug["pred"], 10, labels=False, duplicates="drop"
    )
    cal = (
        test_aug.groupby("pred_decile", observed=True)
        .agg(pred_mean=("pred", "mean"),
             target_mean=(TARGET_COL, "mean"),
             n=("pred", "count"))
        .reset_index()
    )
    print(f"\n2016 calibration (deciles of prediction -> realized target):")
    print(cal.to_string(index=False))

    # ---- Volatility regime breakdown ----
    test_aug["vol_bin"] = pd.qcut(
        test_aug["vol_20"].fillna(test_aug["vol_20"].median()),
        3, labels=["low", "med", "high"]
    )
    print(f"\n2016 by 20d-vol regime:")
    for regime, g in test_aug.groupby("vol_bin", observed=True):
        s = competition_score(
            g["pred"].to_numpy(), g[TARGET_COL].to_numpy(),
            g["universe"].to_numpy(), g["time"].dt.date.to_numpy(),
        )
        print(f"  {regime:>4}: Sharpe={s:+.4f}  n={len(g):>7,}")

    # ---- Feature importance ----
    imp = pd.DataFrame({
        "feature": FEAT,
        "gain": model.feature_importances_,
        "is_news": [c in NEWS_FEATURE_COLS for c in FEAT],
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", imp.index + 1)
    print(f"\nTop 20 features (final model):")
    print(imp.head(20).to_string(index=False))

    # ---- Prediction distribution ----
    print(f"\nPrediction distribution on 2016:")
    print(f"  min/median/max : {test_pred.min():+.5f} / "
          f"{np.median(test_pred):+.5f} / {test_pred.max():+.5f}")
    print(f"  mean / std     : {test_pred.mean():+.5f} / {test_pred.std():.5f}")

    # ====================================================================
    # ABLATION: market-only at same hyperparameters
    # ====================================================================
    print("\n=== Ablation: market-only (same hyperparameters) ===")
    t0 = time.time()
    model_mo = lgb.LGBMRegressor(**params)
    model_mo.fit(
        train[FEATURE_COLS], train[TARGET_COL],
        eval_set=[(val[FEATURE_COLS], val[TARGET_COL])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    print(f"  trained in {time.time() - t0:.1f}s, best_iter={model_mo.best_iteration_}")
    _, val_score_mo, _, _ = score_set(model_mo, val[FEATURE_COLS], val, "val 2015 mkt-only")
    pred_mo, test_score_mo, _, _ = score_set(
        model_mo, test[FEATURE_COLS], test, "TEST 2016 mkt-only"
    )

    # ====================================================================
    # Persist predictions and report
    # ====================================================================
    out = pd.DataFrame({
        "assetCode": test["assetCode"].to_numpy(),
        "time": test["time"].to_numpy(),
        "target": test[TARGET_COL].to_numpy(),
        "universe": test["universe"].to_numpy(),
        "pred_full": test_pred,
        "pred_market_only": pred_mo,
    })
    out.to_parquet(ROOT / "test_predictions.parquet", compression="snappy")
    print(f"\nWrote test_predictions.parquet ({len(out):,} rows)")

    report = {
        "headline": {
            "test_2016_sharpe_market_plus_news": round(float(test_score), 4),
            "test_2016_sharpe_market_only":      round(float(test_score_mo), 4),
            "news_lift_on_test":                 round(float(test_score - test_score_mo), 4),
            "long_only_2016_baseline":           round(float(test_lo), 4),
        },
        "val_sanity": {
            "val_2015_sharpe_market_plus_news":  round(float(val_score_mn), 4),
            "val_2015_sharpe_market_only":       round(float(val_score_mo), 4),
        },
        "cv_reference": {
            "step5_fold_mean_sharpe": json.load(open(ROOT / "best_params.json"))["fold_mean_sharpe"],
            "step5_fold_std_sharpe":  json.load(open(ROOT / "best_params.json"))["fold_std_sharpe"],
        },
        "model_state": {
            "best_iter_market_plus_news": int(model.best_iteration_),
            "best_iter_market_only":      int(model_mo.best_iteration_),
        },
        "monthly_2016": monthly,
        "calibration_2016": cal.to_dict(orient="records"),
        "top_15_features": imp.head(15).to_dict(orient="records"),
    }
    json.dump(report, open(ROOT / "final_results.json", "w"), indent=2, default=str)
    print(f"Wrote final_results.json")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print(f"HEADLINE: 2016 OOS competition Sharpe = {test_score:+.4f}")
    print(f"  long-only 2016 baseline       = {test_lo:+.4f}")
    print(f"  market-only ablation          = {test_score_mo:+.4f}")
    print(f"  CV fold-mean (step 5)         = {report['cv_reference']['step5_fold_mean_sharpe']:+.4f}")
    print("=" * 70)
    if test_score > 0.5:
        print(f"SUCCESS: cleared the locked-in +0.50 pass criterion.")
    elif test_score > test_lo:
        print(f"PARTIAL: positive Sharpe and beats long-only, but did not clear +0.50.")
    elif test_score > 0:
        print(f"WEAK PASS: positive Sharpe but worse than always-long baseline.")
    else:
        print(f"FAIL: non-positive Sharpe on held-out 2016.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
