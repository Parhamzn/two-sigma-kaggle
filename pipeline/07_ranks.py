"""
Experiment: cross-sectional rank features (#1) + rank-based confidence mapping (#2).

5-fold walk-forward CV identical to step 5. Same hyperparameters (best_params.json).
2016 is NOT touched here — we evaluate purely on the 2011..2015 walk-forward folds.

Four variants per fold:
  A: step 5 features, raw confidence       <-- baseline reproduction
  B: +rank features,  raw confidence       <-- effect of features alone
  C: step 5 features, rank confidence      <-- effect of mapping alone
  D: +rank features,  rank confidence      <-- the full proposal

Per-fold comparison is the honest read; the four variants share train/val splits
within each fold so we control for fold-specific noise.
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
    add_cross_sectional_ranks,
    rank_confidence,
    FEATURE_COLS,
    NEWS_FEATURE_COLS,
    RANK_FEATURE_COLS,
    TARGET_COL,
)
from metric import competition_score


ROOT = PIPE.parent

# Same folds as step 5 (walk-forward expanding, ~18-day embargo)
FOLDS = [
    ("2010-01-01", "2010-12-31", "2011-01-18", "2011-12-31"),
    ("2010-01-01", "2011-12-31", "2012-01-18", "2012-12-31"),
    ("2010-01-01", "2012-12-31", "2013-01-18", "2013-12-31"),
    ("2010-01-01", "2013-12-31", "2014-01-18", "2014-12-31"),
    ("2010-01-01", "2014-12-31", "2015-01-18", "2015-12-31"),
]


def load_params() -> dict:
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
    return p, cfg


def slice_universe(df, start, end):
    m = (
        (df["time"] >= pd.Timestamp(start, tz="UTC"))
        & (df["time"] <= pd.Timestamp(end, tz="UTC"))
        & (df["universe"] == 1)
    )
    return df.loc[m]


def fit_and_predict(train, val, feat_cols, params):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[feat_cols],
        train[TARGET_COL],
        eval_set=[(val[feat_cols], val[TARGET_COL])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return model.predict(val[feat_cols]), int(model.best_iteration_)


def score_with(preds, val, mode: str) -> float:
    days = val["time"].dt.date.to_numpy()
    if mode == "raw":
        conf = preds
    elif mode == "rank":
        conf = rank_confidence(preds, days)
    else:
        raise ValueError(mode)
    return competition_score(
        confidence=conf,
        returns=val[TARGET_COL].to_numpy(),
        universe=val["universe"].to_numpy(),
        day=days,
    )


def main() -> int:
    params, cfg = load_params()
    print(f"Using tuned params from step 5 "
          f"(fold-mean Sharpe was {cfg['fold_mean_sharpe']:+.4f})")

    print("\nLoading data and building features ...")
    t0 = time.time()
    market = papq.read_table(ROOT / "market.parquet").to_pandas()
    news_daily = papq.read_table(ROOT / "news_daily.parquet").to_pandas()
    df = build_market_features(market)
    df = merge_news_features(df, news_daily)
    df = add_cross_sectional_ranks(df)
    df = df[
        (df["time"] >= pd.Timestamp("2009-01-01", tz="UTC"))
        & (df["time"] < pd.Timestamp("2016-01-01", tz="UTC"))
    ].dropna(subset=[TARGET_COL]).copy()
    print(f"  pool: {len(df):,} rows  ({time.time() - t0:.1f}s)")
    del market, news_daily

    BASE_FEAT = FEATURE_COLS + NEWS_FEATURE_COLS                  # step 5 set
    RANK_FEAT = BASE_FEAT + RANK_FEATURE_COLS                     # +rank features

    rows = []
    t_total = time.time()
    for i, (ts, te, vs, ve) in enumerate(FOLDS, 1):
        train = slice_universe(df, ts, te)
        val = slice_universe(df, vs, ve)
        print(
            f"\n--- Fold {i}  "
            f"train {ts}..{te} ({len(train):,})  "
            f"val {vs}..{ve} ({len(val):,}) ---"
        )

        # Model M1: step-5 feature set
        t0 = time.time()
        preds_base, iter_base = fit_and_predict(train, val, BASE_FEAT, params)
        t_base = time.time() - t0

        # Model M2: step-5 + rank features
        t0 = time.time()
        preds_rank, iter_rank = fit_and_predict(train, val, RANK_FEAT, params)
        t_rank = time.time() - t0

        # Score all four variants
        a = score_with(preds_base, val, "raw")
        b = score_with(preds_rank, val, "raw")
        c = score_with(preds_base, val, "rank")
        d = score_with(preds_rank, val, "rank")

        print(
            f"  A base+raw  : Sharpe={a:+.4f}  iter={iter_base}  fit={t_base:.1f}s\n"
            f"  B rank+raw  : Sharpe={b:+.4f}  iter={iter_rank}  fit={t_rank:.1f}s\n"
            f"  C base+rank : Sharpe={c:+.4f}\n"
            f"  D rank+rank : Sharpe={d:+.4f}  <-- proposal"
        )
        rows.append({"fold": i, "A_base_raw": a, "B_rank_raw": b,
                     "C_base_rank": c, "D_rank_rank": d,
                     "iter_base": iter_base, "iter_rank": iter_rank})

    table = pd.DataFrame(rows).set_index("fold")
    means = table.iloc[:, :4].mean()
    stds = table.iloc[:, :4].std(ddof=1)

    print("\n" + "=" * 70)
    print("Fold-by-fold competition Sharpe:")
    print(table.iloc[:, :4].round(4).to_string())
    print("\nFold-mean (and std across folds):")
    for col in table.columns[:4]:
        print(f"  {col:<14} {means[col]:+.4f}  (std {stds[col]:.4f})")

    print("\nLifts vs A (baseline = step 5 reproduction):")
    for col in ["B_rank_raw", "C_base_rank", "D_rank_rank"]:
        delta = means[col] - means["A_base_raw"]
        print(f"  {col:<14} delta = {delta:+.4f}")

    # Save
    out = {
        "fold_mean_A_base_raw":  round(float(means["A_base_raw"]), 4),
        "fold_mean_B_rank_raw":  round(float(means["B_rank_raw"]), 4),
        "fold_mean_C_base_rank": round(float(means["C_base_rank"]), 4),
        "fold_mean_D_rank_rank": round(float(means["D_rank_rank"]), 4),
        "delta_B_minus_A": round(float(means["B_rank_raw"] - means["A_base_raw"]), 4),
        "delta_C_minus_A": round(float(means["C_base_rank"] - means["A_base_raw"]), 4),
        "delta_D_minus_A": round(float(means["D_rank_rank"] - means["A_base_raw"]), 4),
        "fold_table": table.reset_index().to_dict(orient="records"),
        "step5_fold_mean_for_reference": cfg["fold_mean_sharpe"],
    }
    out_path = ROOT / "ranks_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path.name}")

    print(f"\nTotal wall time: {(time.time() - t_total) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
