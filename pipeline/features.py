"""
Market feature engineering.

Adds per-asset rolling features to the raw market panel. All features are
computed from data up to and including day t, so they are safe to use as
predictors of the forward-looking target on day t.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


BASE_RET = "returnsClosePrevRaw1"


def build_market_features(market: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling features per asset to the market panel.

    Input columns required: time, assetCode, close, open, volume, BASE_RET
    Output: same DataFrame with new feature columns appended.
    """
    t0 = time.time()
    df = market.sort_values(["assetCode", "time"]).copy()
    g = df.groupby("assetCode", sort=False)

    # --- Rolling volatility of 1-day close-to-close returns ---
    for w in (5, 10, 20, 60):
        df[f"vol_{w}"] = (
            g[BASE_RET]
            .rolling(window=w, min_periods=max(w // 2, 2))
            .std()
            .reset_index(level=0, drop=True)
        )

    # --- Rolling mean of 1-day returns (short-window momentum proxies) ---
    for w in (5, 10, 20, 60):
        df[f"ret_ma_{w}"] = (
            g[BASE_RET]
            .rolling(window=w, min_periods=max(w // 2, 2))
            .mean()
            .reset_index(level=0, drop=True)
        )

    # --- Price relative to moving average (medium / long momentum) ---
    for w in (20, 60):
        ma = (
            g["close"]
            .rolling(window=w, min_periods=max(w // 2, 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"price_to_ma_{w}"] = df["close"] / ma - 1.0

    # --- Volume features ---
    vol_ma_20 = (
        g["volume"]
        .rolling(window=20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    vol_std_20 = (
        g["volume"]
        .rolling(window=20, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["volume_z_20"] = (df["volume"] - vol_ma_20) / vol_std_20
    df["log_volume"] = np.log1p(df["volume"])
    df["log_dollar_volume"] = np.log1p(df["volume"] * df["close"])

    # --- Calendar features ---
    df["day_of_week"] = df["time"].dt.dayofweek.astype("int8")
    df["month"] = df["time"].dt.month.astype("int8")

    print(
        f"  build_market_features: {len(df):,} rows, "
        f"{df.shape[1]} cols, {time.time() - t0:.1f}s"
    )
    return df


# Feature columns the model will see.
FEATURE_COLS = [
    # Raw market state
    "close",
    "open",
    "volume",
    # Pre-computed lookback returns (already in dataset)
    "returnsClosePrevRaw1",
    "returnsOpenPrevRaw1",
    "returnsClosePrevMktres1",
    "returnsOpenPrevMktres1",
    "returnsClosePrevRaw10",
    "returnsOpenPrevRaw10",
    "returnsClosePrevMktres10",
    "returnsOpenPrevMktres10",
    # Engineered rolling features
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_60",
    "ret_ma_5",
    "ret_ma_10",
    "ret_ma_20",
    "ret_ma_60",
    "price_to_ma_20",
    "price_to_ma_60",
    "volume_z_20",
    "log_volume",
    "log_dollar_volume",
    # Calendar
    "day_of_week",
    "month",
]

TARGET_COL = "returnsOpenNextMktres10"


# ----------------------------------------------------------------------------
# News features
# ----------------------------------------------------------------------------

NEWS_FEATURE_COLS = [
    "n_articles",
    "n_alerts",
    "total_relevance",
    "mean_relevance",
    "max_relevance",
    "std_sent_class",
    "mean_sent_neg",
    "mean_sent_neu",
    "mean_sent_pos",
    "mean_sent_class",
    "mean_body_size",
    "mean_word_count",
    "mean_novelty_3d",
    "mean_volume_count_3d",
    "has_news",
]


def merge_news_features(market_df: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join `news_daily` onto a market panel by (assetCode, date).

    market_df.time is at 22:00 UTC of the trading day; news_daily.news_date is
    the trading day a news item is assigned to (UTC-shifted by +2h). Both are
    normalized to a tz-naive midnight datetime for the join.
    """
    t0 = time.time()
    market_df = market_df.copy()
    market_df["__key_date"] = (
        market_df["time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    )

    # news_daily.news_date is already tz-naive normalized to midnight (per aggregate_news.py)
    nd = news_daily.rename(columns={"news_date": "__key_date"})

    merged = market_df.merge(
        nd, on=["assetCode", "__key_date"], how="left"
    )
    merged["has_news"] = merged["n_articles"].notna().astype("int8")
    merged = merged.drop(columns=["__key_date"])

    print(
        f"  merge_news_features: market={len(market_df):,} rows -> "
        f"{len(merged):,} merged ({100 * merged['has_news'].mean():.1f}% with news), "
        f"{time.time() - t0:.1f}s"
    )
    return merged


# ----------------------------------------------------------------------------
# Cross-sectional rank features  (Two Sigma top-finisher pattern)
# ----------------------------------------------------------------------------

# Features for which a within-day cross-sectional rank is meaningful.
# Skip calendar (month, day_of_week — ranking is meaningless) and raw close/open
# (rank approximates size and is partly redundant with log_dollar_volume_xrank).
_RANK_BASE_COLS = [
    "returnsClosePrevRaw1",
    "returnsOpenPrevRaw1",
    "returnsClosePrevMktres1",
    "returnsOpenPrevMktres1",
    "returnsClosePrevRaw10",
    "returnsOpenPrevRaw10",
    "returnsClosePrevMktres10",
    "returnsOpenPrevMktres10",
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_60",
    "ret_ma_5",
    "ret_ma_10",
    "ret_ma_20",
    "ret_ma_60",
    "price_to_ma_20",
    "price_to_ma_60",
    "volume_z_20",
    "log_dollar_volume",
]

RANK_FEATURE_COLS = [f"{c}_xrank" for c in _RANK_BASE_COLS]


def add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each base feature, add a within-day percentile-rank column in [0, 1].
    NaN inputs receive NaN ranks (LightGBM handles natively).

    The rank is computed across ALL rows in a given day, not just universe==1.
    The universe filter is a scoring concept, not a "what's available cross-sectionally" concept.
    """
    t0 = time.time()
    df = df.copy()
    # Group key: date of the trading day. df['time'] is UTC 22:00; .dt.date is the trading day.
    day_key = df["time"].dt.date
    for col in _RANK_BASE_COLS:
        df[f"{col}_xrank"] = df.groupby(day_key, sort=False)[col].rank(
            pct=True, method="average"
        )
    print(
        f"  add_cross_sectional_ranks: +{len(_RANK_BASE_COLS)} rank cols, "
        f"{time.time() - t0:.1f}s"
    )
    return df


def rank_confidence(preds, days):
    """
    Convert raw regression predictions to within-day percentile-rank confidence in [-1, +1].
    This is metric-aware post-processing: it stabilizes per-day prediction
    magnitudes (lowering the std denominator in the competition Sharpe) while
    preserving the directional ordering the model learned.
    """
    s = pd.DataFrame({"p": preds, "d": days})
    s["rank_pct"] = s.groupby("d", sort=False)["p"].rank(pct=True, method="average")
    return (2.0 * s["rank_pct"] - 1.0).to_numpy()
