"""
Aggregate news.parquet to one row per (assetCode, news_date).

`news_date` is the trading day a news item belongs to:
  - news with timestamp <  22:00 UTC of day t -> day t  (available before close)
  - news with timestamp >= 22:00 UTC of day t -> day t+1 (after close, not actionable until next day)

This is implemented as `news_date = (time + 2h).normalize().date()`.

Sentiment is aggregated as a relevance-weighted mean (separating sentiment from
volume; the previous attempt summed and conflated the two).

Output: news_daily.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as papq


ROOT = Path(__file__).resolve().parent.parent
NEWS_PATH = ROOT / "news.parquet"
MARKET_PATH = ROOT / "market.parquet"
OUT_PATH = ROOT / "news_daily.parquet"


# Only the columns we actually use downstream — keeps the in-memory footprint
# of the exploded news frame manageable.
NEWS_COLS = [
    "time",
    "assetCodes",
    "urgency",
    "relevance",
    "sentimentClass",
    "sentimentNegative",
    "sentimentNeutral",
    "sentimentPositive",
    "bodySize",
    "wordCount",
    "noveltyCount3D",
    "volumeCounts3D",
]


def main() -> int:
    t_total = time.time()

    print("Loading tradeable assetCodes from market.parquet ...")
    market_codes = set(
        papq.read_table(MARKET_PATH, columns=["assetCode"])
        .column("assetCode")
        .unique()
        .to_pylist()
    )
    print(f"  {len(market_codes):,} unique tradeable codes")

    print(f"\nLoading news.parquet ({len(NEWS_COLS)} cols) ...")
    t0 = time.time()
    news = papq.read_table(NEWS_PATH, columns=NEWS_COLS).to_pandas()
    print(f"  {len(news):,} rows in {time.time() - t0:.1f}s")

    # ---- news_date: shift by +2h so the day boundary is 22:00 UTC ----
    news["news_date"] = (
        (news["time"] + pd.Timedelta(hours=2))
        .dt.tz_convert("UTC")
        .dt.normalize()
        .dt.tz_localize(None)
    )

    # ---- precomputed flags + relevance-weighted contributions ----
    news["is_alert"] = (news["urgency"] == 1).astype(np.int8)
    rel = news["relevance"].astype("float32")
    news["w_neg"] = rel * news["sentimentNegative"]
    news["w_neu"] = rel * news["sentimentNeutral"]
    news["w_pos"] = rel * news["sentimentPositive"]
    news["w_class"] = rel * news["sentimentClass"].astype("float32")

    # ---- parse assetCodes set-string and explode ----
    print("\nParsing and exploding assetCodes ...")
    t0 = time.time()
    news["codes"] = news["assetCodes"].str.findall(r"'([^']+)'")
    n_empty = int(news["codes"].apply(len).eq(0).sum())
    if n_empty > 0:
        print(
            f"[WARN] aggregate_news.parse: expected=all rows parse to >=1 code, "
            f"got={n_empty} rows empty/unparseable, fallback=dropped"
        )
        news = news[news["codes"].apply(len) > 0]
    news = news.drop(columns=["assetCodes"])
    news = news.explode("codes").rename(columns={"codes": "assetCode"})
    print(f"  exploded to {len(news):,} rows in {time.time() - t0:.1f}s")

    # ---- filter to tradeable codes ----
    print("\nFiltering to tradeable assetCodes ...")
    t0 = time.time()
    n_before = len(news)
    news = news[news["assetCode"].isin(market_codes)]
    n_after = len(news)
    print(
        f"  kept {n_after:,} rows  "
        f"(dropped {n_before - n_after:,} = "
        f"{100 * (n_before - n_after) / n_before:.1f}% foreign/unmatched)  "
        f"({time.time() - t0:.1f}s)"
    )

    # ---- aggregate per (assetCode, news_date) ----
    print("\nAggregating per (assetCode, news_date) ...")
    t0 = time.time()
    agg = (
        news.groupby(["assetCode", "news_date"], sort=False)
        .agg(
            n_articles=("relevance", "count"),
            n_alerts=("is_alert", "sum"),
            sum_relevance=("relevance", "sum"),
            sum_w_neg=("w_neg", "sum"),
            sum_w_neu=("w_neu", "sum"),
            sum_w_pos=("w_pos", "sum"),
            sum_w_class=("w_class", "sum"),
            std_sent_class=("sentimentClass", "std"),
            mean_relevance=("relevance", "mean"),
            max_relevance=("relevance", "max"),
            mean_body_size=("bodySize", "mean"),
            mean_word_count=("wordCount", "mean"),
            mean_novelty_3d=("noveltyCount3D", "mean"),
            mean_volume_count_3d=("volumeCounts3D", "mean"),
        )
        .reset_index()
    )
    print(f"  produced {len(agg):,} (assetCode, day) rows in {time.time() - t0:.1f}s")

    # ---- relevance-weighted sentiment means ----
    sw = agg["sum_relevance"].where(agg["sum_relevance"] > 0, np.nan)
    agg["mean_sent_neg"] = agg["sum_w_neg"] / sw
    agg["mean_sent_neu"] = agg["sum_w_neu"] / sw
    agg["mean_sent_pos"] = agg["sum_w_pos"] / sw
    agg["mean_sent_class"] = agg["sum_w_class"] / sw

    # drop the intermediate weighted sums; keep total_relevance as a signal-strength feature
    agg = agg.drop(columns=["sum_w_neg", "sum_w_neu", "sum_w_pos", "sum_w_class"])
    agg = agg.rename(columns={"sum_relevance": "total_relevance"})

    # ---- sanity preview ----
    print("\nSample rows:")
    print(agg.head(5).to_string(index=False))
    print("\nSummary stats:")
    print(agg.describe().T.to_string())

    agg.to_parquet(OUT_PATH, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nWrote {OUT_PATH.name}  ({size_mb:.1f} MB, {len(agg):,} rows)")

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
