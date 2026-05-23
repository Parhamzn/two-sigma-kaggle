"""
Two Sigma competition scoring metric.

For each trading day t:
    x_t = sum_i ( confidence_{t,i} * return_{t,i} * universe_{t,i} )

score = mean(x_t) / std(x_t)

This is the metric Kaggle used to rank submissions. It rewards both directional
correctness AND confident sizing, and penalizes day-to-day PnL volatility.

The canonical starter code uses pandas `.std()` which defaults to ddof=1; this
module matches that as the default. ddof is exposed as a parameter for users
who want to reproduce alternate community implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def competition_score(
    confidence,
    returns,
    universe,
    day,
    ddof: int = 1,
    return_daily: bool = False,
):
    """
    Two Sigma competition Sharpe-like score.

    Parameters
    ----------
    confidence : array-like
        Predicted confidence per (asset, day), each value in [-1, +1].
    returns : array-like
        Realized return (returnsOpenNextMktres10) per (asset, day).
    universe : array-like
        0/1 indicator: is this (asset, day) in the scoring universe?
    day : array-like
        Trading-day grouping key (can be int, date, timestamp).
    ddof : int
        Delta degrees of freedom for std. Default 1 (pandas convention).
    return_daily : bool
        If True, also return the array of daily x_t values for diagnostics.

    Returns
    -------
    float, or (float, np.ndarray) if return_daily=True.

    Notes
    -----
    Returns 0.0 if std is undefined (single day) or exactly zero (constant
    daily PnL). This is a deterministic fallback for CV folds; in practice
    real validation folds have many days and nonzero variance.
    """
    confidence = np.asarray(confidence, dtype=float)
    returns = np.asarray(returns, dtype=float)
    universe = np.asarray(universe, dtype=float)
    day = np.asarray(day)

    if not (len(confidence) == len(returns) == len(universe) == len(day)):
        raise ValueError(
            f"length mismatch: confidence={len(confidence)} returns={len(returns)} "
            f"universe={len(universe)} day={len(day)}"
        )

    contrib = confidence * returns * universe
    daily = (
        pd.DataFrame({"day": day, "x": contrib})
        .groupby("day", sort=True)["x"]
        .sum()
        .to_numpy()
    )

    mean = daily.mean() if daily.size else 0.0
    std = daily.std(ddof=ddof) if daily.size > ddof else float("nan")

    if std == 0 or np.isnan(std):
        print(
            f"[WARN] competition_score: expected=valid std, "
            f"got=std={std} (n_days={daily.size}, ddof={ddof}), "
            f"fallback=score=0.0",
            flush=True,
        )
        score = 0.0
    else:
        score = float(mean / std)

    if return_daily:
        return score, daily
    return score


# ----------------------------------------------------------------------------
# Self tests
# ----------------------------------------------------------------------------

def _synthetic_panel(n_days: int = 100, n_assets: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(n_days), n_assets)
    universe = np.ones(n_days * n_assets, dtype=float)
    # mu=0.1%, sigma=2% — roughly daily equity return shape
    returns = rng.normal(0.001, 0.02, n_days * n_assets)
    return days, returns, universe


def _self_test():
    print("=== metric self-test ===")

    days, returns, universe = _synthetic_panel()

    # 1. Zero confidence -> contributions are all zero -> std=0 -> score 0
    s = competition_score(np.zeros_like(returns), returns, universe, days)
    assert s == 0.0, f"zero predictor: expected 0, got {s}"
    print(f"  [OK] zero predictor                       : {s:.4f}")

    # 2. Perfect foresight (conf = sign(r)) -> consistently positive daily PnL
    conf_perfect = np.sign(returns)
    s_perfect = competition_score(conf_perfect, returns, universe, days)
    assert s_perfect > 5.0, f"perfect predictor: expected >>0, got {s_perfect}"
    print(f"  [OK] perfect foresight (sign(r))          : {s_perfect:.4f}")

    # 3. Anti-perfect -> exactly -score(perfect)
    s_anti = competition_score(-conf_perfect, returns, universe, days)
    assert abs(s_anti + s_perfect) < 1e-9, (
        f"anti-perfect: expected {-s_perfect}, got {s_anti}"
    )
    print(f"  [OK] anti-perfect = -perfect              : {s_anti:.4f}")

    # 4. Random predictor over many seeds: mean ~ 0, with spread
    rs = []
    for seed in range(50):
        rng = np.random.default_rng(1000 + seed)
        conf = rng.uniform(-1, 1, len(returns))
        rs.append(competition_score(conf, returns, universe, days))
    rs = np.array(rs)
    assert abs(rs.mean()) < 0.3, f"random predictor mean: expected ~0, got {rs.mean()}"
    print(
        f"  [OK] random uniform predictor (50 seeds)  : "
        f"mean={rs.mean():+.4f}  std={rs.std():.4f}"
    )

    # 5. Universe filter is equivalent to zeroing confidences out of universe
    universe_half = (np.arange(len(returns)) % 2 == 0).astype(float)
    s_via_universe = competition_score(conf_perfect, returns, universe_half, days)
    s_via_conf = competition_score(
        conf_perfect * universe_half, returns, np.ones_like(universe), days
    )
    assert abs(s_via_universe - s_via_conf) < 1e-12, (
        f"universe filter mismatch: {s_via_universe} vs {s_via_conf}"
    )
    print(
        f"  [OK] universe filter == zeroing conf       : "
        f"{s_via_universe:.4f} == {s_via_conf:.4f}"
    )

    # 6. Always-long predictor on synthetic data — should be positive (mu > 0)
    conf_long = np.ones_like(returns)
    s_long = competition_score(conf_long, returns, universe, days)
    assert s_long > 0, f"always-long on mu>0 data: expected >0, got {s_long}"
    print(f"  [OK] always-long on bullish synthetic     : {s_long:.4f}")

    # 7. Scale-invariance check: scaling all confidences by c>0 does NOT change
    #    the score (numerator and denominator scale together).
    s_2x = competition_score(2 * conf_perfect.clip(-1, 1), returns, universe, days)
    assert abs(s_2x - s_perfect) < 1e-12, f"scale-invariance broken: {s_2x} vs {s_perfect}"
    print(f"  [OK] scale-invariance c=2                 : {s_2x:.4f} == {s_perfect:.4f}")

    # 8. Single-day edge case -> std undefined -> deterministic 0 with [WARN]
    s_single = competition_score(
        np.array([0.5, -0.3]),
        np.array([0.01, -0.02]),
        np.array([1.0, 1.0]),
        np.array([0, 0]),
    )
    assert s_single == 0.0, f"single-day edge: expected 0, got {s_single}"
    print(f"  [OK] single-day edge case (n=1)           : {s_single:.4f}")

    print("\nAll synthetic tests passed.\n")


def _real_data_probe():
    """Compute the long-only baseline on real market data. Establishes the
    Sharpe a 'always +1 on every universe asset' strategy would have achieved
    over the full 2007–2016 sample. The model has to beat this to be useful."""
    import pyarrow.parquet as papq
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    print("=== real-data probe (always-long on universe assets) ===")
    cols = ["time", "returnsOpenNextMktres10", "universe"]
    m = papq.read_table(root / "market.parquet", columns=cols).to_pandas()
    m = m.dropna(subset=["returnsOpenNextMktres10"])  # spec says target is never null, but defensive
    m["date"] = m["time"].dt.date

    for label, mask in [
        ("full sample (2007-02 .. 2016-12)", slice(None)),
        ("2010 onward",   m["time"] >= pd.Timestamp("2010-01-01", tz="UTC")),
        ("2015 (val)",    (m["time"] >= pd.Timestamp("2015-01-01", tz="UTC")) &
                          (m["time"] <  pd.Timestamp("2016-01-01", tz="UTC"))),
        ("2016 (test)",   m["time"] >= pd.Timestamp("2016-01-01", tz="UTC")),
    ]:
        sub = m.loc[mask] if not isinstance(mask, slice) else m
        s, daily = competition_score(
            confidence=np.ones(len(sub)),
            returns=sub["returnsOpenNextMktres10"].to_numpy(),
            universe=sub["universe"].to_numpy(),
            day=sub["date"].to_numpy(),
            return_daily=True,
        )
        n_days = len(daily)
        print(
            f"  {label:<36} score={s:+.4f}  "
            f"daily_mean={daily.mean():+.5f}  daily_std={daily.std(ddof=1):.5f}  "
            f"n_days={n_days}"
        )


if __name__ == "__main__":
    _self_test()
    _real_data_probe()
