"""
data_loader.py - point-in-time (PIT) safe data loading PATTERN (vendor-agnostic).

The patterns here keep a research dataset honest:
  1. As-of joins that attach only data KNOWN at each timestamp (no future leak).
  2. A survivorship-aware universe (includes names that LATER delisted).
  3. Corporate-action adjustment from raw prices + factors.
  4. Hooks for exchange-calendar / session alignment and a parquet cache.

Replace the synthetic frames in __main__ and the TODO markers with your real
vendor feed. Store all raw timestamps in UTC.

Dependencies: numpy, pandas (pyarrow or fastparquet for the parquet cache).
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


# --- 1. Point-in-time as-of join ---------------------------------------------- #
def make_available_date(fundamentals: pd.DataFrame, period_end: str = "period_end",
                        lag_days: int = 45) -> pd.DataFrame:
    """Add a conservative availability date when no announcement date is known.
    Quarterly US filings: ~45 days (10-Q); annual ~90 (10-K). VERIFY per dataset
    and jurisdiction - using period_end directly leaks earnings weeks early."""
    out = fundamentals.copy()
    out["available_date"] = pd.to_datetime(out[period_end]) + pd.Timedelta(days=lag_days)
    return out


def pit_join(prices: pd.DataFrame, fundamentals: pd.DataFrame,
             price_date: str = "date", avail_date: str = "available_date",
             by: str = "symbol") -> pd.DataFrame:
    """Attach the most recent fundamental KNOWN as of each price date.

    `fundamentals[avail_date]` MUST be when the value became public (announcement
    / filing date), NOT the period_end it describes.

    merge_asof(direction='backward') = "last value at or before this date" = PIT.
    Any other direction ('forward'/'nearest') leaks the future. Both frames must
    be sorted by the as-of key.
    """
    p = prices.sort_values(price_date)
    f = fundamentals.sort_values(avail_date)
    return pd.merge_asof(
        p, f, left_on=price_date, right_on=avail_date, by=by,
        direction="backward", allow_exact_matches=True,
    )


# --- 2. Survivorship-aware universe ------------------------------------------- #
def universe_on(date, membership: pd.DataFrame, start: str = "start_date",
                end: str = "end_date", symbol: str = "symbol") -> list:
    """Symbols that were live members on `date`. `end` may be NaT for current
    members. Including later-delisted names is what removes survivorship bias.
    Key on a permanent security id in production - tickers get reused."""
    d = pd.Timestamp(date)
    m = membership
    live = (m[start] <= d) & (m[end].isna() | (m[end] > d))
    return sorted(m.loc[live, symbol].unique())


# --- 3. Corporate-action adjustment ------------------------------------------- #
def adjust_prices(raw: pd.DataFrame, factors: pd.DataFrame, on=("date", "symbol"),
                  price_col: str = "close", factor_col: str = "adj_factor") -> pd.DataFrame:
    """Multiply raw prices by a cumulative adjustment factor.
    Total-return factors include reinvested dividends; split-only factors don't.
    KEEP raw prices: level signals (round numbers, option strikes) must use raw,
    because dividend adjustment rewrites historical price levels."""
    df = raw.merge(factors, on=list(on), how="left")
    df[factor_col] = df[factor_col].fillna(1.0)
    df["adj_" + price_col] = df[price_col] * df[factor_col]
    return df


# --- 4a. Exchange-calendar / session hook (pluggable) ------------------------- #
def align_to_sessions(df: pd.DataFrame, sessions: Iterable, date: str = "date") -> pd.DataFrame:
    """Keep only rows on valid trading sessions. In production supply `sessions`
    from pandas-market-calendars or exchange_calendars (handles holidays /
    half-days / 24-7 crypto). TODO: plug your calendar here."""
    valid = set(pd.to_datetime(list(sessions)))
    return df[pd.to_datetime(df[date]).isin(valid)].reset_index(drop=True)


# --- 4b. Parquet cache -------------------------------------------------------- #
def cached_parquet(path: str, loader, *args, refresh: bool = False, **kwargs) -> pd.DataFrame:
    """Return a parquet cache if present, else build it via loader() and cache."""
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    df = loader(*args, **kwargs)  # TODO: real vendor loader plugs in here
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Self-tests / demo - run: python data_loader.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    dates = pd.bdate_range("2023-01-02", periods=10)
    prices = pd.DataFrame({"date": dates, "symbol": "AAA",
                           "close": np.arange(100.0, 110.0)})

    # fundamental for period ending 2022-12-31, ANNOUNCED 2023-01-09
    fundamentals = pd.DataFrame({
        "symbol": ["AAA"],
        "period_end": [pd.Timestamp("2022-12-31")],
        "available_date": [pd.Timestamp("2023-01-09")],
        "eps": [3.14],
    })

    joined = pit_join(prices, fundamentals)
    before = joined.loc[joined["date"] < "2023-01-09", "eps"]
    after = joined.loc[joined["date"] >= "2023-01-09", "eps"]
    # PIT proof: eps is UNKNOWN before its announcement, known on/after it
    assert before.isna().all(), "fundamental leaked before its announcement date!"
    assert (after == 3.14).all(), "fundamental missing after its announcement date!"

    # survivorship: a delisted name stays in-universe on dates when it was live
    membership = pd.DataFrame({
        "symbol": ["AAA", "ZZZ"],
        "start_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
        "end_date": [pd.NaT, pd.Timestamp("2023-01-05")],  # ZZZ delisted 2023-01-05
    })
    assert universe_on("2023-01-03", membership) == ["AAA", "ZZZ"]
    assert universe_on("2023-01-06", membership) == ["AAA"]

    # corporate-action adjustment produces an adjusted column
    factors = pd.DataFrame({"date": dates, "symbol": "AAA",
                            "adj_factor": np.linspace(0.9, 1.0, 10)})
    adj = adjust_prices(prices, factors)
    assert "adj_close" in adj.columns and len(adj) == len(prices)

    # availability-date helper applies the lag
    fd = make_available_date(fundamentals.drop(columns=["available_date"]), lag_days=45)
    assert (fd["available_date"] == pd.Timestamp("2022-12-31") + pd.Timedelta(days=45)).all()

    print("data_loader.py: all self-tests passed")
