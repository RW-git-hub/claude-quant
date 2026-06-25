"""
backtest_skeleton.py - a MINIMAL, leak-free backtest skeleton (teaching, not production).

The whole point is to make the most common fatal bug structurally impossible:
look-ahead via same-bar execution. Positions are LAGGED relative to the returns
they earn:

    pnl_t = position_{t-1} * return_t      # weight decided at close t-1 earns return_t

Pluggable cost model and position sizer; a walk-forward splitter with embargo; and
a runnable demo that PROVES same-bar execution fabricates profit.

This is a single-asset skeleton kept deliberately small. For real work add:
multi-asset weights, capacity/liquidity caps, borrow/funding, and a proper
event-driven engine (backtrader / zipline-reloaded) or vectorbt for sweeps.

Dependencies: numpy, pandas. See metrics.py for the canonical metric definitions.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


# --- minimal metrics (canonical, fuller versions live in metrics.py) ---------- #
def _sharpe(returns: pd.Series, ppy: int = 252) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd and sd > 0 else float("nan")


def _ann_return(returns: pd.Series, ppy: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    g = float((1 + r).prod())
    return g ** (ppy / len(r)) - 1 if g > 0 else -1.0


def _ann_vol(returns: pd.Series, ppy: int = 252) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(ppy)) if len(r) > 1 else float("nan")


def _max_drawdown(returns: pd.Series) -> float:
    eq = (1 + returns.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


# --- cost model: callable(turnover_series) -> per-period cost in return units -- #
def fixed_bps_cost(commission_bps: float = 1.0,
                   half_spread_bps: float = 2.0) -> Callable[[pd.Series], pd.Series]:
    rate = (commission_bps + half_spread_bps) / 1e4
    def cost(turnover: pd.Series) -> pd.Series:
        return turnover.abs() * rate
    return cost


# --- position sizer: callable(raw_signal, returns) -> target weights ---------- #
def fixed_sizer(gross: float = 1.0) -> Callable[[pd.Series, pd.Series], pd.Series]:
    def size(signal: pd.Series, returns: pd.Series) -> pd.Series:
        return signal.clip(-1, 1) * gross
    return size


def vol_target_sizer(target_ann_vol: float = 0.10, lookback: int = 20,
                     ppy: int = 252, max_leverage: float = 3.0,
                     vol_floor: float = 0.0
                     ) -> Callable[[pd.Series, pd.Series], pd.Series]:
    def size(signal: pd.Series, returns: pd.Series) -> pd.Series:
        # realized vol from PAST returns only (.shift(1)) -> no look-ahead
        realized = returns.rolling(lookback).std(ddof=1).shift(1) * np.sqrt(ppy)
        # Floor the vol estimate: in a near-zero-vol window target/realized explodes.
        # max_leverage caps the OUTPUT but not the instability; vol_floor bounds the
        # ratio directly (cap = target_ann_vol/vol_floor). Default 0.0 = prior behavior.
        if vol_floor > 0.0:
            realized = realized.clip(lower=vol_floor)
        scaler = (target_ann_vol / realized).clip(upper=max_leverage).fillna(0.0)
        return signal.clip(-1, 1) * scaler
    return size


def build_positions(raw_weights: pd.Series, lag: int = 1) -> pd.Series:
    """Lag target weights so a position can only earn FUTURE returns.
    lag=1 (default) is correct. lag=0 is the same-bar look-ahead BUG, kept only
    so the demo below can show how badly it inflates results."""
    return raw_weights.shift(lag)


def run_backtest(prices,
                 signal_func: Callable[[pd.Series], pd.Series],
                 sizer: Optional[Callable[[pd.Series, pd.Series], pd.Series]] = None,
                 cost_model: Optional[Callable[[pd.Series], pd.Series]] = None,
                 lag: int = 1, ppy: int = 252) -> dict:
    sizer = sizer or fixed_sizer()
    cost_model = cost_model or fixed_bps_cost()

    prices = pd.Series(prices, dtype="float64")
    returns = prices.pct_change(fill_method=None)

    signal = signal_func(prices)                  # uses only past/current prices
    target = sizer(signal, returns)               # target weights (pre-lag)
    positions = build_positions(target, lag=lag)  # LAGGED -> leak-free for lag>=1

    turnover = positions.diff().abs()             # |w_t - w_{t-1}|
    costs = cost_model(turnover)
    gross = positions * returns                    # pnl_t = position_{t-1} * return_t
    net = (gross - costs).dropna()

    return {
        "equity": (1 + net).cumprod(),
        "net_returns": net,
        "positions": positions,
        "metrics": {
            "sharpe": _sharpe(net, ppy),
            "ann_return": _ann_return(net, ppy),
            "ann_vol": _ann_vol(net, ppy),
            "max_drawdown": _max_drawdown(net),
            "avg_turnover": float(turnover.dropna().mean()),
            "n_periods": int(net.size),
        },
    }


def walk_forward_splits(n_obs: int, train_size: int, test_size: int, embargo: int = 0):
    """Yield (train_idx, test_idx) for rolling walk-forward evaluation.
    `embargo` drops observations between train end and test start to prevent
    leakage from label overlap / autocorrelation (Lopez de Prado). Use an
    embargo >= the label horizon of your target."""
    start = 0
    while start + train_size + embargo + test_size <= n_obs:
        train_idx = np.arange(start, start + train_size)
        test_start = start + train_size + embargo
        test_idx = np.arange(test_start, test_start + test_size)
        yield train_idx, test_idx
        start += test_size


def ma_crossover(prices: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """+1 when the fast MA is above the slow MA, -1 otherwise. Uses only data up
    to and including bar t (the execution lag is applied later, in build_positions)."""
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    return np.sign(f - s)


# --------------------------------------------------------------------------- #
# Self-tests / demo - run: python backtest_skeleton.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))))  # random walk, no real edge

    res = run_backtest(px, ma_crossover, sizer=fixed_sizer(1.0),
                       cost_model=fixed_bps_cost(1.0, 2.0), lag=1)
    print("MA crossover on a random walk (leak-free, after costs):")
    for k, v in res["metrics"].items():
        print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")

    # walk-forward sanity: splits are ordered, gapped by the embargo, non-overlapping
    splits = list(walk_forward_splits(1000, train_size=250, test_size=50, embargo=10))
    assert splits, "expected at least one walk-forward split"
    for tr, te in splits:
        assert te[0] - tr[-1] - 1 == 10, "embargo gap between train and test is wrong"

    # --- PROOF that same-bar execution fabricates profit -------------------- #
    returns = px.pct_change(fill_method=None)
    same_bar_signal = np.sign(returns)                                  # uses the bar's OWN return (cheating)
    leaky = (build_positions(same_bar_signal, lag=0) * returns).dropna()   # lag=0 BUG -> = |returns|
    proper = (build_positions(same_bar_signal, lag=1) * returns).dropna()  # lag=1 fix -> ~noise
    sr_leaky, sr_proper = _sharpe(leaky), _sharpe(proper)
    print(f"\nLook-ahead demo:  leaky Sharpe={sr_leaky:.2f}  proper Sharpe={sr_proper:.2f}")
    assert sr_leaky > 10, "same-bar execution should look absurdly (impossibly) good"
    assert abs(sr_proper) < 1.0, "lagged execution on pure noise should have ~0 Sharpe"
    assert sr_leaky > sr_proper

    # vol_floor bounds the sizer in a near-zero-vol window (else target/realized blows up)
    calm = pd.Series([0.0] * 40 + list(rng.normal(0.0, 0.02, 60)))
    one = pd.Series(1.0, index=calm.index)
    s_floor = vol_target_sizer(target_ann_vol=0.10, lookback=20, vol_floor=0.05)(one, calm)
    assert np.isfinite(s_floor.to_numpy()).all(), "vol_floor must keep the scaler finite"
    assert s_floor.abs().max() <= 0.10 / 0.05 + 1e-9, "vol_floor should cap the scaler at target/floor"

    print("backtest_skeleton.py: all self-tests passed")
