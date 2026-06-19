"""costs.py -- Transaction-cost & frictions toolkit for quant backtests.

Scope: COST MODELS & CAPACITY only. Execution scheduling (TWAP/VWAP/Almgren
trajectory optimization) is intentionally out of scope here -- it belongs to a
separate execution-algorithms module.

Convention (read this first)
----------------------------
Unless a function name ends in ``_cost`` and is documented as DOLLARS, every
cost in this module is expressed as a RETURN FRACTION of traded notional:

    cost_fraction = dollars_paid / notional_traded

So 1 bp = 1e-4 as a fraction. A cost fraction is subtracted directly from a
simple (arithmetic) return earned on that notional. This keeps costs additive
with simple returns and composable with turnover.

Why fractions (not bps, not dollars) as the core unit:
  - Backtests carry returns as fractions; net = gross - cost_fraction is exact.
  - Notional-independent, so the same number applies at any book size until you
    hit the capacity limits that the impact terms encode.

Two functions return DOLLARS because they are inherently financing flows tied
to a *held* position over time rather than a per-trade slippage on notional:
``borrow_cost`` (short borrow fee) and ``funding_cost`` (perp funding). They are
clearly documented and named so they are not confused with the fraction API.

Cost components modeled
-----------------------
  commission      : exchange/broker fee, ~constant in bps of notional.
  half-spread     : you cross half the quoted bid/ask on a marketable order.
  market impact   : price moves against you as you consume liquidity. Two forms:
                    - square-root (concave, the empirical workhorse), and
                    - linear (simple, conservative for small participation).
  borrow          : cost of borrowing stock to short (annualized, dollar flow).
  funding         : perpetual-swap funding paid/received by the position holder.

Participation
-------------
``participation = order_size / ADV`` (fraction of average daily volume). Impact
models scale in participation. ``order_size`` and ``adv`` must be in the SAME
units (shares, contracts, or notional -- just be consistent); only their ratio
matters.

Detect / fix pitfalls
---------------------
  * Detect: applying impact as a flat bps regardless of size -> understates cost
    for large orders, overstates capacity. Fix: use ``square_root_impact`` with
    a per-asset ``daily_vol`` and ``coef`` calibrated to fills.
  * Detect: ADV == 0 (illiquid / halted name) producing inf/NaN impact. Fix:
    impact functions guard ``adv > 0`` and raise on non-positive ADV.
  * Detect: double-counting the spread (charging full spread AND a separate
    half-spread). Fix: ``slippage_total`` charges exactly one half-spread plus
    impact; commissions are added separately by the caller.
  * Detect: subtracting cost twice or on un-lagged turnover. Fix: feed
    ``apply_costs`` a turnover series aligned to the period in which the trade
    actually happens.

All functions are pure and deterministic.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[float, np.ndarray, pd.Series]

BPS_PER_UNIT: float = 1e4  # 1.0 (=100%) == 10,000 bps; 1 bp == 1e-4.


# --------------------------------------------------------------------------- #
# Per-trade slippage components (return fractions of notional)
# --------------------------------------------------------------------------- #
def commission_return(fee_bps: ArrayLike) -> ArrayLike:
    """Commission as a return fraction of notional.

    Parameters
    ----------
    fee_bps : commission in basis points of traded notional (e.g. 5 == 5 bps).

    Returns
    -------
    Return fraction = ``fee_bps / 1e4`` (so 5 bps -> 5e-4).
    """
    return fee_bps / BPS_PER_UNIT


def half_spread_cost(half_spread_bps: ArrayLike) -> ArrayLike:
    """Cost of crossing half the bid/ask spread, as a return fraction.

    A marketable (liquidity-taking) order pays roughly half the quoted spread
    relative to the mid. Pass the HALF-spread in bps; this just converts to a
    fraction. (If you only have the full spread S bps, pass S/2 here.)

    Parameters
    ----------
    half_spread_bps : half the quoted spread in bps of notional.

    Returns
    -------
    Return fraction = ``half_spread_bps / 1e4`` (so 2 bps -> 2e-4).
    """
    return half_spread_bps / BPS_PER_UNIT


def square_root_impact(
    order_size: float,
    adv: float,
    daily_vol: float,
    coef: float = 1.0,
) -> float:
    """Square-root market-impact model, as a return fraction of notional.

    Formula
    -------
        participation = order_size / adv
        impact        = coef * daily_vol * sqrt(participation)

    The square-root (concave) law is the standard empirical form: impact grows
    sub-linearly in size, scaled by the asset's own daily return volatility.
    ``coef`` (O(1)) is calibrated per market/venue from realized fills.

    Parameters
    ----------
    order_size : size of the order (shares/contracts/notional).
    adv        : average daily volume in the SAME units as ``order_size``.
    daily_vol  : daily return volatility as a fraction (e.g. 0.02 == 2%/day).
    coef       : dimensionless calibration constant (default 1.0).

    Returns
    -------
    Impact as a return fraction (>= 0 for sane inputs).

    Raises
    ------
    ValueError : if ``adv <= 0`` (impact undefined for zero-liquidity names).
    """
    if not adv > 0:
        raise ValueError(f"adv must be > 0 (got {adv!r}); impact is undefined.")
    participation = order_size / adv
    return coef * daily_vol * math.sqrt(participation)


def linear_impact(participation: ArrayLike, coef: float = 0.1) -> ArrayLike:
    """Linear market-impact model, as a return fraction of notional.

    Formula
    -------
        impact = coef * participation

    Linear-in-participation impact is simple and conservative; reasonable at low
    participation but it OVERSTATES cost for very large orders relative to the
    square-root law. Prefer ``square_root_impact`` when calibration is available.

    Parameters
    ----------
    participation : order_size / ADV (fraction of average daily volume).
    coef          : impact per unit participation (default 0.1 == 10 bps at
                    100% of ADV is 0.1*1.0 = 0.10).

    Returns
    -------
    Impact as a return fraction.
    """
    return coef * participation


def slippage_total(
    participation: float,
    daily_vol: float,
    half_spread_bps: float,
    impact_coef: float = 1.0,
) -> float:
    """Total per-trade slippage = half-spread + square-root impact (fraction).

    Combines exactly ONE half-spread term and the square-root impact term.
    Commissions are NOT included here -- add ``commission_return`` separately so
    fixed fees and variable slippage stay distinguishable.

    Because impact is expressed via ``participation`` directly, this uses
    ``sqrt(participation)`` (equivalent to ``square_root_impact`` with
    order_size/adv = participation).

    Formula
    -------
        half_spread = half_spread_bps / 1e4
        impact      = impact_coef * daily_vol * sqrt(participation)
        total       = half_spread + impact

    Parameters
    ----------
    participation   : order_size / ADV.
    daily_vol       : daily return volatility as a fraction.
    half_spread_bps : half the quoted spread in bps.
    impact_coef     : square-root impact calibration constant (default 1.0).

    Returns
    -------
    Total slippage as a return fraction.

    Raises
    ------
    ValueError : if ``participation < 0``.
    """
    if participation < 0:
        raise ValueError(f"participation must be >= 0 (got {participation!r}).")
    hs = half_spread_cost(half_spread_bps)
    impact = impact_coef * daily_vol * math.sqrt(participation)
    return hs + impact


# --------------------------------------------------------------------------- #
# Financing / carry costs (DOLLARS -- not return fractions)
# --------------------------------------------------------------------------- #
def borrow_cost(short_notional: float, annual_rate: float, days: float) -> float:
    """Stock-borrow financing cost in DOLLARS for holding a short.

    Formula (ACT/365, simple, no compounding)
    -----------------------------------------
        cost = short_notional * annual_rate * days / 365

    Parameters
    ----------
    short_notional : absolute dollar size of the short position (>= 0).
    annual_rate    : annualized borrow fee as a fraction (e.g. 0.05 == 5%/yr).
                     Hard-to-borrow names can be tens of percent.
    days           : holding period in calendar days.

    Returns
    -------
    Borrow cost in dollars (positive == cost paid).
    """
    return short_notional * annual_rate * days / 365.0


def funding_cost(notional: float, funding_rate: float, n_intervals: int) -> float:
    """Perpetual-swap funding cost in DOLLARS (signed) over n funding intervals.

    Convention
    ----------
        cost = notional * funding_rate * n_intervals

    where ``notional`` is SIGNED (long > 0, short < 0) and ``funding_rate`` is
    the per-interval rate. With the standard perp convention, when funding is
    POSITIVE, longs PAY shorts; so a long (notional > 0) at a positive rate
    yields a positive number == a COST to the long. A short (notional < 0) at a
    positive rate yields a negative number == funding RECEIVED. Flip signs when
    funding is negative.

    Parameters
    ----------
    notional     : signed position notional in dollars (long +, short -).
    funding_rate : per-interval funding rate as a fraction (e.g. 0.0001).
    n_intervals  : number of funding intervals held (e.g. 3 for 24h at 8h).

    Returns
    -------
    Signed funding flow in dollars (positive == net cost to the position).
    """
    return notional * funding_rate * n_intervals


# --------------------------------------------------------------------------- #
# Capacity / portfolio-level helpers
# --------------------------------------------------------------------------- #
def breakeven_cost_bps(gross_ann_return: float, annual_turnover: float) -> float:
    """Per-trade cost (bps) at which a strategy's net annual return hits zero.

    Formula
    -------
        breakeven_bps = gross_ann_return / annual_turnover * 1e4

    Interpretation: with ``annual_turnover`` round-trips-equivalent of trading
    per year, total annual cost = annual_turnover * cost_per_trade. Setting that
    equal to ``gross_ann_return`` and solving gives the cost (as a fraction) the
    strategy can absorb before its edge is gone; *1e4 converts to bps. If your
    realistic estimated per-trade cost exceeds this, the strategy is uninvestable
    at that turnover.

    Parameters
    ----------
    gross_ann_return : gross annual return as a fraction (e.g. 0.10 == 10%).
    annual_turnover  : annual turnover in units of notional traded / capital
                       (e.g. 5.0 == trade 5x the book per year).

    Returns
    -------
    Breakeven per-trade cost in basis points.

    Raises
    ------
    ValueError : if ``annual_turnover <= 0``.
    """
    if not annual_turnover > 0:
        raise ValueError(
            f"annual_turnover must be > 0 (got {annual_turnover!r})."
        )
    return gross_ann_return / annual_turnover * BPS_PER_UNIT


def apply_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_per_turnover: float,
) -> pd.Series:
    """Subtract trading costs from gross returns, elementwise.

    Formula
    -------
        net_t = gross_t - turnover_t * cost_per_turnover

    ``turnover_t`` is the fraction of the book traded in period t (sum of abs
    position changes), and ``cost_per_turnover`` is the per-unit-turnover cost as
    a return fraction (e.g. ``slippage_total`` + ``commission_return``). The two
    Series must be aligned on the same index; align/lag turnover to the period in
    which trades actually execute BEFORE calling this.

    Parameters
    ----------
    gross_returns     : per-period gross strategy returns (fractions).
    turnover          : per-period turnover (fraction of book traded), same index.
    cost_per_turnover : cost per unit turnover, as a return fraction.

    Returns
    -------
    Net return Series (same index as ``gross_returns``).
    """
    gross_returns, turnover = gross_returns.align(turnover, join="left")
    turnover = turnover.fillna(0.0)
    return gross_returns - turnover * cost_per_turnover


# --------------------------------------------------------------------------- #
# Self-tests: analytic / synthetic cases. Run: python costs.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    TOL = 1e-12

    # --- commission_return: 5 bps -> 5e-4 -------------------------------------
    assert abs(commission_return(5) - 5e-4) < TOL
    assert abs(commission_return(0) - 0.0) < TOL
    assert abs(commission_return(100) - 1e-2) < TOL

    # --- half_spread_cost: 2 bps -> 2e-4 -------------------------------------
    assert abs(half_spread_cost(2) - 2e-4) < TOL
    assert abs(half_spread_cost(10) - 1e-3) < TOL

    # --- square_root_impact: monotone increasing in order_size ---------------
    adv, dvol, c = 1_000_000.0, 0.02, 1.0
    sizes = [1_000.0, 10_000.0, 50_000.0, 100_000.0, 250_000.0]
    impacts = [square_root_impact(s, adv, dvol, c) for s in sizes]
    assert all(b > a for a, b in zip(impacts, impacts[1:])), impacts
    assert all(x >= 0.0 for x in impacts)

    # --- square_root_impact: sqrt scaling, 4x size -> 2x impact --------------
    base = square_root_impact(10_000.0, adv, dvol, c)
    quad = square_root_impact(40_000.0, adv, dvol, c)
    assert abs(quad / base - 2.0) < 1e-9, (base, quad, quad / base)
    # 9x size -> 3x impact, as a further check of the sqrt law.
    nine = square_root_impact(90_000.0, adv, dvol, c)
    assert abs(nine / base - 3.0) < 1e-9

    # --- square_root_impact: explicit analytic value -------------------------
    # participation = 1e6/1e6 = 1.0 ; impact = 1.0*0.02*sqrt(1)=0.02
    assert abs(square_root_impact(1_000_000.0, 1_000_000.0, 0.02, 1.0) - 0.02) < TOL

    # --- square_root_impact: guards adv > 0 ----------------------------------
    for bad_adv in (0.0, -1.0):
        try:
            square_root_impact(1_000.0, bad_adv, 0.02)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for adv={bad_adv}")

    # --- linear_impact: linear in participation ------------------------------
    assert abs(linear_impact(0.0, 0.1) - 0.0) < TOL
    assert abs(linear_impact(1.0, 0.1) - 0.1) < TOL
    assert abs(linear_impact(0.5, 0.2) - 0.1) < TOL
    # 2x participation -> 2x impact (exact linearity).
    assert abs(linear_impact(0.2, 0.1) * 2.0 - linear_impact(0.4, 0.1)) < TOL

    # --- slippage_total: half-spread + sqrt impact ---------------------------
    # participation=0.04, dvol=0.02, half_spread=2bps, coef=1.0
    #   half_spread = 2e-4
    #   impact      = 1.0*0.02*sqrt(0.04) = 0.02*0.2 = 0.004
    #   total       = 0.0042
    st = slippage_total(0.04, 0.02, 2.0, impact_coef=1.0)
    assert abs(st - (2e-4 + 0.02 * math.sqrt(0.04))) < TOL
    assert abs(st - 0.0042) < TOL
    # zero participation -> only the half-spread remains.
    assert abs(slippage_total(0.0, 0.02, 2.0) - 2e-4) < TOL
    # negative participation rejected.
    try:
        slippage_total(-0.1, 0.02, 2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative participation")

    # --- borrow_cost: full year and prorated --------------------------------
    assert abs(borrow_cost(1e6, 0.05, 365) - 50_000.0) < 1e-6
    assert abs(borrow_cost(1e6, 0.05, 73) - 10_000.0) < 1e-6  # 73/365 = 0.2
    assert abs(borrow_cost(0.0, 0.05, 365) - 0.0) < TOL
    # linear in days: half the days -> half the cost.
    assert abs(borrow_cost(1e6, 0.05, 182.5) - 25_000.0) < 1e-6

    # --- funding_cost: sign conventions --------------------------------------
    # Positive funding + LONG notional => positive => a COST to the long.
    assert funding_cost(1e6, 0.0001, 3) > 0.0
    assert abs(funding_cost(1e6, 0.0001, 3) - 300.0) < 1e-9
    # Positive funding + SHORT notional => negative => funding RECEIVED.
    assert funding_cost(-1e6, 0.0001, 3) < 0.0
    # Negative funding + LONG => long receives (negative cost).
    assert funding_cost(1e6, -0.0001, 3) < 0.0
    # Zero intervals => no flow.
    assert abs(funding_cost(1e6, 0.0001, 0) - 0.0) < TOL

    # --- breakeven_cost_bps: analytic ---------------------------------------
    # 10% gross / 5x turnover -> 0.10/5*1e4 = 200 bps.
    assert abs(breakeven_cost_bps(0.10, 5.0) - 200.0) < 1e-9
    assert abs(breakeven_cost_bps(0.20, 20.0) - 100.0) < 1e-9
    # higher turnover -> lower breakeven (less cost tolerated per trade).
    assert breakeven_cost_bps(0.10, 50.0) < breakeven_cost_bps(0.10, 5.0)
    try:
        breakeven_cost_bps(0.10, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for zero turnover")

    # --- apply_costs: net = gross - turnover*cost, elementwise ----------------
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    gross = pd.Series([0.010, -0.005, 0.000, 0.020, -0.010], index=idx)
    turn = pd.Series([1.0, 0.5, 0.0, 2.0, 1.0], index=idx)
    cpt = 0.0010  # 10 bps per unit turnover
    net = apply_costs(gross, turn, cpt)
    expected = gross - turn * cpt
    assert np.allclose(net.values, expected.values, atol=TOL)
    # spot-check one element: 0.020 - 2.0*0.0010 = 0.018
    assert abs(net.iloc[3] - 0.018) < TOL
    # zero turnover row is unchanged.
    assert abs(net.iloc[2] - gross.iloc[2]) < TOL
    # costs never increase returns.
    assert (net <= gross + TOL).all()
    # zero cost -> net == gross exactly.
    assert np.allclose(apply_costs(gross, turn, 0.0).values, gross.values, atol=TOL)
    # index preserved.
    assert net.index.equals(gross.index)

    print("costs.py: all self-tests passed.")
