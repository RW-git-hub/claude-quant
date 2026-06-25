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
def borrow_cost(
    short_notional: float,
    annual_rate: float,
    days: float,
    day_count: float = 365.0,
) -> float:
    """Stock-borrow financing cost in DOLLARS for holding a short.

    Formula (ACT/``day_count``, simple, no compounding)
    ---------------------------------------------------
        cost = short_notional * annual_rate * days / day_count

    ``day_count`` selects the day-count basis for the annualized rate. The
    default 365.0 is ACT/365 (the usual equity-borrow convention). Pass 360.0
    for ACT/360 (money-market basis used by many financing desks); the same
    annual_rate over the same calendar ``days`` then costs MORE, because a
    360-day year amortizes the annual fee over fewer days
    (``days/360 > days/365``).

    Parameters
    ----------
    short_notional : absolute dollar size of the short position (>= 0).
    annual_rate    : annualized borrow fee as a fraction (e.g. 0.05 == 5%/yr).
                     Hard-to-borrow names can be tens of percent.
    days           : holding period in calendar days.
    day_count      : days-per-year basis for the rate (365.0 default, or 360.0
                     for ACT/360). Must be > 0.

    Returns
    -------
    Borrow cost in dollars (positive == cost paid).

    Raises
    ------
    ValueError : if ``day_count <= 0``.
    """
    if not day_count > 0:
        raise ValueError(f"day_count must be > 0 (got {day_count!r}).")
    return short_notional * annual_rate * days / day_count


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


def cost_sweep(
    returns: pd.Series,
    turnover: pd.Series,
    bps_grid: ArrayLike,
    periods_per_year: float = 252.0,
) -> pd.Series:
    """Net annualized Sharpe at each per-turnover cost level (a cost sensitivity).

    For every cost level ``c`` in ``bps_grid`` (basis points charged per unit of
    turnover), recompute net returns ``net_t = gross_t - turnover_t * c/1e4`` and
    report the annualized Sharpe of that net series. This shows how fast the edge
    decays as you make the cost assumption more pessimistic -- a strategy whose
    Sharpe collapses by 10 bps is fragile to execution.

    Sharpe = mean(net)/std(net) * sqrt(periods_per_year), using the SAME sample
    std (ddof=1, pandas default) at every grid point so the comparison is clean.
    Because raising ``c`` only ever subtracts more from each period's return, the
    net Sharpe is non-increasing in ``c`` for a non-negative turnover series.

    Leak-safety
    -----------
    Causal: each net return at t uses only gross[t] and turnover[t]. Lag/align
    ``turnover`` to the period in which trades actually execute BEFORE calling
    (same requirement as ``apply_costs``); this function does no shifting.

    Parameters
    ----------
    returns          : per-period gross strategy returns (fractions).
    turnover         : per-period turnover (fraction of book traded), same index.
    bps_grid         : iterable of per-turnover costs in BPS (e.g. [0, 5, 10, 20]).
    periods_per_year : annualization factor (252 daily, 52 weekly, 12 monthly).

    Returns
    -------
    pd.Series of net annualized Sharpe indexed by the bps level (float index).

    Raises
    ------
    ValueError : if ``periods_per_year <= 0``.
    """
    if not periods_per_year > 0:
        raise ValueError(
            f"periods_per_year must be > 0 (got {periods_per_year!r})."
        )
    returns, turnover = returns.align(turnover, join="left")
    turnover = turnover.fillna(0.0)
    ann = math.sqrt(periods_per_year)
    grid = np.asarray(bps_grid, dtype=float)
    out = {}
    for bps in grid:
        net = returns - turnover * (bps / BPS_PER_UNIT)
        sd = net.std()  # ddof=1
        if not sd > 0:
            sharpe = 0.0 if net.mean() == 0.0 else float("inf") * np.sign(net.mean())
        else:
            sharpe = net.mean() / sd * ann
        out[float(bps)] = sharpe
    return pd.Series(out, name="net_sharpe")


def capacity_curve(
    aum_grid: ArrayLike,
    gross_ann_return: float,
    annual_turnover: float,
    adv_dollars: float,
    daily_vol: float,
    impact_coef: float = 1.0,
    fixed_cost_bps: float = 0.0,
) -> pd.Series:
    """Hump-shaped net annual DOLLAR profit vs AUM, re-costing sqrt impact.

    Models the classic capacity trade-off. Gross alpha (as a fraction of capital)
    is roughly scale-invariant, so gross dollar profit grows LINEARLY in AUM. But
    each rebalance must trade a larger fraction of ADV as the book grows, so the
    square-root impact charged per unit turnover GROWS like ``sqrt(AUM)`` and the
    dollar impact bill grows like ``AUM^1.5`` -- super-linearly. Net dollar profit
    therefore RISES (alpha dominates) then FALLS (impact dominates): a hump with a
    single interior optimum, after which adding capital DESTROYS dollar profit.

    Returning DOLLARS (not a return fraction) is what produces the hump: as a
    fraction of capital, net return is monotone-decreasing in AUM, but the dollar
    curve -- the quantity a fund actually maximizes -- is concave with an interior
    peak. See Kyle/Obizhaeva sqrt-impact capacity intuition.

    Per-AUM construction
    --------------------
    Let ``A`` be AUM in dollars. Annual notional traded is ``annual_turnover*A``.
    Participation of a representative rebalance trade is ``A / adv_dollars``::

        impact_per_turn = impact_coef * daily_vol * sqrt(A / adv_dollars)   # frac
        cost_frac       = annual_turnover * (fixed_cost_bps/1e4 + impact_per_turn)
        net_dollar_pnl  = A * (gross_ann_return - cost_frac)

    Closed form for the peak (ignoring the linear fixed term, which only shifts
    the height): with ``k = annual_turnover*impact_coef*daily_vol/sqrt(adv)`` the
    net PnL is ``A*g - k*A^1.5``; ``d/dA = 0`` at ``A* = (2g/(3k))**2``, a single
    interior maximum whenever ``g > 0`` and ``k > 0``.

    Leak-safety
    -----------
    Pure cross-sectional re-costing of a single gross-alpha assumption; no time
    series, no look-ahead. ``gross_ann_return`` must itself be an honest,
    cost-free estimate produced upstream without leakage.

    Parameters
    ----------
    aum_grid         : iterable of book sizes in DOLLARS (must be > 0).
    gross_ann_return : gross annual return as a fraction (scale-invariant alpha).
    annual_turnover  : annual turnover (notional traded / capital, e.g. 5.0).
    adv_dollars      : average daily volume in DOLLARS for the traded universe.
    daily_vol        : daily return volatility as a fraction (e.g. 0.02 == 2%).
    impact_coef      : square-root impact calibration constant (default 1.0).
    fixed_cost_bps   : size-independent per-turnover cost in bps (commission +
                       half-spread), default 0.0.

    Returns
    -------
    pd.Series of net annual DOLLAR profit indexed by AUM (float index).

    Raises
    ------
    ValueError : if ``adv_dollars <= 0`` or any AUM <= 0.
    """
    if not adv_dollars > 0:
        raise ValueError(
            f"adv_dollars must be > 0 (got {adv_dollars!r})."
        )
    aum = np.asarray(aum_grid, dtype=float)
    if not np.all(aum > 0):
        raise ValueError("all aum_grid values must be > 0.")
    fixed = fixed_cost_bps / BPS_PER_UNIT
    out = {}
    for a in aum:
        participation = a / adv_dollars
        impact_per_turn = impact_coef * daily_vol * math.sqrt(participation)
        cost_frac = annual_turnover * (fixed + impact_per_turn)
        out[float(a)] = a * (gross_ann_return - cost_frac)
    return pd.Series(out, name="net_dollar_pnl")


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
    # day_count: default is ACT/365 and unchanged by the new param.
    assert abs(borrow_cost(1e6, 0.05, 365) - borrow_cost(1e6, 0.05, 365, day_count=365)) < 1e-6
    # ACT/360 costs MORE than ACT/365 for the same rate/days (365/360 ratio).
    assert borrow_cost(1e6, 0.05, 30, day_count=360) > borrow_cost(1e6, 0.05, 30, day_count=365)
    # analytic: 1e6*0.05*360/360 = 50,000 over a full 360-day "year".
    assert abs(borrow_cost(1e6, 0.05, 360, day_count=360) - 50_000.0) < 1e-6
    # exact ratio between bases is 365/360.
    r360 = borrow_cost(1e6, 0.05, 30, day_count=360)
    r365 = borrow_cost(1e6, 0.05, 30, day_count=365)
    assert abs(r360 / r365 - 365.0 / 360.0) < 1e-12
    # day_count guard.
    for bad_dc in (0.0, -360.0):
        try:
            borrow_cost(1e6, 0.05, 30, day_count=bad_dc)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for day_count={bad_dc}")

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

    # --- cost_sweep: net Sharpe is non-increasing as bps rise ----------------
    rs = np.random.RandomState(0)
    cs_idx = pd.date_range("2024-01-01", periods=500, freq="D")
    # Positive-drift gross returns so the zero-cost Sharpe is clearly > 0.
    cs_gross = pd.Series(0.0008 + 0.01 * rs.randn(500), index=cs_idx)
    cs_turn = pd.Series(0.5 + 0.1 * np.abs(rs.randn(500)), index=cs_idx)  # >0
    bps_grid = [0.0, 5.0, 10.0, 20.0, 50.0]
    sweep = cost_sweep(cs_gross, cs_turn, bps_grid, periods_per_year=252.0)
    assert list(sweep.index) == [float(b) for b in bps_grid]
    # strictly decreasing here (turnover strictly positive, mean drift positive).
    sv = sweep.values
    assert all(b < a for a, b in zip(sv, sv[1:])), sv
    # zero-bps row equals the raw annualized Sharpe of the gross series.
    raw_sharpe = cs_gross.mean() / cs_gross.std() * math.sqrt(252.0)
    assert abs(sweep.loc[0.0] - raw_sharpe) < 1e-12
    # analytic single-point check against apply_costs at 10 bps.
    net10 = apply_costs(cs_gross, cs_turn, 10.0 / BPS_PER_UNIT)
    sharpe10 = net10.mean() / net10.std() * math.sqrt(252.0)
    assert abs(sweep.loc[10.0] - sharpe10) < 1e-12
    # periods_per_year guard.
    try:
        cost_sweep(cs_gross, cs_turn, bps_grid, periods_per_year=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for periods_per_year=0")

    # --- capacity_curve: hump-shaped net dollar PnL (rises then falls) -------
    # g=0.15, turnover=10, adv=1e8, dvol=0.02, coef=1, fixed=0 give analytic
    #   k = 10*1*0.02/sqrt(1e8) = 2e-5 ; A* = (2g/(3k))^2 = (0.3/6e-5)^2 = 2.5e7.
    g_, to_, adv_, dv_ = 0.15, 10.0, 1e8, 0.02
    aum_grid = [1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 1e9]
    cap = capacity_curve(
        aum_grid,
        gross_ann_return=g_,
        annual_turnover=to_,
        adv_dollars=adv_,
        daily_vol=dv_,
        impact_coef=1.0,
        fixed_cost_bps=0.0,
    )
    assert list(cap.index) == [float(a) for a in aum_grid]
    cv = cap.values
    # A hump: argmax is interior, non-decreasing up to the peak, then decreasing.
    peak = int(np.argmax(cv))
    assert 0 < peak < len(cv) - 1, (peak, cv)
    assert all(cv[i] <= cv[i + 1] + TOL for i in range(peak)), cv
    assert all(cv[i] > cv[i + 1] for i in range(peak, len(cv) - 1)), cv
    # The analytic optimum 2.5e7 is on the grid and is the peak.
    assert cap.index[peak] == 2.5e7
    # Analytic value at A=adv=1e8: participation=1, impact_per_turn=0.02,
    #   cost_frac = 10*0.02 = 0.20 ; net = 1e8*(0.15-0.20) = -5e6.
    assert abs(cap.loc[1e8] - 1e8 * (0.15 - 10.0 * 0.02)) < 1e-3
    # Past capacity the curve goes negative (impact bill exceeds gross alpha).
    assert cap.iloc[-1] < 0.0
    # fixed_cost_bps only lowers the curve, never raises it.
    cap_fixed = capacity_curve(
        aum_grid, g_, to_, adv_, dv_, impact_coef=1.0, fixed_cost_bps=5.0
    )
    assert (cap_fixed.values <= cv + TOL).all()
    # guards.
    try:
        capacity_curve([1e6], 0.15, 10.0, 0.0, 0.02)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for adv_dollars=0")
    try:
        capacity_curve([1e6, -1.0], 0.15, 10.0, 1e8, 0.02)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive aum")

    print("costs.py: all self-tests passed.")
