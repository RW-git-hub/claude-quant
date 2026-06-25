"""crypto_defi.py - Crypto / DeFi quant toolkit.

Numpy / pandas / standard-library only (no scipy). Self-testing: run this file
directly to execute the assert-based checks in __main__.

Scope: 24/7 perpetual & dated futures (funding, basis, cash-and-carry),
constant-product AMM execution (price impact, impermanent loss), and a
simplified isolated-margin liquidation price.

Conventions
-----------
- Funding sign: on most perpetual venues a POSITIVE funding rate means LONGS
  PAY SHORTS. We encode payment as a SIGNED cash flow from the trader's
  perspective: negative = the trader pays, positive = the trader receives.
- Rates are per-interval (e.g. one 8-hour funding window), expressed as a
  decimal fraction of notional (0.0001 = 1 bp), unless a function name says
  otherwise.
- APR annualization uses 365 calendar days (crypto trades every day; there is
  no 252-day trading-year convention here). Returns are simple (linear) APRs,
  not APYs - no intra-year compounding is assumed.
- Prices/reserves are positive floats; notional and amounts are in quote/asset
  units consistent with the caller.
"""

from __future__ import annotations

import math
from typing import Literal

Side = Literal["long", "short"]


# --------------------------------------------------------------------------- #
# Perpetual funding
# --------------------------------------------------------------------------- #
def funding_payment(notional: float, funding_rate: float, side: Side = "long") -> float:
    """Signed funding cash flow for one interval, from the trader's perspective.

    Positive funding_rate => longs pay shorts. A long therefore has a NEGATIVE
    cash flow (cost) and a short a POSITIVE one (income); the magnitudes are
    equal: |payment| = notional * |funding_rate|.

    Parameters
    ----------
    notional : float
        Absolute position notional (price * size), >= 0.
    funding_rate : float
        Funding rate for the interval (decimal fraction, e.g. 0.0001 = 1 bp).
    side : {'long', 'short'}
        Position direction.

    Returns
    -------
    float
        Signed payment: negative = trader pays, positive = trader receives.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    sign = -1.0 if side == "long" else 1.0
    return sign * notional * funding_rate


def funding_pnl(
    notional: float,
    funding_rate: float,
    n_intervals: int,
    side: Side = "long",
) -> float:
    """Cumulative funding cash flow over ``n_intervals`` at a constant rate.

    Assumes notional and rate are held constant across intervals (a simplifying
    assumption: in practice both notional and the realized rate drift). For a
    time-varying rate, sum ``funding_payment`` per interval instead.
    """
    if n_intervals < 0:
        raise ValueError("n_intervals must be >= 0")
    return funding_payment(notional, funding_rate, side) * n_intervals


def annualized_funding(funding_rate: float, intervals_per_day: int = 3) -> float:
    """Annualize a per-interval funding rate to a simple APR.

    APR = funding_rate * intervals_per_day * 365. Default 3 intervals/day
    matches the common 8-hour funding cadence. This is a linear (non-compounded)
    annualization; realized carry compounds and will differ.
    """
    return funding_rate * intervals_per_day * 365


# --------------------------------------------------------------------------- #
# Basis / carry
# --------------------------------------------------------------------------- #
def perp_basis(perp_price: float, index_price: float) -> float:
    """Perpetual basis as a fraction: perp/index - 1.

    Positive => perp trades above its index (rich); typically coincides with
    positive funding that pulls the perp back toward the index.
    """
    if index_price <= 0:
        raise ValueError("index_price must be > 0")
    return perp_price / index_price - 1.0


def annualized_basis(futures_price: float, spot_price: float, days_to_expiry: float) -> float:
    """Annualized basis of a DATED future: (futures/spot - 1) * (365/days).

    For contango (futures > spot) this is positive - the APR an arb earns by
    shorting the future and holding spot to expiry, before costs. Diverges as
    days_to_expiry -> 0.
    """
    if spot_price <= 0:
        raise ValueError("spot_price must be > 0")
    if days_to_expiry <= 0:
        raise ValueError("days_to_expiry must be > 0")
    return (futures_price / spot_price - 1.0) * (365.0 / days_to_expiry)


def cash_and_carry_apr(funding_rate: float, intervals_per_day: int = 3) -> float:
    """APR of the delta-neutral perp cash-and-carry (short perp / long spot).

    When funding is positive the short-perp leg COLLECTS funding while the long
    spot leg neutralizes price risk, so the harvested APR is positive and equals
    the annualized funding rate. Gross of borrow, trading, and rebalancing costs,
    and of basis convergence risk if exiting before the rate normalizes.

    Note: this is the same number as ``annualized_funding`` but named for the
    strategy; the sign reads naturally (positive funding -> positive carry)
    because the short perp is on the receiving side of funding.
    """
    return annualized_funding(funding_rate, intervals_per_day)


def net_cash_and_carry_apr(
    funding_rate: float,
    borrow_apr: float = 0.0,
    fee_per_leg: float = 0.0005,
    n_legs: int = 4,
    rebalance_slippage: float = 0.0,
    holding_days: float = 365.0,
    intervals_per_day: int = 3,
) -> float:
    """Net APR of the delta-neutral perp cash-and-carry, AFTER frictions.

    Gross carry is the annualized funding harvested by the short-perp leg
    (``cash_and_carry_apr``). From it we subtract three friction APRs:

    1. Borrow cost (``borrow_apr``): the annualized rate paid to borrow the
       spot leg (or the financing on the long-spot collateral). Already an APR,
       subtracted directly.
    2. Round-trip trading fees: ``n_legs`` taker fills (default 4 = open spot,
       open perp, close spot, close perp) at ``fee_per_leg`` each. The total fee
       drag ``n_legs * fee_per_leg`` is paid ONCE over the holding period, so it
       is annualized by ``365 / holding_days`` to put it on an APR footing.
    3. Rebalance slippage: ``rebalance_slippage`` is the total slippage cost
       (fraction of notional) incurred re-hedging over the whole holding period;
       it is likewise annualized by ``365 / holding_days``.

        net_apr = gross_funding_apr
                  - borrow_apr
                  - (n_legs * fee_per_leg) * (365 / holding_days)
                  - rebalance_slippage     * (365 / holding_days)

    With any positive friction (borrow, fees, or slippage) the result is
    strictly below the gross carry. Costs are expressed as fractions of the
    (delta-neutral, roughly constant) notional; this assumes a single entry/exit
    round trip plus periodic re-hedging, and ignores convexity in the hedge.

    Causality / Iron Law: all inputs (realized funding, the borrow rate in
    effect, the fees actually paid, and realized slippage) are observed at or
    before the trade; this is an after-the-fact accounting identity with no
    forward-looking term, so it is leak-free when fed lagged/realized inputs.

    Parameters
    ----------
    funding_rate : float
        Per-interval funding rate (decimal fraction).
    borrow_apr : float
        Annualized borrow/financing cost of the spot leg (decimal APR).
    fee_per_leg : float
        Taker fee per fill (fraction of notional). Default 5 bp.
    n_legs : int
        Number of fee-bearing fills in the round trip. Default 4.
    rebalance_slippage : float
        Total re-hedge slippage over the holding period (fraction of notional).
    holding_days : float
        Holding-period length in days, used to annualize one-off costs.
    intervals_per_day : int
        Funding intervals per day for the gross-funding annualization.

    Returns
    -------
    float
        Net carry APR (simple, non-compounded).
    """
    if fee_per_leg < 0.0:
        raise ValueError("fee_per_leg must be >= 0")
    if n_legs < 0:
        raise ValueError("n_legs must be >= 0")
    if rebalance_slippage < 0.0:
        raise ValueError("rebalance_slippage must be >= 0")
    if holding_days <= 0.0:
        raise ValueError("holding_days must be > 0")
    gross = cash_and_carry_apr(funding_rate, intervals_per_day)
    annualize = 365.0 / holding_days
    fee_drag = n_legs * fee_per_leg * annualize
    slip_drag = rebalance_slippage * annualize
    return gross - borrow_apr - fee_drag - slip_drag


# --------------------------------------------------------------------------- #
# Constant-product AMM (Uniswap-v2 style)
# --------------------------------------------------------------------------- #
def amm_constant_product_out(
    reserve_in: float,
    reserve_out: float,
    amount_in: float,
    fee: float = 0.003,
) -> float:
    """Output amount of a constant-product (x*y=k) swap with a fee.

    amount_in_after_fee = amount_in * (1 - fee)
    out = reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)

    The fee stays in the pool, so the post-swap invariant k' >= k. Default
    fee=0.003 is the canonical 30 bp Uniswap-v2 tier.
    """
    if reserve_in <= 0 or reserve_out <= 0:
        raise ValueError("reserves must be > 0")
    if amount_in < 0:
        raise ValueError("amount_in must be >= 0")
    if not 0.0 <= fee < 1.0:
        raise ValueError("fee must be in [0, 1)")
    amount_in_after_fee = amount_in * (1.0 - fee)
    return reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)


def amm_price_impact(
    reserve_in: float,
    reserve_out: float,
    amount_in: float,
    fee: float = 0.003,
) -> float:
    """Price impact of a swap vs the pool spot price, as a positive fraction.

    Spot price (out per in) = reserve_out / reserve_in. The realized execution
    price (amount_out / amount_in) is always worse, so we return how much worse:

        impact = spot_price / exec_price - 1   (>= 0)

    Impact rises monotonically with ``amount_in`` (convexity of x*y=k) and with
    the fee. Returns 0.0 for a zero-size order.
    """
    if amount_in <= 0:
        return 0.0
    spot_price = reserve_out / reserve_in
    amount_out = amm_constant_product_out(reserve_in, reserve_out, amount_in, fee)
    exec_price = amount_out / amount_in
    return spot_price / exec_price - 1.0


def impermanent_loss(price_ratio: float) -> float:
    """Impermanent loss of a 50/50 constant-product LP vs holding (HODL).

    price_ratio = P_now / P_initial of the volatile asset (in the other asset).

        IL = 2*sqrt(r)/(1 + r) - 1   <= 0  for all r > 0

    Symmetric in r and 1/r, zero at r=1, and strictly negative otherwise. This
    is the divergence loss BEFORE fee income; LPs are net profitable only when
    accrued fees exceed |IL|.
    """
    if price_ratio <= 0:
        raise ValueError("price_ratio must be > 0")
    return 2.0 * math.sqrt(price_ratio) / (1.0 + price_ratio) - 1.0


def lvr_constant_product(
    sigma: float,
    pool_value: float,
    horizon_years: float,
) -> float:
    """Loss-Versus-Rebalancing (LVR) for a 50/50 constant-product AMM.

    LVR (Milionis, Moallemi, Roughgarden & Zhang, 2022, "Automated Market
    Making and Loss-Versus-Rebalancing") is the running cost an LP bears
    relative to a continuously-rebalanced reference portfolio that holds the
    same instantaneous exposure but trades at the external (CEX) price instead
    of the stale pool price. For a geometric-mean (x*y=k, 50/50) pool with
    instantaneous LVR rate ``(sigma^2 / 8) * pool_value`` per unit time, the
    expected LVR accumulated over a horizon is

        LVR = (sigma^2 / 8) * pool_value * horizon_years      (>= 0)

    where ``sigma`` is the annualized volatility of the volatile asset's price
    (in the numeraire) and the horizon is in years. Unlike impermanent loss,
    LVR is monotonically increasing in time and is the loss net of the
    arbitrageur's gain - i.e. the adverse-selection cost LPs pay to arbitrageurs.
    It is the quantity LP fee income must exceed for an LP to be profitable.

    This is the closed-form expectation for constant ``sigma`` and a pool whose
    value is held ~constant over the window; it ignores fees, the discreteness
    of arbitrage, and pool-value drift. Returned in the same units as
    ``pool_value`` (a positive cost).

    Causality / Iron Law: ``sigma`` must be a volatility estimate formed from
    information available at or before the period start (e.g. a lagged realized
    or implied vol); fed a forward-looking sigma this would leak. The formula
    itself is a forward expectation and contains no same-bar future data.

    Parameters
    ----------
    sigma : float
        Annualized volatility of the volatile asset (decimal, e.g. 0.8 = 80%).
    pool_value : float
        Total pool value in the numeraire, >= 0.
    horizon_years : float
        Length of the holding window in years, >= 0.

    Returns
    -------
    float
        Expected LVR over the horizon (>= 0), in ``pool_value`` units.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if pool_value < 0.0:
        raise ValueError("pool_value must be >= 0")
    if horizon_years < 0.0:
        raise ValueError("horizon_years must be >= 0")
    return (sigma * sigma / 8.0) * pool_value * horizon_years


# --------------------------------------------------------------------------- #
# Liquidations
# --------------------------------------------------------------------------- #
def liquidation_price(
    entry_price: float,
    leverage: float,
    maintenance_margin: float = 0.005,
    side: Side = "long",
) -> float:
    """Simplified isolated-margin liquidation price.

    long:  entry * (1 - 1/leverage + maintenance_margin)
    short: entry * (1 + 1/leverage - maintenance_margin)

    A long is liquidated when the mark falls far enough that equity hits the
    maintenance requirement; a short when it rises. This is a first-order
    approximation: it ignores funding accrual, fees, the exact venue margin
    formula (tiered maintenance, mark vs last price), and any added margin.
    Treat it as a risk sizing guide, not an exact venue trigger.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    if side == "long":
        return entry_price * (1.0 - 1.0 / leverage + maintenance_margin)
    if side == "short":
        return entry_price * (1.0 + 1.0 / leverage - maintenance_margin)
    raise ValueError(f"side must be 'long' or 'short', got {side!r}")


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _run_tests() -> None:
    # --- funding payment sign & symmetry --------------------------------- #
    long_pay = funding_payment(10_000.0, 0.0001, "long")
    short_pay = funding_payment(10_000.0, 0.0001, "short")
    assert long_pay < 0.0, "long pays when funding positive"
    assert short_pay > 0.0, "short receives when funding positive"
    assert _approx(long_pay, -short_pay), "long/short funding equal magnitude"
    assert _approx(long_pay, -1.0), "10000 * 0.0001 = 1.0 cost for long"

    # negative funding flips the signs
    assert funding_payment(10_000.0, -0.0001, "long") > 0.0
    assert funding_payment(10_000.0, -0.0001, "short") < 0.0

    # --- funding pnl is linear in intervals ------------------------------ #
    assert _approx(
        funding_pnl(10_000.0, 0.0001, 3, "long"),
        3 * funding_payment(10_000.0, 0.0001, "long"),
    )
    assert _approx(funding_pnl(10_000.0, 0.0001, 0, "long"), 0.0)

    # --- annualized funding ---------------------------------------------- #
    assert _approx(annualized_funding(0.0001, 3), 0.0001 * 3 * 365)
    assert _approx(annualized_funding(0.0001, 3), 0.1095)

    # --- perp basis ------------------------------------------------------ #
    assert _approx(perp_basis(101.0, 100.0), 0.01)
    assert _approx(perp_basis(100.0, 100.0), 0.0)
    assert perp_basis(99.0, 100.0) < 0.0

    # --- annualized basis (dated future) --------------------------------- #
    # 2% over 90 days -> ~8.11% APR
    ab = annualized_basis(102.0, 100.0, 90.0)
    assert _approx(ab, 0.02 * (365.0 / 90.0))
    assert ab > 0.0
    assert annualized_basis(98.0, 100.0, 90.0) < 0.0  # backwardation

    # --- cash and carry -------------------------------------------------- #
    assert cash_and_carry_apr(0.0001, 3) > 0.0, "positive funding -> positive carry"
    assert _approx(cash_and_carry_apr(0.0001, 3), annualized_funding(0.0001, 3))
    assert cash_and_carry_apr(-0.0001, 3) < 0.0  # negative funding -> pay to carry

    # --- net cash and carry (after frictions) ---------------------------- #
    gross = cash_and_carry_apr(0.0001, 3)  # 0.1095
    # with zero frictions, net == gross (analytic anchor)
    assert _approx(
        net_cash_and_carry_apr(
            0.0001, borrow_apr=0.0, fee_per_leg=0.0, n_legs=4, rebalance_slippage=0.0
        ),
        gross,
    )
    # any positive friction makes net strictly below gross
    net = net_cash_and_carry_apr(
        0.0001,
        borrow_apr=0.02,
        fee_per_leg=0.0005,
        n_legs=4,
        rebalance_slippage=0.001,
        holding_days=365.0,
    )
    assert net < gross, "frictions reduce net below gross"
    # explicit accounting identity over a full-year hold (annualize factor = 1):
    #   net = 0.1095 - 0.02 - 4*0.0005 - 0.001 = 0.0865
    assert _approx(net, 0.1095 - 0.02 - 4 * 0.0005 - 0.001)
    # one-off costs annualize: a 73-day hold scales fee+slip drag by 365/73 = 5x
    net_short = net_cash_and_carry_apr(
        0.0001,
        borrow_apr=0.0,
        fee_per_leg=0.0005,
        n_legs=4,
        rebalance_slippage=0.001,
        holding_days=73.0,
    )
    assert _approx(net_short, gross - (4 * 0.0005 + 0.001) * (365.0 / 73.0))
    # shorter hold -> larger annualized drag -> lower net
    assert net_short < net_cash_and_carry_apr(
        0.0001, fee_per_leg=0.0005, n_legs=4, rebalance_slippage=0.001, holding_days=365.0
    )

    # --- impermanent loss ------------------------------------------------ #
    assert _approx(impermanent_loss(1.0), 0.0), "no divergence -> no IL"
    assert _approx(impermanent_loss(2.0), -0.05719095841793653, tol=1e-9), "IL(2) ~ -5.72%"
    assert abs(impermanent_loss(2.0) - (-0.0572)) < 1e-3
    # symmetry IL(r) == IL(1/r)
    for r in (0.25, 0.5, 1.5, 2.0, 4.0, 10.0):
        assert _approx(impermanent_loss(r), impermanent_loss(1.0 / r)), f"IL symmetry at r={r}"
        assert impermanent_loss(r) <= 0.0, f"IL must be <= 0 at r={r}"
    # IL deepens with larger divergence
    assert impermanent_loss(4.0) < impermanent_loss(2.0) < 0.0

    # --- loss-versus-rebalancing (LVR) ----------------------------------- #
    # IL at no price move is exactly zero, but LVR over a round trip is positive:
    assert _approx(impermanent_loss(1.0), 0.0), "IL(1) == 0"
    lvr = lvr_constant_product(sigma=0.8, pool_value=1_000_000.0, horizon_years=1.0)
    assert lvr > 0.0, "LVR is positive over a holding window even if IL(1)=0"
    # analytic anchor: (sigma^2 / 8) * V * T = (0.64/8) * 1e6 * 1 = 80_000
    assert _approx(lvr, (0.8 ** 2 / 8.0) * 1_000_000.0 * 1.0)
    assert _approx(lvr, 80_000.0)
    # linear in horizon and in pool value; quadratic in sigma
    assert _approx(
        lvr_constant_product(0.8, 1_000_000.0, 2.0),
        2.0 * lvr_constant_product(0.8, 1_000_000.0, 1.0),
    )
    assert _approx(
        lvr_constant_product(1.6, 1_000_000.0, 1.0),
        4.0 * lvr_constant_product(0.8, 1_000_000.0, 1.0),
    )
    # degenerate cases collapse to zero cost
    assert _approx(lvr_constant_product(0.0, 1_000_000.0, 1.0), 0.0)
    assert _approx(lvr_constant_product(0.8, 1_000_000.0, 0.0), 0.0)

    # --- AMM constant-product output ------------------------------------- #
    rin, rout = 1_000_000.0, 1_000_000.0
    k_before = rin * rout
    amt = 10_000.0
    out = amm_constant_product_out(rin, rout, amt, fee=0.003)
    assert 0.0 < out < rout
    # invariant grows because fees stay in the pool: k' >= k
    rin_after = rin + amt
    rout_after = rout - out
    assert rin_after * rout_after >= k_before - 1e-6, "fees keep invariant >= k"
    # zero fee preserves invariant almost exactly (k' >= k still holds)
    out_nofee = amm_constant_product_out(rin, rout, amt, fee=0.0)
    assert (rin + amt) * (rout - out_nofee) >= k_before - 1e-3
    # output increases with amount_in
    out_small = amm_constant_product_out(rin, rout, 1_000.0)
    out_big = amm_constant_product_out(rin, rout, 100_000.0)
    assert out_small < out < out_big
    # zero-size swap returns zero
    assert _approx(amm_constant_product_out(rin, rout, 0.0), 0.0)

    # --- AMM price impact ------------------------------------------------ #
    pi_small = amm_price_impact(rin, rout, 1_000.0)
    pi_mid = amm_price_impact(rin, rout, 10_000.0)
    pi_big = amm_price_impact(rin, rout, 100_000.0)
    assert pi_small > 0.0, "any non-zero swap has positive impact (incl. fee)"
    assert pi_small < pi_mid < pi_big, "impact grows with size"
    assert _approx(amm_price_impact(rin, rout, 0.0), 0.0)
    # higher fee -> higher impact at fixed size
    assert amm_price_impact(rin, rout, 10_000.0, fee=0.01) > amm_price_impact(
        rin, rout, 10_000.0, fee=0.003
    )

    # --- liquidation price ----------------------------------------------- #
    liq_long = liquidation_price(100.0, 10.0, maintenance_margin=0.005, side="long")
    assert _approx(liq_long, 100.0 * (1 - 0.1 + 0.005))
    assert _approx(liq_long, 90.5)
    assert abs(liq_long - 100.0 * 0.905) < 1e-9
    liq_short = liquidation_price(100.0, 10.0, maintenance_margin=0.005, side="short")
    assert _approx(liq_short, 100.0 * (1 + 0.1 - 0.005))
    assert liq_long < 100.0 < liq_short, "long liquidates below, short above entry"
    # higher leverage -> liquidation closer to entry
    assert liquidation_price(100.0, 20.0, side="long") > liquidation_price(
        100.0, 10.0, side="long"
    )

    print("crypto_defi.py: all self-tests passed.")


if __name__ == "__main__":
    _run_tests()
