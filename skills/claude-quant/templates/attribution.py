"""attribution.py - performance attribution toolkit.

Decomposes a portfolio's RETURN relative to a benchmark into interpretable,
*additive* pieces. numpy / pandas / stdlib ONLY (no scipy / sklearn /
statsmodels).

What lives here
---------------
- brinson_fachler  : single-period sector allocation / selection / interaction
                     that sum EXACTLY to the total active return.
- carino_link      : multi-period linking of single-period arithmetic active
                     returns into a set of contributions that sum EXACTLY to the
                     geometrically-compounded total active return (Carino 1999),
                     with a guarded 0/0 limit.
- factor_attribution : decompose a portfolio return series into factor
                     contributions (beta * factor return) plus alpha + residual
                     via a plain OLS time-series regression.
- implementation_shortfall : Perold (1988) IS waterfall from a paper/decision
                     price to the realized average fill, split into delay,
                     trading (impact/timing) and opportunity (missed-trade) cost.

Conventions
-----------
- All returns are *simple* periodic returns (a +1% return is 0.01). Simple
  returns compound multiplicatively across time: 1 + R = prod(1 + r_t); they add
  across holdings within a period: R_p = sum_i w_i * r_i.
- Active (a.k.a. excess) return is ARITHMETIC by default: a = R_p - R_b. The
  arithmetic active return does NOT compound additively across periods - that is
  exactly the gap Carino linking closes (Section: carino_link).
- Weights w_p / w_b are the *beginning-of-period* holding weights (each sums to
  1 within a period for a fully-invested book). Brinson-Fachler is a
  single-period model; multi-period Brinson must be Carino- (or Menchero-)
  linked, never naively summed.

Iron-Law / leakage notes
------------------------
- Every function here is a DIAGNOSTIC computed on REALIZED returns over a CLOSED
  period: it consumes only data from inside the attribution window and produces
  no forward-looking signal, so there is nothing to lag. None of these outputs
  may be fed back into position sizing for the SAME period (that would use the
  realized return of the period to set that period's weights) - use them only
  ex-post or with a strict one-period lag if you build a signal from them.
- factor_attribution runs an IN-SAMPLE OLS over the whole supplied window; its
  betas are fit on the same period they explain, so they are descriptive, not a
  causal/tradeable forecast. For a tradeable factor exposure, estimate beta on a
  trailing window ending strictly before the period being attributed.

References
----------
- Brinson, Hood & Beebower (1986); Brinson & Fachler (1986), "Measuring
  Non-US Equity Portfolio Performance" - the allocation/selection/interaction
  decomposition (Fachler uses (r_b_i - R_b) in the allocation term so a benchmark
  overweight is only rewarded for beating the TOTAL benchmark).
- Carino, D. (1999), "Combining Attribution Effects Over Time", Journal of
  Performance Measurement - the logarithmic linking coefficient used here.
- Perold, A. (1988), "The Implementation Shortfall: Paper versus Reality",
  Journal of Portfolio Management - the IS cost waterfall.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

_EPS = 1e-12  # below this an active spread / denominator is treated as exactly 0


# --------------------------------------------------------------------------- #
# Brinson-Fachler single-period attribution
# --------------------------------------------------------------------------- #
def brinson_fachler(w_p, w_b, r_p, r_b, sectors=None) -> Dict[str, object]:
    """Single-period Brinson-Fachler sector attribution.

    Splits the total ARITHMETIC active return  A = R_p - R_b  into three additive
    effects per sector i:

        allocation_i  = (w_p_i - w_b_i) * (r_b_i - R_b)
        selection_i   =  w_b_i          * (r_p_i - r_b_i)
        interaction_i = (w_p_i - w_b_i) * (r_p_i - r_b_i)

    where  R_p = sum_i w_p_i r_p_i ,  R_b = sum_i w_b_i r_b_i  are the total
    portfolio and benchmark returns. The three terms sum EXACTLY to the total
    active return; this identity is asserted inside the function.

    Why Fachler (not plain Brinson-Hood-Beebower): the allocation term uses
    (r_b_i - R_b), the sector benchmark return relative to the TOTAL benchmark.
    Overweighting a sector is credited as allocation skill only if that sector
    beat the overall benchmark - the economically correct interpretation. (Plain
    BHB uses r_b_i and pushes the difference into a residual.)

    Algebraic identity (per sector, before grouping):
        w_p r_p - w_b r_b
          = (w_p - w_b)(r_b - R_b)        # allocation (Fachler-centered)
          +  w_b      (r_p - r_b)         # selection
          + (w_p - w_b)(r_p - r_b)        # interaction
          + (w_p - w_b) R_b
    Summed over i the last cross term vanishes because
    sum_i (w_p_i - w_b_i) = 1 - 1 = 0 (both books fully invested), recovering
    sum_i(w_p r_p) - sum_i(w_b r_b) = R_p - R_b exactly.

    Parameters
    ----------
    w_p, w_b : array-like portfolio / benchmark sector weights (each summing to
        ~1 for a fully-invested book; a small cash residual is tolerated and
        shows up faithfully in the totals).
    r_p, r_b : array-like portfolio / benchmark per-sector returns.
    sectors  : optional labels; if given the per-sector frame is indexed by them.

    Returns
    -------
    dict with:
      'by_sector' : DataFrame indexed by sector with columns
                    [allocation, selection, interaction, total].
      'allocation', 'selection', 'interaction' : floats (summed effects).
      'active_return' : float, R_p - R_b.
      'R_p', 'R_b'    : floats, the total returns.
    """
    wp = np.asarray(w_p, dtype=float).ravel()
    wb = np.asarray(w_b, dtype=float).ravel()
    rp = np.asarray(r_p, dtype=float).ravel()
    rb = np.asarray(r_b, dtype=float).ravel()
    n = wp.shape[0]
    if not (wb.shape[0] == rp.shape[0] == rb.shape[0] == n):
        raise ValueError("w_p, w_b, r_p, r_b must all have the same length")
    if n == 0:
        raise ValueError("need at least one sector")

    R_p = float(wp @ rp)
    R_b = float(wb @ rb)

    allocation = (wp - wb) * (rb - R_b)
    selection = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)
    total = allocation + selection + interaction

    active = R_p - R_b
    # The decomposition is an algebraic identity; verify it numerically.
    assert abs(total.sum() - active) < 1e-9, (
        f"Brinson components {total.sum():.3e} != active {active:.3e}; "
        "check that w_p and w_b each sum to 1 (a different cash budget breaks "
        "the cross-term cancellation)."
    )

    if sectors is None:
        idx = pd.RangeIndex(n)
    else:
        idx = pd.Index(list(sectors), name="sector")
    by_sector = pd.DataFrame(
        {"allocation": allocation, "selection": selection,
         "interaction": interaction, "total": total},
        index=idx,
    )
    return {
        "by_sector": by_sector,
        "allocation": float(allocation.sum()),
        "selection": float(selection.sum()),
        "interaction": float(interaction.sum()),
        "active_return": active,
        "R_p": R_p,
        "R_b": R_b,
    }


# --------------------------------------------------------------------------- #
# Carino multi-period linking
# --------------------------------------------------------------------------- #
def _carino_coef(r: float, b: float) -> float:
    """Per-period Carino linking coefficient k_t.

        k_t = ( ln(1+r) - ln(1+b) ) / ( r - b )           if r != b
        k_t = 1 / (1 + r)                                  (the 0/0 limit r -> b)

    Derivation of the limit: as the arithmetic active a = r - b -> 0, the ratio
    [ln(1+r) - ln(1+b)] / (r - b) is the difference quotient of f(x)=ln(1+x),
    whose derivative is f'(x) = 1/(1+x). So the 0/0 form converges to
    1/(1+b) = 1/(1+r) (they coincide at r = b). Using the raw ratio there would
    divide 0 by 0 and return NaN; this guard makes k_t continuous everywhere.
    """
    if 1.0 + r <= 0.0 or 1.0 + b <= 0.0:
        raise ValueError("period return <= -100%; log-linking is undefined")
    if abs(r - b) < _EPS:
        return 1.0 / (1.0 + r)
    return (math.log1p(r) - math.log1p(b)) / (r - b)


def carino_link(period_active, R_p, R_b):
    """Carino (1999) logarithmic linking of single-period active returns.

    Problem: arithmetic single-period active returns a_t = r_t - b_t do NOT add
    up to the geometric multi-period active return
        A = R_p - R_b = prod(1+r_t) - 1 - (prod(1+b_t) - 1),
    because the portfolio and benchmark compound on different bases. Naively
    summing a_t leaves a (often large) cross-product residual.

    Carino's fix scales each a_t by k_t / k, where
        k_t = [ln(1+r_t) - ln(1+b_t)] / (r_t - b_t)    (per-period coefficient)
        k   = [ln(1+R_p)  - ln(1+R_b)] / (R_p - R_b)   (total-period coefficient)
    Then  sum_t (k_t / k) * a_t  ==  R_p - R_b  EXACTLY. Intuition: k_t maps each
    arithmetic active into log space (where active returns DO add), and dividing
    by the total-period k maps the summed log-active back to the arithmetic
    total. Both k_t and k carry a guarded 0/0 limit -> 1/(1+R) (see _carino_coef).

    Parameters
    ----------
    period_active : array-like of length T. Each entry is the *paired* per-period
        (r_t, b_t) needed to form k_t, so this argument must be a (T, 2) array /
        DataFrame whose columns are [r_t, b_t]. (k_t depends on the levels r_t,
        b_t, not just their difference, so the raw active a_t alone is
        insufficient - this is the single most common Carino implementation bug.)
    R_p, R_b : floats, the COMPOUNDED total portfolio / benchmark returns over the
        whole window, R_p = prod(1+r_t) - 1, R_b = prod(1+b_t) - 1. Passed in
        explicitly so the caller controls the compounding (and so this links
        Brinson EFFECTS, whose r_t/b_t are sub-portfolio returns, the same way).

    Returns
    -------
    dict with:
      'linked'        : ndarray (T,), the linked per-period contributions
                        (k_t / k) * a_t that sum to R_p - R_b.
      'k_total'       : float, the total-period coefficient k.
      'k_period'      : ndarray (T,), the per-period coefficients k_t.
      'active_total'  : float, R_p - R_b (the geometric total the linked sum hits).
      'raw_active'    : ndarray (T,), the unlinked a_t = r_t - b_t.
    """
    arr = np.asarray(pd.DataFrame(period_active, dtype="float64"), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "period_active must be (T, 2) with columns [r_t, b_t]; k_t needs the "
            "return LEVELS, not just the active difference"
        )
    r = arr[:, 0]
    b = arr[:, 1]
    raw_active = r - b

    k_period = np.array([_carino_coef(float(ri), float(bi)) for ri, bi in zip(r, b)])
    A = float(R_p) - float(R_b)
    k_total = _carino_coef(float(R_p), float(R_b))  # guarded 0/0 -> 1/(1+R_p)

    if abs(k_total) < _EPS:
        raise ValueError("total linking coefficient is ~0; check R_p / R_b")
    linked = (k_period / k_total) * raw_active

    # Identity check: the linked contributions reproduce the geometric active.
    assert abs(linked.sum() - A) < 1e-9, (linked.sum(), A)

    return {
        "linked": linked,
        "k_total": k_total,
        "k_period": k_period,
        "active_total": A,
        "raw_active": raw_active,
    }


# --------------------------------------------------------------------------- #
# Factor attribution (time-series OLS decomposition)
# --------------------------------------------------------------------------- #
def factor_attribution(returns, factor_returns) -> Dict[str, object]:
    """Decompose a return series into factor contributions via OLS.

    Fits  r_t = alpha + sum_k beta_k * f_{k,t} + eps_t  by ordinary least squares
    (intercept included) over the supplied window, then reports the AVERAGE
    contribution of each piece:

        contribution_k = beta_k * mean(f_k)      (per-factor)
        alpha_contrib  = alpha                   (the OLS intercept, per period)
        residual_mean  = mean(eps_t) (~0 by OLS construction)

    Identity:  mean(r) = alpha + sum_k beta_k * mean(f_k)  (exact for OLS with an
    intercept, since the residuals have zero mean). The returned contributions
    therefore sum to the realized mean return - asserted in the self-tests.

    IN-SAMPLE caveat (see module docstring): betas are estimated on the same
    window they explain, so this is a descriptive decomposition, not a tradeable
    forecast. Solved with numpy.linalg.lstsq (no statsmodels).

    Parameters
    ----------
    returns        : array-like (T,) portfolio (or active) returns.
    factor_returns : array-like (T, K) or DataFrame of K factor return series.

    Returns
    -------
    dict with:
      'alpha'             : float, OLS intercept (per-period alpha).
      'betas'             : ndarray (K,) factor loadings.
      'factor_names'      : list[str].
      'contributions'     : dict name -> beta_k * mean(f_k).
      'alpha_contrib'     : float (== alpha).
      'residual_mean'     : float (~0).
      'mean_return'       : float, realized mean(r).
      'r_squared'         : float, in-sample R^2.
    """
    y = np.asarray(returns, dtype=float).ravel()
    F_df = pd.DataFrame(factor_returns, dtype="float64")
    if F_df.shape[0] != y.shape[0]:
        raise ValueError("returns and factor_returns must have the same length")
    names = [str(c) for c in F_df.columns]
    F = F_df.to_numpy(dtype=float)
    T, K = F.shape
    if T <= K + 1:
        raise ValueError(f"need T > K+1 observations for OLS (T={T}, K={K})")

    X = np.column_stack([np.ones(T), F])          # [intercept | factors]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coef[0])
    betas = coef[1:].astype(float)

    fitted = X @ coef
    resid = y - fitted
    f_means = F.mean(axis=0)
    contributions = {nm: float(b * m) for nm, b, m in zip(names, betas, f_means)}

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > _EPS else float("nan")

    return {
        "alpha": alpha,
        "betas": betas,
        "factor_names": names,
        "contributions": contributions,
        "alpha_contrib": alpha,
        "residual_mean": float(resid.mean()),
        "mean_return": float(y.mean()),
        "r_squared": r_squared,
    }


# --------------------------------------------------------------------------- #
# Perold implementation-shortfall waterfall
# --------------------------------------------------------------------------- #
def implementation_shortfall(
    decision_price: float,
    arrival_price: float,
    avg_exec_price: float,
    close_price: float,
    target_shares: float,
    filled_shares: float,
    side: str = "buy",
    commission_per_share: float = 0.0,
) -> Dict[str, float]:
    """Perold (1988) implementation-shortfall waterfall (all in CURRENCY terms).

    Implementation shortfall is the paper-vs-reality gap: the return of a costless
    "paper" portfolio that transacts the full target at the DECISION price, minus
    the realized return of the actual (partially) filled book. We report it
    decomposed into the standard cost buckets (signed so each is a COST when
    positive, i.e. it ate into return), for a BUY by default:

        delay (slippage)   = filled * (arrival_price  - decision_price)
            price drift between the decision and when the order reached the market.
        trading (impact)   = filled * (avg_exec_price - arrival_price)
            market impact + timing while working the order, on the FILLED shares.
        opportunity cost   = unfilled * (close_price   - decision_price)
            adverse move on the shares you MEANT to trade but never got (Perold's
            key insight: the missed trade is a real cost, here marked to the
            period close), where unfilled = target_shares - filled_shares.
        commission         = filled * commission_per_share   (explicit cost)

    total_shortfall = delay + trading + opportunity + commission.

    Sign convention: for a SELL the price differences are negated (a price that
    rises after a sell decision HELPS you, so it is a negative cost). For a buy,
    paying MORE than the decision price is a positive (adverse) cost.

    All quantities are causal/ex-post realized prices over a closed execution
    window - a pure post-trade diagnostic, never an input to sizing the same
    order (that would use the order's own future fills).

    Parameters
    ----------
    decision_price : the price when the PM decided to trade (the "paper" price).
    arrival_price  : the price when the order arrived at the market/broker.
    avg_exec_price : volume-weighted average realized fill price.
    close_price    : the price used to mark the UNFILLED remainder (period close).
    target_shares  : intended order size (shares; positive magnitude).
    filled_shares  : shares actually executed (0 <= filled <= target).
    side           : 'buy' or 'sell'.
    commission_per_share : explicit per-share commission/fees.

    Returns
    -------
    dict with 'delay', 'trading', 'opportunity', 'commission',
    'total_shortfall', and 'total_bps' (shortfall as bps of the paper notional
    target_shares * decision_price).
    """
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if target_shares <= 0:
        raise ValueError("target_shares must be positive (use side to set direction)")
    if not (0.0 <= filled_shares <= target_shares + _EPS):
        raise ValueError("filled_shares must satisfy 0 <= filled <= target")

    sign = 1.0 if side == "buy" else -1.0
    unfilled = target_shares - filled_shares

    delay = sign * filled_shares * (arrival_price - decision_price)
    trading = sign * filled_shares * (avg_exec_price - arrival_price)
    opportunity = sign * unfilled * (close_price - decision_price)
    commission = filled_shares * commission_per_share

    total = delay + trading + opportunity + commission

    paper_notional = target_shares * decision_price
    total_bps = (total / paper_notional * 1e4) if abs(paper_notional) > _EPS else float("nan")

    return {
        "delay": float(delay),
        "trading": float(trading),
        "opportunity": float(opportunity),
        "commission": float(commission),
        "total_shortfall": float(total),
        "total_bps": float(total_bps),
    }


# --------------------------------------------------------------------------- #
# Self-tests (analytic anchors) - run: python attribution.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ----------------------------------------------------------------------- #
    # Brinson-Fachler: components sum EXACTLY to total active (random & hand) #
    # ----------------------------------------------------------------------- #
    # (a) random many-sector book: both weight vectors sum to 1.
    for _ in range(200):
        k = int(rng.integers(2, 8))
        wp = rng.dirichlet(np.ones(k))
        wb = rng.dirichlet(np.ones(k))
        rp = rng.normal(0.0, 0.05, k)
        rb = rng.normal(0.0, 0.05, k)
        out = brinson_fachler(wp, wb, rp, rb)
        comp = out["allocation"] + out["selection"] + out["interaction"]
        assert abs(comp - out["active_return"]) < 1e-12, (comp, out["active_return"])
        # active_return equals R_p - R_b by construction
        assert np.isclose(out["active_return"], float(wp @ rp) - float(wb @ rb))

    # (b) hand-checked two-sector case.
    #   sectors A,B; w_p=[0.6,0.4], w_b=[0.5,0.5]; r_p=[0.10,0.04], r_b=[0.08,0.02]
    wp = np.array([0.6, 0.4]); wb = np.array([0.5, 0.5])
    rp = np.array([0.10, 0.04]); rb = np.array([0.08, 0.02])
    out = brinson_fachler(wp, wb, rp, rb, sectors=["A", "B"])
    R_b = 0.5 * 0.08 + 0.5 * 0.02            # = 0.05
    R_p = 0.6 * 0.10 + 0.4 * 0.04            # = 0.076
    # allocation = (w_p-w_b)(r_b - R_b): A:(0.1)(0.03)=0.003; B:(-0.1)(-0.03)=0.003 -> 0.006
    assert np.isclose(out["allocation"], 0.006)
    # selection = w_b(r_p - r_b): A:0.5*0.02=0.01; B:0.5*0.02=0.01 -> 0.02
    assert np.isclose(out["selection"], 0.02)
    # interaction = (w_p-w_b)(r_p-r_b): A:0.1*0.02=0.002; B:-0.1*0.02=-0.002 -> 0.0
    assert np.isclose(out["interaction"], 0.0)
    assert np.isclose(out["active_return"], R_p - R_b)
    assert np.isclose(out["allocation"] + out["selection"] + out["interaction"],
                      R_p - R_b)
    # per-sector frame total = alloc+sel+inter, which equals the per-sector
    # active MINUS the Fachler cross term (w_p-w_b)*R_b (that term only cancels
    # in the SUM); the column sum still recovers the total active return.
    bysec = out["by_sector"]
    assert np.allclose(bysec["total"].to_numpy(),
                       wp * rp - wb * rb - (wp - wb) * R_b)
    assert np.isclose(bysec["total"].sum(), R_p - R_b)

    # ----------------------------------------------------------------------- #
    # Carino linking: linked sum == geometric (compounded) active return     #
    # ----------------------------------------------------------------------- #
    # Build a multi-period path of portfolio & benchmark per-period returns.
    r = np.array([0.05, -0.02, 0.03, 0.04, -0.01])
    b = np.array([0.03, -0.01, 0.02, 0.05, -0.02])
    R_p = float(np.prod(1 + r) - 1)
    R_b = float(np.prod(1 + b) - 1)
    linked = carino_link(np.column_stack([r, b]), R_p, R_b)
    # the headline property: linked contributions sum to the GEOMETRIC active
    assert np.isclose(linked["linked"].sum(), R_p - R_b, atol=1e-12)
    assert np.isclose(linked["active_total"], R_p - R_b)
    # the naive (unlinked) arithmetic sum does NOT equal the geometric active
    # here (cross-compounding residual), proving linking is doing real work
    assert abs(linked["raw_active"].sum() - (R_p - R_b)) > 1e-5

    # k_t analytic anchor: for a single period with r != b,
    #   k_1 = (ln(1+r) - ln(1+b)) / (r - b)
    assert np.isclose(_carino_coef(0.05, 0.03),
                      (math.log(1.05) - math.log(1.03)) / 0.02)

    # ----------------------------------------------------------------------- #
    # Carino 0/0 guard: r == b  =>  k_t -> 1/(1+r) (no NaN)                   #
    # ----------------------------------------------------------------------- #
    k_eq = _carino_coef(0.07, 0.07)
    assert np.isclose(k_eq, 1.0 / 1.07), k_eq
    assert not math.isnan(k_eq)
    # continuity: as b -> r the raw ratio approaches the guarded limit
    approach = [_carino_coef(0.07, 0.07 + d) for d in (1e-3, 1e-5, 1e-7)]
    assert all(np.isclose(x, 1.0 / 1.07, atol=1e-2) for x in approach)
    assert np.isclose(approach[-1], 1.0 / 1.07, atol=1e-6)
    # total-period 0/0 guard: when R_p == R_b the active is 0 and every linked
    # contribution is 0 (k_total = 1/(1+R) finite, raw_active all 0 since r==b)
    z = carino_link(np.column_stack([r, r]), 0.123, 0.123)
    assert np.isclose(z["k_total"], 1.0 / 1.123)
    assert np.allclose(z["linked"], 0.0)
    assert np.isclose(z["linked"].sum(), 0.0)

    # ----------------------------------------------------------------------- #
    # Factor attribution: contributions + alpha + residual == mean return    #
    # ----------------------------------------------------------------------- #
    T = 600
    f1 = rng.normal(0.0004, 0.01, T)
    f2 = rng.normal(0.0002, 0.008, T)
    true_alpha, beta1, beta2 = 0.0001, 1.3, -0.5
    eps = rng.normal(0.0, 0.002, T)
    port = true_alpha + beta1 * f1 + beta2 * f2 + eps
    fa = factor_attribution(port, pd.DataFrame({"MKT": f1, "VAL": f2}))
    # recovered betas close to truth
    assert abs(fa["betas"][0] - beta1) < 0.05, fa["betas"]
    assert abs(fa["betas"][1] - beta2) < 0.05, fa["betas"]
    # exact OLS identity: alpha + sum(beta_k * mean f_k) == mean return
    recon = fa["alpha_contrib"] + sum(fa["contributions"].values())
    assert np.isclose(recon, fa["mean_return"], atol=1e-12), (recon, fa["mean_return"])
    assert abs(fa["residual_mean"]) < 1e-12  # OLS w/ intercept -> zero-mean resid
    assert 0.0 <= fa["r_squared"] <= 1.0

    # ----------------------------------------------------------------------- #
    # Perold implementation shortfall: buckets sum to total; hand-checked     #
    # ----------------------------------------------------------------------- #
    # Buy 1000 shares. decision=100, arrival=100.2, avg_fill=100.5, close=101.
    # Filled 800 (200 unfilled). commission 0.01/sh.
    isf = implementation_shortfall(
        decision_price=100.0, arrival_price=100.2, avg_exec_price=100.5,
        close_price=101.0, target_shares=1000, filled_shares=800,
        side="buy", commission_per_share=0.01,
    )
    # delay       = 800 * (100.2 - 100.0)   = 160
    # trading     = 800 * (100.5 - 100.2)   = 240
    # opportunity = 200 * (101.0 - 100.0)   = 200
    # commission  = 800 * 0.01              = 8
    assert np.isclose(isf["delay"], 160.0)
    assert np.isclose(isf["trading"], 240.0)
    assert np.isclose(isf["opportunity"], 200.0)
    assert np.isclose(isf["commission"], 8.0)
    assert np.isclose(isf["total_shortfall"], 160 + 240 + 200 + 8)
    # total_bps relative to paper notional 1000*100 = 100_000
    assert np.isclose(isf["total_bps"], 608.0 / 100_000 * 1e4)

    # full fill => zero opportunity cost; buckets still reconcile
    isf_full = implementation_shortfall(
        decision_price=50.0, arrival_price=50.0, avg_exec_price=50.1,
        close_price=49.0, target_shares=500, filled_shares=500, side="buy",
    )
    assert np.isclose(isf_full["opportunity"], 0.0)
    assert np.isclose(isf_full["total_shortfall"],
                      isf_full["delay"] + isf_full["trading"] + isf_full["commission"])

    # SELL sign flip: a price RISE after a sell decision is a NEGATIVE cost
    # (favorable). Sell 1000, decision=100, avg_fill=100.5 (sold higher = good).
    isf_sell = implementation_shortfall(
        decision_price=100.0, arrival_price=100.0, avg_exec_price=100.5,
        close_price=100.0, target_shares=1000, filled_shares=1000, side="sell",
    )
    # trading = -1 * 1000 * (100.5 - 100.0) = -500  (favorable -> negative cost)
    assert np.isclose(isf_sell["trading"], -500.0)
    assert np.isclose(isf_sell["total_shortfall"], -500.0)

    print("attribution.py: all self-tests passed")
