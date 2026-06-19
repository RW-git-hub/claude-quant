"""execution.py - Execution-scheduling algorithms for optimal order execution.

Market microstructure & optimal execution: how a parent order of known size is
sliced into child orders over time. This module covers EXECUTION TRAJECTORIES
(the schedule of how much to trade in each bar) and the cost/risk tradeoff that
governs them. It is DISTINCT from transaction-costs.md / costs.py, which model
the *magnitude* of cost (spread, impact coefficients); here we assume those
coefficients and solve for the *trajectory*.

Conventions
-----------
- Quantities are signed-agnostic: pass positive total_qty; the side only matters
  for cost attribution (implementation_shortfall). A schedule for a sell is the
  same shape as for a buy.
- "Schedule" = per-bar child-order sizes (a trade list), nonnegative, summing to
  total_qty (up to the cap for POV).
- "Trajectory" = REMAINING holdings x_0..x_n with x_0 = X (full position) and
  x_n = 0 (fully executed). The trade in bar k is x_{k-1} - x_k.
- tau = length of one bar (in the same time units used to annualize sigma); n
  bars, so total horizon T = n*tau.
- All impact/vol coefficients are in price-per-share / share units consistent
  with the trajectory units; we do not impose a particular currency.

The classic reference is Almgren & Chriss (2000), "Optimal execution of
portfolio transactions", J. Risk. The two competing forces:

    1. Market impact: trading fast moves the price against you (cost grows with
       trade rate). Favors slow, spread-out execution (TWAP-like).
    2. Timing/volatility risk: holding an unexecuted position exposes you to
       price moves before you finish. Favors fast execution (front-loading).

The risk-aversion lam (lambda) trades these off. lam -> 0 ignores risk and gives
a near-linear (TWAP) trajectory; large lam front-loads aggressively.

Pitfalls (detect / fix)
-----------------------
- DETECT: a "VWAP" schedule built from REALIZED volume of the execution day ->
  look-ahead. FIX: use a volume profile forecast known before the bar (e.g. a
  trailing historical intraday curve), as vwap_schedule() assumes.
- DETECT: POV schedule that lets cumulative qty overshoot total_qty in the last
  bar. FIX: cap and top-up (pov_schedule does this).
- DETECT: implementation shortfall computed against the *first fill price*
  instead of the decision/arrival price -> hides delay cost. FIX: always vs the
  decision price captured when the order was created.
- DETECT: comparing IS without sign convention -> a buy filled cheap should be
  NEGATIVE cost (a gain). FIX: side-aware sign (this module).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Simple benchmark schedules
# ---------------------------------------------------------------------------
def twap_schedule(total_qty: float, n_slices: int) -> np.ndarray:
    """Time-Weighted Average Price schedule: equal child orders.

    Splits total_qty into n_slices equal pieces. The benchmark a TWAP order is
    measured against is the simple time average of prices over the window.

    Parameters
    ----------
    total_qty : total parent quantity (use positive magnitude).
    n_slices  : number of equally spaced child orders (bars).

    Returns
    -------
    np.ndarray of length n_slices, each = total_qty / n_slices, summing exactly
    to total_qty (the last element absorbs floating-point remainder).
    """
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    sched = np.full(n_slices, total_qty / n_slices, dtype=float)
    # Absorb any float drift into the last slice so the sum is exact.
    sched[-1] = total_qty - sched[:-1].sum()
    return sched


def vwap_schedule(total_qty: float, volume_profile: Sequence[float]) -> np.ndarray:
    """Volume-Weighted Average Price schedule: child orders ~ expected volume.

    Allocates total_qty across bars proportionally to a FORECAST volume profile
    (e.g. a typical intraday U-shape). Using realized same-day volume would be
    look-ahead; the profile must be known before trading.

    Parameters
    ----------
    total_qty      : total parent quantity (positive magnitude).
    volume_profile : per-bar expected volume (weights). Need not be normalized;
                     must be nonnegative with positive sum.

    Returns
    -------
    np.ndarray of child sizes, proportional to volume_profile, summing exactly
    to total_qty.
    """
    w = np.asarray(volume_profile, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("volume_profile must be a non-empty 1-D sequence")
    if np.any(w < 0):
        raise ValueError("volume_profile must be nonnegative")
    s = w.sum()
    if s <= 0:
        raise ValueError("volume_profile must have a positive sum")
    sched = total_qty * (w / s)
    sched[-1] = total_qty - sched[:-1].sum()  # exact sum
    return sched


def pov_schedule(
    total_qty: float,
    volume_forecast: Sequence[float],
    participation: float,
) -> np.ndarray:
    """Percent-Of-Volume (participation) schedule.

    Trade `participation * volume_t` each bar until total_qty is exhausted. The
    bar that would push cumulative qty over total_qty is capped to the residual,
    and all subsequent bars trade zero. This makes the schedule adapt to volume
    rather than to the clock: in high-volume bars you trade more.

    Parameters
    ----------
    total_qty       : total parent quantity (positive magnitude).
    volume_forecast : per-bar forecast market volume (nonnegative).
    participation   : target fraction of each bar's volume, in (0, 1].

    Returns
    -------
    np.ndarray, same length as volume_forecast, of per-bar child sizes whose sum
    equals min(total_qty, participation * sum(volume_forecast)). If volume is
    sufficient, the sum equals total_qty exactly (last active bar tops up).
    """
    if not (0.0 < participation <= 1.0):
        raise ValueError("participation must be in (0, 1]")
    v = np.asarray(volume_forecast, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("volume_forecast must be a non-empty 1-D sequence")
    if np.any(v < 0):
        raise ValueError("volume_forecast must be nonnegative")

    sched = np.zeros_like(v)
    remaining = float(total_qty)
    for t in range(v.size):
        if remaining <= 0:
            break
        want = participation * v[t]
        take = min(want, remaining)  # cap so cumulative never exceeds total_qty
        sched[t] = take
        remaining -= take
    return sched


# ---------------------------------------------------------------------------
# Almgren-Chriss optimal trajectory
# ---------------------------------------------------------------------------
def almgren_chriss_trajectory(
    X: float,
    n: int,
    tau: float,
    eta: float,
    gamma: float,
    sigma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Almgren-Chriss optimal liquidation trajectory.

    Solves the mean-variance optimal execution problem for liquidating (or
    acquiring) X shares over n bars of length tau. The closed-form optimal
    REMAINING-holdings trajectory is

        x_k = X * sinh(kappa * (n - k) * tau) / sinh(kappa * n * tau),
        k = 0, 1, ..., n,  with x_0 = X and x_n = 0,

    where kappa (the "urgency" curvature) solves

        cosh(kappa * tau) = 1 + (lam * sigma^2 * tau^2) / (2 * eta_tilde),
        eta_tilde = eta - 0.5 * gamma * tau.

    Model
    -----
    - Permanent impact:  price drifts by gamma * (shares traded) (the gamma term
      cancels out of the optimal *shape* but enters total cost).
    - Temporary impact:  trading n_k shares in bar k costs eta * (n_k / tau) per
      share, i.e. linear in trade RATE. eta_tilde absorbs the half-bar
      permanent-impact correction.
    - Risk:              variance of the unexecuted position, weighted by lam.

    The continuous-time analogue gives kappa ~= sqrt(lam * sigma^2 / eta_tilde);
    we use the exact discrete cosh relation, which reduces to that for small tau.

    Behavior of lam
    ---------------
    - lam -> 0  : kappa -> 0, sinh terms -> linear, so x_k -> X*(n-k)/n. This is
      exactly the TWAP (linear) trajectory: trade equal amounts each bar,
      ignoring risk.
    - lam large : kappa large, trajectory becomes convex and FRONT-LOADED -
      sell most of the position early to cut exposure, accepting higher impact.

    Parameters
    ----------
    X     : total quantity to execute (signed-agnostic; pass magnitude).
    n     : number of bars (n >= 1). Returns n+1 holdings points.
    tau   : bar length (time units consistent with sigma's annualization).
    eta   : temporary impact coefficient (price per unit trade rate).
    gamma : permanent impact coefficient (price per unit traded).
    sigma : per-bar (or per-time-unit) volatility consistent with tau.
    lam   : risk aversion (>= 0). 0 => TWAP-linear.

    Returns
    -------
    (holdings, trades)
      holdings : np.ndarray length n+1, remaining position x_0..x_n.
      trades   : np.ndarray length n, child sizes trades[k] = x_k - x_{k+1}
                 (positive for a liquidation), summing to X.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if lam < 0:
        raise ValueError("lam must be nonnegative")

    eta_tilde = eta - 0.5 * gamma * tau
    if eta_tilde <= 0:
        raise ValueError(
            "eta_tilde = eta - 0.5*gamma*tau must be positive "
            "(permanent impact too large relative to temporary impact)"
        )

    k = np.arange(n + 1, dtype=float)

    if lam == 0.0:
        # Degenerate: kappa = 0 => linear (TWAP) trajectory.
        holdings = X * (n - k) / n
    else:
        # Exact discrete kappa from the cosh relation.
        rhs = 1.0 + (lam * sigma**2 * tau**2) / (2.0 * eta_tilde)
        # rhs >= 1 always (lam,sigma^2,tau^2,eta_tilde > 0), so arccosh is valid.
        kappa = float(np.arccosh(rhs)) / tau
        denom = np.sinh(kappa * n * tau)
        holdings = X * np.sinh(kappa * (n - k) * tau) / denom

    # Pin endpoints exactly (guard against float drift).
    holdings[0] = X
    holdings[-1] = 0.0
    trades = holdings[:-1] - holdings[1:]
    return holdings, trades


# ---------------------------------------------------------------------------
# Cost / performance measurement
# ---------------------------------------------------------------------------
def implementation_shortfall(
    decision_price: float,
    fills: Sequence[Tuple[float, float]],
    side: str = "buy",
) -> float:
    """Implementation shortfall in basis points vs the decision/arrival price.

    IS measures the cost of execution relative to the price at the moment the
    trading decision was made (the "arrival" or "decision" price), capturing
    both market impact and any delay/drift. It must be measured against the
    decision price, NOT the first fill, or delay cost is hidden.

    Definition (per share, side-aware)
    ----------------------------------
        avg_fill = sum(p*q) / sum(q)
        buy : cost = (avg_fill - decision_price)   # paying more = positive cost
        sell: cost = (decision_price - avg_fill)   # selling lower = positive cost
        IS_bps = 1e4 * cost / decision_price

    A positive value is a LOSS (worse than arrival); negative is a gain.

    Parameters
    ----------
    decision_price : arrival/decision benchmark price (> 0).
    fills          : list of (price, qty) child fills; qty > 0.
    side           : 'buy' or 'sell'.

    Returns
    -------
    IS in basis points (float).
    """
    if decision_price <= 0:
        raise ValueError("decision_price must be positive")
    if not fills:
        raise ValueError("fills must be non-empty")
    s = side.lower()
    if s not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    prices = np.asarray([p for p, _ in fills], dtype=float)
    qtys = np.asarray([q for _, q in fills], dtype=float)
    if np.any(qtys <= 0):
        raise ValueError("all fill quantities must be positive")
    total_q = qtys.sum()
    avg_fill = float((prices * qtys).sum() / total_q)

    if s == "buy":
        cost = avg_fill - decision_price
    else:
        cost = decision_price - avg_fill
    return 1e4 * cost / decision_price


def expected_execution_cost(
    trajectory: Sequence[float],
    eta: float,
    gamma: float,
    sigma: float,
    tau: float,
) -> Tuple[float, float, float]:
    """Expected execution cost = temporary impact + permanent impact + risk term.

    Given a holdings trajectory x_0..x_n (x_0 = X total, x_n = 0), with trades
    n_k = x_{k-1} - x_k in bar k, the Almgren-Chriss cost decomposition is

      Expected (mean) cost:
        permanent = gamma * sum_k n_k * (x_k + x_{k-1})/2     (drift on remaining)
                  = 0.5 * gamma * X^2   (telescopes; depends only on X)
        temporary = sum_k (eta / tau) * n_k^2                 (rate^2 * tau)
      Variance of cost (timing risk):
        var = sigma^2 * tau * sum_k x_k^2     (k=1..n, unexecuted holdings)

    We return (expected_cost, variance, std) where
        expected_cost = permanent + temporary
    so the mean-variance objective minimized by almgren_chriss_trajectory is
    expected_cost + lam * variance.

    Note: the permanent-impact term telescopes to 0.5*gamma*X^2 regardless of
    schedule, which is why it does not affect the optimal SHAPE - only eta, sigma
    and lam (via kappa) do.

    Parameters
    ----------
    trajectory : remaining-holdings path x_0..x_n (length n+1).
    eta, gamma, sigma, tau : model coefficients (see almgren_chriss_trajectory).

    Returns
    -------
    (expected_cost, variance, std_dev).
    """
    x = np.asarray(trajectory, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("trajectory must have length >= 2")
    if tau <= 0:
        raise ValueError("tau must be positive")

    trades = x[:-1] - x[1:]                    # n_k, k = 1..n
    midpoints = 0.5 * (x[:-1] + x[1:])

    permanent = gamma * float(np.sum(trades * midpoints))
    temporary = (eta / tau) * float(np.sum(trades**2))
    expected_cost = permanent + temporary

    # Variance from holding the unexecuted position x_1..x_n over each bar.
    variance = sigma**2 * tau * float(np.sum(x[1:] ** 2))
    std_dev = float(np.sqrt(variance))
    return expected_cost, variance, std_dev


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- twap_schedule -----------------------------------------------------
    t = twap_schedule(100.0, 4)
    assert np.allclose(t, [25.0, 25.0, 25.0, 25.0]), t
    assert abs(t.sum() - 100.0) < 1e-12
    # Non-divisible total still sums exactly.
    t2 = twap_schedule(100.0, 3)
    assert abs(t2.sum() - 100.0) < 1e-12
    assert np.allclose(t2[:2], 100.0 / 3.0)

    # --- vwap_schedule -----------------------------------------------------
    v = vwap_schedule(100.0, [1, 2, 1])
    assert abs(v.sum() - 100.0) < 1e-12
    assert np.allclose(v, [25.0, 50.0, 25.0]), v
    # Proportionality: child / weight is constant.
    w = np.array([1.0, 2.0, 1.0])
    ratios = v / w
    assert np.allclose(ratios, ratios[0]), ratios
    # Unnormalized weights give same result as normalized.
    v_norm = vwap_schedule(100.0, [0.25, 0.5, 0.25])
    assert np.allclose(v, v_norm)

    # --- pov_schedule ------------------------------------------------------
    # Volume large enough: full qty executed, cap respected per bar.
    vol = [1000.0, 1000.0, 1000.0, 1000.0]
    p = pov_schedule(100.0, vol, participation=0.10)
    # 0.10*1000 = 100 each bar; total_qty=100 -> first bar takes all 100.
    assert abs(p.sum() - 100.0) < 1e-12
    assert p[0] == 100.0 and p[1] == 0.0
    # No bar exceeds participation * volume.
    for pt, vt in zip(p, vol):
        assert pt <= 0.10 * vt + 1e-12
    # Spread across bars when participation cap binds before completion.
    p2 = pov_schedule(100.0, [200.0, 200.0, 200.0, 200.0], participation=0.10)
    # 20 per bar -> 20,20,20,20,... need 100 over 5 bars but only 4 given,
    # so sum = 80 (volume-limited), last bar capped at 20.
    assert abs(p2.sum() - 80.0) < 1e-9, p2
    assert np.allclose(p2, [20.0, 20.0, 20.0, 20.0])
    # Top-up case: total_qty falls between cap multiples.
    p3 = pov_schedule(50.0, [200.0, 200.0, 200.0], participation=0.10)
    # 20,20 -> 40, then residual 10 tops up bar 3 (cap 20 not binding).
    assert np.allclose(p3, [20.0, 20.0, 10.0]), p3
    assert abs(p3.sum() - 50.0) < 1e-12

    # --- almgren_chriss_trajectory: monotone, endpoints --------------------
    X = 1.0
    n = 10
    tau = 1.0
    eta = 0.1
    gamma = 0.01
    sigma = 0.3

    hi, tr_hi = almgren_chriss_trajectory(X, n, tau, eta, gamma, sigma, lam=2.0)
    lo, tr_lo = almgren_chriss_trajectory(X, n, tau, eta, gamma, sigma, lam=1e-6)

    # Endpoints.
    assert abs(hi[0] - X) < 1e-12 and abs(hi[-1]) < 1e-12
    assert abs(lo[0] - X) < 1e-12 and abs(lo[-1]) < 1e-12
    # Monotonically decreasing (non-increasing within float tol).
    assert np.all(np.diff(hi) <= 1e-12), hi
    assert np.all(np.diff(lo) <= 1e-12), lo
    # Trades sum to X and are nonnegative.
    assert abs(tr_hi.sum() - X) < 1e-12
    assert np.all(tr_hi >= -1e-12)

    # High risk-aversion front-loads: after the first step, LOWER holdings.
    assert hi[1] < lo[1], (hi[1], lo[1])
    # In fact high-lam holdings are <= low-lam everywhere (front-loaded).
    assert np.all(hi[1:-1] <= lo[1:-1] + 1e-12)

    # lam -> 0 approaches the linear TWAP trajectory.
    linear = np.linspace(X, 0.0, n + 1)
    assert np.allclose(lo, linear, atol=1e-3), (lo, linear)
    # Exact lam == 0 is exactly linear.
    z, _ = almgren_chriss_trajectory(X, n, tau, eta, gamma, sigma, lam=0.0)
    assert np.allclose(z, linear, atol=1e-12)

    # Sanity: kappa relation reproduces holdings for a mid lam.
    lam = 0.5
    eta_tilde = eta - 0.5 * gamma * tau
    rhs = 1.0 + (lam * sigma**2 * tau**2) / (2.0 * eta_tilde)
    kappa = np.arccosh(rhs) / tau
    k_idx = np.arange(n + 1)
    expected = X * np.sinh(kappa * (n - k_idx) * tau) / np.sinh(kappa * n * tau)
    mid, _ = almgren_chriss_trajectory(X, n, tau, eta, gamma, sigma, lam=lam)
    assert np.allclose(mid, expected, atol=1e-12)

    # --- implementation_shortfall ------------------------------------------
    # Buy filled ABOVE arrival -> positive IS (a cost).
    arrival = 100.0
    fills = [(100.5, 50.0), (100.5, 50.0)]  # avg 100.5
    is_bps = implementation_shortfall(arrival, fills, side="buy")
    # (100.5-100)/100 * 1e4 = 50 bps.
    assert abs(is_bps - 50.0) < 1e-9, is_bps
    # Buy filled BELOW arrival -> negative IS (a gain).
    is_neg = implementation_shortfall(arrival, [(99.5, 100.0)], side="buy")
    assert abs(is_neg - (-50.0)) < 1e-9, is_neg
    # Sell sign flips: selling below arrival is a cost (positive).
    is_sell = implementation_shortfall(arrival, [(99.0, 100.0)], side="sell")
    assert abs(is_sell - 100.0) < 1e-9, is_sell
    # Quantity-weighted average is used, not simple average.
    is_w = implementation_shortfall(
        100.0, [(101.0, 90.0), (110.0, 10.0)], side="buy"
    )
    # avg = (101*90 + 110*10)/100 = 101.9 -> 190 bps.
    assert abs(is_w - 190.0) < 1e-9, is_w

    # --- expected_execution_cost -------------------------------------------
    # Permanent term telescopes to 0.5*gamma*X^2 regardless of schedule.
    traj_a = np.linspace(X, 0.0, n + 1)
    traj_b, _ = almgren_chriss_trajectory(X, n, tau, eta, gamma, sigma, lam=2.0)
    ec_a, var_a, sd_a = expected_execution_cost(traj_a, eta, gamma, sigma, tau)
    ec_b, var_b, sd_b = expected_execution_cost(traj_b, eta, gamma, sigma, tau)

    perm = 0.5 * gamma * X**2
    # Recover permanent-only by checking against analytic telescope.
    trades_a = traj_a[:-1] - traj_a[1:]
    mid_a = 0.5 * (traj_a[:-1] + traj_a[1:])
    assert abs(gamma * np.sum(trades_a * mid_a) - perm) < 1e-12

    # Front-loaded (high-lam) schedule has LOWER variance (less time at risk)
    # but HIGHER temporary impact than the linear schedule.
    assert var_b < var_a, (var_b, var_a)
    assert ec_b > ec_a, (ec_b, ec_a)  # more impact cost for front-loading
    assert abs(sd_a - np.sqrt(var_a)) < 1e-12

    # The AC trajectory minimizes mean + lam*var vs the linear schedule at lam=2.
    obj_ac = ec_b + 2.0 * var_b
    obj_lin = ec_a + 2.0 * var_a
    assert obj_ac <= obj_lin + 1e-9, (obj_ac, obj_lin)

    print("execution.py: all self-tests passed.")
