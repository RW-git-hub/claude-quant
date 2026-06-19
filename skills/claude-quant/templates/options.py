"""
Black-Scholes-Merton (BSM) option pricing, the full Greek set, implied
volatility, a CRR binomial tree for American options, and a discrete
delta-hedging P&L simulator with gamma-theta attribution.

Self-contained: standard library + (optional) numpy only. No scipy.
The normal CDF comes from statistics.NormalDist; the pdf is written out
analytically. Everything is closed-form except implied_vol (a Newton
iteration with a robust bisection fallback) and crr_american (a lattice).

Conventions
-----------
S      : spot price of the underlying (>0)
K      : strike (>0)
T      : time to expiry in YEARS (Act/365 or Act/252 - pick one and be
         consistent with how you annualize sigma; T<=0 is treated as expiry)
r      : continuously-compounded risk-free rate (per year, decimal)
q      : continuous dividend / carry yield (per year, decimal). This is the
         single knob that adapts BSM to every asset class:
           equity index      -> q = dividend yield
           single stock       -> q = continuous proxy for discrete divs
           FX (Garman-Kohlhagen) -> q = r_foreign, r = r_domestic, S = units
                                  of domestic per 1 unit of foreign
           commodity/futures  -> price the future directly with q = r
                                  (Black-76: forward grows at 0 net carry),
                                  or set q = r - convenience_yield on spot
sigma  : annualized volatility of log-returns (decimal, >0)
kind   : 'call' or 'put'

Definitions used throughout
---------------------------
F  = S * exp((r - q) * T)                         (forward price)
d1 = (ln(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
N  = standard-normal CDF      (NormalDist().cdf)
n  = standard-normal pdf      = exp(-x**2 / 2) / sqrt(2*pi)

call = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
put  = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

Put-call parity (with carry):
    call - put = S*exp(-q*T) - K*exp(-r*T)

Greek conventions returned here (state them when reporting!):
    delta  : dV/dS                       (per $1 of spot)
    gamma  : d2V/dS2                      (per $1^2 of spot)
    vega   : dV/dsigma, PER 1.00 of vol  (divide by 100 for "per vol point")
    theta  : dV/dT_calendar PER YEAR, i.e. -dV/d(time-to-expiry).
             Divide by 365 for "per calendar day".
    rho    : dV/dr, PER 1.00 of rate     (divide by 100 for "per bp*100")
    vanna  : dDelta/dsigma = dVega/dS, PER 1.00 of vol per $1 of spot
    volga  : dVega/dsigma  (vomma), PER 1.00 of vol  (the convexity of value
             in vol; long strangles/butterflies are long volga)
    charm  : dDelta/dT_calendar PER YEAR (delta decay), = -dDelta/d(tau).
             Divide by 365 for per-calendar-day delta drift.

Pitfalls (detect/fix)
---------------------
* T<=0 or sigma<=0: the formulas divide by sigma*sqrt(T). We handle the
  degenerate cases explicitly and return the discounted intrinsic / forward
  payoff rather than producing nan or a ZeroDivisionError.
* vega/vanna/volga are reported per 1.00 vol, not per vol point. A "1% move"
  is 0.01. Mixing the two is the most common Greek-scaling bug.
* theta and charm sign: a long ATM option loses value (theta<0) and its delta
  drifts (charm) as time passes. Both are PER YEAR in calendar time here.
* implied_vol can fail to bracket when the quoted price violates no-arb
  bounds (below intrinsic or above the underlying). We return nan instead
  of a bogus root, so callers must check for nan.
* CRR: u/d/p are sized so the tree is arbitrage-free only if 0<=p<=1; with a
  large (r-q) and few steps p can leave [0,1]. We raise rather than return a
  silently-biased price. Use enough steps (a few hundred) for smooth Greeks.
* Delta-hedging (Iron Law 1 & 3): BSM assumes continuous costless hedging.
  The simulator hedges DISCRETELY using only start-of-interval information
  (no look-ahead) and charges a per-share cost on every rebalance. The
  gamma-theta identity 0.5*Gamma*dS^2 + Theta*dt is the dt->0 limit; at finite
  dt the residual IS the discrete-hedging error - report it, don't hide it.
* FX: get the numeraire right. r is domestic, q is foreign. Swapping them
  silently mis-signs the carry and biases every Greek.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional

_N = NormalDist()  # standard normal, mu=0 sigma=1


def _cdf(x: float) -> float:
    """Standard-normal CDF N(x)."""
    return _N.cdf(x)


def _pdf(x: float) -> float:
    """Standard-normal pdf n(x) = exp(-x^2/2)/sqrt(2*pi)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_kind(kind: str) -> str:
    k = kind.strip().lower()
    if k in ("c", "call"):
        return "call"
    if k in ("p", "put"):
        return "put"
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def forward_price(S: float, r: float, q: float, T: float) -> float:
    """Forward price F = S * exp((r - q) * T)."""
    return S * math.exp((r - q) * T)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    """Return (d1, d2). Caller must ensure T>0 and sigma>0."""
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


def _intrinsic_discounted(S: float, K: float, T: float, r: float,
                          q: float, kind: str) -> float:
    """
    Limit price when sigma*sqrt(T) -> 0 (expiry or zero vol): the option is
    worth the discounted payoff evaluated at the forward. This keeps every
    function finite at the degenerate boundary instead of dividing by zero.
    """
    fwd = forward_price(S, r, q, T)
    disc = math.exp(-r * T)
    if kind == "call":
        return disc * max(fwd - K, 0.0)
    return disc * max(K - fwd, 0.0)


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call") -> float:
    """
    Black-Scholes-Merton price with continuous yield q.

        call = S*e^{-qT} N(d1) - K*e^{-rT} N(d2)
        put  = K*e^{-rT} N(-d2) - S*e^{-qT} N(-d1)

    Degenerate T<=0 or sigma<=0 returns the discounted intrinsic value.
    """
    kind = _norm_kind(kind)
    if T <= 0.0 or sigma <= 0.0:
        return _intrinsic_discounted(S, K, T, r, q, kind)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    if kind == "call":
        return S * df_q * _cdf(d1) - K * df_r * _cdf(d2)
    return K * df_r * _cdf(-d2) - S * df_q * _cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call") -> float:
    """
    delta = dV/dS.
        call:  e^{-qT} N(d1)        in (0, 1)
        put:   e^{-qT} (N(d1) - 1)  in (-1, 0)
    """
    kind = _norm_kind(kind)
    if T <= 0.0 or sigma <= 0.0:
        # Step function at expiry: 1/0 for ITM/OTM call (sign flipped for put),
        # scaled by the dividend discount.
        fwd = forward_price(S, r, q, T)
        df_q = math.exp(-q * T)
        if kind == "call":
            return df_q if fwd > K else 0.0
        return -df_q if fwd < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)
    if kind == "call":
        return df_q * _cdf(d1)
    return df_q * (_cdf(d1) - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0) -> float:
    """
    gamma = d2V/dS2 = e^{-qT} n(d1) / (S sigma sqrt(T)).
    Identical for calls and puts. Always >= 0.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return math.exp(-q * T) * _pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float,
            q: float = 0.0) -> float:
    """
    vega = dV/dsigma = S e^{-qT} n(d1) sqrt(T), PER 1.00 of vol.
    Divide by 100 to express per vol point (1%). Same for calls and puts; >=0.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * _pdf(d1) * math.sqrt(T)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call") -> float:
    """
    theta = dV/d(calendar time) = -dV/dT, PER YEAR. Divide by 365 for per-day.

        call: -S e^{-qT} n(d1) sigma / (2 sqrt(T))
              + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)
        put:  -S e^{-qT} n(d1) sigma / (2 sqrt(T))
              - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)
    """
    kind = _norm_kind(kind)
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    decay = -(S * df_q * _pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if kind == "call":
        return decay + q * S * df_q * _cdf(d1) - r * K * df_r * _cdf(d2)
    return decay - q * S * df_q * _cdf(-d1) + r * K * df_r * _cdf(-d2)


def bs_rho(S: float, K: float, T: float, r: float, sigma: float,
           q: float = 0.0, kind: str = "call") -> float:
    """
    rho = dV/dr, PER 1.00 of rate. Divide by 100 for per-1%-rate.
        call:  K T e^{-rT} N(d2)
        put:  -K T e^{-rT} N(-d2)
    """
    kind = _norm_kind(kind)
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r = math.exp(-r * T)
    if kind == "call":
        return K * T * df_r * _cdf(d2)
    return -K * T * df_r * _cdf(-d2)


# ----------------------------------------------------------------------
# Second-order (cross) Greeks: vanna, volga, charm. These are the skew/
# vol-of-vol/time-decay-of-delta risks that a "delta-neutral" book still
# bleeds on. All are derived directly from the BSM density and reconcile
# to finite differences of delta/vega in the self-tests.
# ----------------------------------------------------------------------
def bs_vanna(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0) -> float:
    """
    vanna = dDelta/dsigma = dVega/dS = -e^{-qT} n(d1) d2 / sigma.
    PER 1.00 of vol per $1 of spot (divide by 100 for per vol point).
    Identical for calls and puts. Drives risk-reversal P&L; sign follows -d2,
    so it flips across the money (sign(vanna) = -sign(d2)).
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return -math.exp(-q * T) * _pdf(d1) * d2 / sigma


def bs_volga(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0) -> float:
    """
    volga (vomma) = dVega/dsigma = Vega * d1 * d2 / sigma.
    PER 1.00 of vol (divide by 100 for per vol point). Identical for calls
    and puts; >= 0 wherever d1*d2 >= 0 (i.e. away from the ATM-forward band,
    where it can dip negative). Long strangles/butterflies are long volga -
    they profit from vol-of-vol. Zero at the strikes where d1*d2 = 0.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    veg = bs_vega(S, K, T, r, sigma, q)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return veg * d1 * d2 / sigma


def bs_charm(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call") -> float:
    """
    charm = dDelta/d(calendar time) = -dDelta/d(tau), PER YEAR (tau = time to
    expiry). Divide by 365 for the per-calendar-day delta drift.

        common = e^{-qT} n(d1) * [ (r - q)/(sigma sqrt(T)) - d2/(2T) ]
        call:   q e^{-qT} N(d1)  - common
        put:   -q e^{-qT} N(-d1) - common

    Charm is why a static, "delta-neutral" option position needs re-hedging on
    a flat day: its delta drifts as T shrinks. It blows up near expiry for
    near-ATM strikes (pin risk) and drives open-rebalancing flows.
    """
    kind = _norm_kind(kind)
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)
    common = df_q * _pdf(d1) * ((r - q) / (sigma * math.sqrt(T)) - d2 / (2.0 * T))
    if kind == "call":
        return q * df_q * _cdf(d1) - common
    return -q * df_q * _cdf(-d1) - common


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           q: float = 0.0, kind: str = "call") -> dict:
    """Convenience bundle: price + first- and second-order Greeks as a dict."""
    kind = _norm_kind(kind)
    return {
        "price": bs_price(S, K, T, r, sigma, q, kind),
        "delta": bs_delta(S, K, T, r, sigma, q, kind),
        "gamma": bs_gamma(S, K, T, r, sigma, q),
        "vega": bs_vega(S, K, T, r, sigma, q),
        "theta": bs_theta(S, K, T, r, sigma, q, kind),
        "rho": bs_rho(S, K, T, r, sigma, q, kind),
        "vanna": bs_vanna(S, K, T, r, sigma, q),
        "volga": bs_volga(S, K, T, r, sigma, q),
        "charm": bs_charm(S, K, T, r, sigma, q, kind),
    }


def put_call_parity_gap(call: float, put: float, S: float, K: float,
                        T: float, r: float, q: float = 0.0) -> float:
    """
    Parity residual:  (call - put) - (S e^{-qT} - K e^{-rT}).
    ~0 for arbitrage-free European prices on a non-dividend-adjusted spot.
    A persistent nonzero gap flags a mis-set q/r, a stale spot, or a true
    box-spread edge net of costs.
    """
    return (call - put) - (S * math.exp(-q * T) - K * math.exp(-r * T))


def _no_arb_bounds(S: float, K: float, T: float, r: float, q: float,
                   kind: str):
    """European price bounds: (lower, upper). Used to validate quotes."""
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    if kind == "call":
        lower = max(S * df_q - K * df_r, 0.0)
        upper = S * df_q
    else:
        lower = max(K * df_r - S * df_q, 0.0)
        upper = K * df_r
    return lower, upper


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                q: float = 0.0, kind: str = "call",
                tol: float = 1e-8, max_iter: int = 100) -> float:
    """
    Invert BSM for sigma given a market `price`.

    Strategy: Newton's method using analytic vega, with a bracketing
    bisection fallback for robustness (flat-vega wings, bad initial guess).
    Returns float('nan') if the price is outside no-arbitrage bounds or no
    root can be bracketed in [1e-6, 10.0] vol. Callers MUST handle nan.
    """
    kind = _norm_kind(kind)
    if not (price > 0.0) or T <= 0.0 or S <= 0.0 or K <= 0.0:
        return float("nan")

    lower, upper = _no_arb_bounds(S, K, T, r, q, kind)
    # Allow a hair of slack for floating-point noise on the bounds.
    if price < lower - 1e-12 or price > upper + 1e-12:
        return float("nan")

    lo, hi = 1e-6, 10.0  # vol search bracket (0.0001% .. 1000%)

    # --- Newton iteration ------------------------------------------------
    # Brenner-Subrahmanyam ATM seed, clamped into the bracket.
    sigma = math.sqrt(2.0 * math.pi / T) * (price / S) if S > 0 else 0.2
    if not (lo < sigma < hi) or not math.isfinite(sigma):
        sigma = 0.2
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, q, kind) - price
        if abs(diff) < tol:
            return sigma
        v = bs_vega(S, K, T, r, sigma, q)
        if v < 1e-12:
            break  # vega too small to trust Newton; hand off to bisection
        step = diff / v
        sigma_new = sigma - step
        if not (lo < sigma_new < hi) or not math.isfinite(sigma_new):
            break  # left the bracket; hand off to bisection
        sigma = sigma_new

    # --- Bisection fallback ---------------------------------------------
    f_lo = bs_price(S, K, T, r, lo, q, kind) - price
    f_hi = bs_price(S, K, T, r, hi, q, kind) - price
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        return float("nan")  # not bracketed -> no solution in range
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        f_mid = bs_price(S, K, T, r, mid, q, kind) - price
        if abs(f_mid) < tol or (b - a) < 1e-12:
            return mid
        if f_lo * f_mid < 0.0:
            b = mid
        else:
            a, f_lo = mid, f_mid
    return 0.5 * (a + b)


# ----------------------------------------------------------------------
# Cox-Ross-Rubinstein binomial tree for American (and European) options.
# u = exp(sigma*sqrt(dt)), d = 1/u, risk-neutral up-prob
# p = (exp((r-q)*dt) - d) / (u - d). Backward induction; the American case
# overlays an early-exercise check value = max(intrinsic, continuation).
# Memory is O(n_steps): a single payoff vector overwritten in place.
# ----------------------------------------------------------------------
def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, kind: str = "call", n_steps: int = 500,
              american: bool = False) -> float:
    """
    CRR binomial price. `american=False` gives the European tree (converges to
    bs_price); `american=True` overlays the early-exercise check.

    Raises ValueError if the risk-neutral probability leaves [0, 1] (too few
    steps for the given carry/vol) or for non-positive inputs - a silently
    out-of-range p would produce an arbitrageable, biased price.
    """
    kind = _norm_kind(kind)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if T <= 0.0:
        return _intrinsic_discounted(S, K, T, r, q, kind) if not american \
            else (max(S - K, 0.0) if kind == "call" else max(K - S, 0.0))
    if sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        raise ValueError("crr_price requires S, K, sigma > 0")

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"risk-neutral up-prob {p:.4f} outside [0,1]; increase n_steps "
            f"(need sigma*sqrt(dt) > |(r-q)*dt|)."
        )

    def intrinsic(Sx: float) -> float:
        return max(Sx - K, 0.0) if kind == "call" else max(K - Sx, 0.0)

    # Terminal layer: S_T at node j (j up-moves, n-j down-moves).
    vals = [intrinsic(S * (u ** j) * (d ** (n_steps - j)))
            for j in range(n_steps + 1)]

    # Backward induction.
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * vals[j + 1] + (1.0 - p) * vals[j])
            if american:
                Sij = S * (u ** j) * (d ** (i - j))
                vals[j] = max(intrinsic(Sij), cont)
            else:
                vals[j] = cont
    return vals[0]


def crr_american(S: float, K: float, T: float, r: float, sigma: float,
                 q: float = 0.0, kind: str = "call",
                 n_steps: int = 500) -> float:
    """American option price via the CRR tree (early-exercise overlay)."""
    return crr_price(S, K, T, r, sigma, q, kind, n_steps, american=True)


# ----------------------------------------------------------------------
# Discrete delta-hedging simulator with gamma-theta P&L attribution.
# Iron Law 1 (no look-ahead): the hedge delta for interval [t, t+dt] is
# chosen from start-of-interval information only. Iron Law 3 (costs): a
# per-share cost is charged on every rebalance and on the final unwind.
# ----------------------------------------------------------------------
def delta_hedge_pnl(path, K: float, T: float, r: float, sigma_implied: float,
                    q: float = 0.0, kind: str = "call",
                    rehedge_every: int = 1,
                    cost_per_share: float = 0.0) -> dict:
    """
    Simulate delta-hedging a LONG 1-contract option along a given price
    `path` (a sequence of length n+1 sampled on an equal time grid dt = T/n).

    At each step we mark the option at BSM(sigma_implied) and hold a hedge of
    -delta shares (short the underlying against the long option), rebalanced
    every `rehedge_every` steps using ONLY the price/time known at the start
    of the next interval (no look-ahead). The realized P&L is the running
    delta-hedged mark-to-market:

        pnl_before_cost = sum_i [ dV_i + hedge_shares_i * dS_i ]

    The same path is decomposed via the gamma-theta identity using
    start-of-interval Greeks:

        attribution = sum_i [ 0.5 * Gamma_i * dS_i^2 + Theta_i * dt ]

    In the continuous limit (dt -> 0) attribution == pnl_before_cost; at
    finite dt the residual is exactly the discrete-hedging error (it shrinks
    as dt is refined and is unbiased across random paths). Aggregated, a long
    option's hedged P&L is a bet that realized variance > implied variance:
    positive when realized vol exceeds sigma_implied, negative when it falls
    short - before costs.

    Returns a dict:
        pnl_before_cost   : realized delta-hedged P&L, gross of costs
        total_pnl         : pnl_before_cost - cost_drag
        cost_drag         : sum of |shares traded| * cost_per_share
        attrib_gamma      : sum 0.5 * Gamma * dS^2  (>=0 for long option)
        attrib_theta      : sum Theta * dt          (<=0 for long option)
        attribution_total : attrib_gamma + attrib_theta
        premium           : BSM premium paid at inception (mark, t=0)
        n_rehedges        : number of rebalances executed (incl. unwind)
    """
    kind = _norm_kind(kind)
    n = len(path) - 1
    if n < 1:
        raise ValueError("path must have at least 2 points")
    dt = T / n

    hedge_shares = 0.0
    total_cost = 0.0
    mtm_pnl = 0.0
    attrib_gamma = 0.0
    attrib_theta = 0.0
    n_rehedges = 0

    # Initial hedge at t=0: short delta shares against the long option.
    d0 = bs_delta(path[0], K, T, r, sigma_implied, q, kind)
    target = -d0
    total_cost += abs(target - hedge_shares) * cost_per_share
    hedge_shares = target
    n_rehedges += 1

    for i in range(n):
        S_t = path[i]
        S_next = path[i + 1]
        tau_t = T - i * dt
        tau_next = T - (i + 1) * dt
        dS = S_next - S_t

        # Delta-hedged mark-to-market P&L over the interval.
        V_t = bs_price(S_t, K, tau_t, r, sigma_implied, q, kind)
        V_next = bs_price(S_next, K, tau_next, r, sigma_implied, q, kind)
        mtm_pnl += (V_next - V_t) + hedge_shares * dS

        # Gamma-theta attribution at start-of-interval Greeks (no look-ahead).
        g = bs_gamma(S_t, K, tau_t, r, sigma_implied, q)
        th = bs_theta(S_t, K, tau_t, r, sigma_implied, q, kind)
        attrib_gamma += 0.5 * g * dS * dS
        attrib_theta += th * dt

        # Rebalance for the next interval (skip the terminal step; we unwind).
        if ((i + 1) % rehedge_every == 0) and (i + 1 < n):
            new_delta = bs_delta(S_next, K, tau_next, r, sigma_implied, q, kind)
            new_target = -new_delta
            total_cost += abs(new_target - hedge_shares) * cost_per_share
            hedge_shares = new_target
            n_rehedges += 1

    # Unwind the residual hedge at expiry.
    total_cost += abs(hedge_shares) * cost_per_share
    if hedge_shares != 0.0:
        n_rehedges += 1

    return {
        "pnl_before_cost": mtm_pnl,
        "total_pnl": mtm_pnl - total_cost,
        "cost_drag": total_cost,
        "attrib_gamma": attrib_gamma,
        "attrib_theta": attrib_theta,
        "attribution_total": attrib_gamma + attrib_theta,
        "premium": bs_price(path[0], K, T, r, sigma_implied, q, kind),
        "n_rehedges": n_rehedges,
    }


# ----------------------------------------------------------------------
# Self-tests: analytic identities + finite-difference checks on fixed
# parameter sets, CRR convergence, and a seeded delta-hedge simulation.
# Deterministic (seeded). Run `python options.py`.
# ----------------------------------------------------------------------
if __name__ == "__main__":

    def _fd_delta(S, K, T, r, sigma, q, kind, h=1e-4):
        up = bs_price(S + h, K, T, r, sigma, q, kind)
        dn = bs_price(S - h, K, T, r, sigma, q, kind)
        return (up - dn) / (2.0 * h)

    def _fd_gamma(S, K, T, r, sigma, q, kind, h=None):
        # scale the bump with spot: a fixed absolute h is too coarse for small-S
        # (e.g. FX) underlyings, which inflates the second-difference error
        if h is None:
            h = 1e-3 * max(abs(S), 1.0)
        up = bs_price(S + h, K, T, r, sigma, q, kind)
        md = bs_price(S, K, T, r, sigma, q, kind)
        dn = bs_price(S - h, K, T, r, sigma, q, kind)
        return (up - 2.0 * md + dn) / (h * h)

    def _fd_vega(S, K, T, r, sigma, q, kind, h=1e-5):
        up = bs_price(S, K, T, r, sigma + h, q, kind)
        dn = bs_price(S, K, T, r, sigma - h, q, kind)
        return (up - dn) / (2.0 * h)

    def _fd_vanna_dDelta_dsig(S, K, T, r, sigma, q, h=1e-5):
        up = bs_delta(S, K, T, r, sigma + h, q, "call")
        dn = bs_delta(S, K, T, r, sigma - h, q, "call")
        return (up - dn) / (2.0 * h)

    def _fd_vanna_dVega_dS(S, K, T, r, sigma, q, h=None):
        if h is None:
            h = 1e-3 * max(abs(S), 1.0)
        up = bs_vega(S + h, K, T, r, sigma, q)
        dn = bs_vega(S - h, K, T, r, sigma, q)
        return (up - dn) / (2.0 * h)

    def _fd_volga(S, K, T, r, sigma, q, h=1e-5):
        up = bs_vega(S, K, T, r, sigma + h, q)
        dn = bs_vega(S, K, T, r, sigma - h, q)
        return (up - dn) / (2.0 * h)

    def _fd_charm(S, K, T, r, sigma, q, kind, h=1e-6):
        # charm = dDelta/d(calendar) = -dDelta/d(tau). FD bumps tau (=T here).
        up = bs_delta(S, K, T + h, r, sigma, q, kind)
        dn = bs_delta(S, K, T - h, r, sigma, q, kind)
        return -(up - dn) / (2.0 * h)

    # A spread of cases: ITM / ATM / OTM, with and without dividend yield,
    # short and long dated, plus an FX-style (r != q) case.
    cases = [
        # S,    K,    T,    r,     sigma, q
        (100., 100., 1.00, 0.03, 0.20, 0.00),  # textbook ATM
        (100., 110., 0.50, 0.05, 0.25, 0.02),  # OTM call / ITM put, divs
        (120.,  90., 2.00, 0.01, 0.35, 0.04),  # deep ITM call, long dated
        ( 50.,  60., 0.25, 0.04, 0.40, 0.00),  # short-dated OTM call
        (1.25, 1.30, 1.00, 0.04, 0.10, 0.01),  # FX-style: r_dom=4%, r_for=1%
    ]

    for (S, K, T, r, sigma, q) in cases:
        c = bs_price(S, K, T, r, sigma, q, "call")
        p = bs_price(S, K, T, r, sigma, q, "put")

        # 1) Put-call parity residual ~ 0.
        gap = put_call_parity_gap(c, p, S, K, T, r, q)
        assert abs(gap) < 1e-9, f"parity gap {gap} for {(S,K,T,r,sigma,q)}"

        # 2) Implied-vol round trip recovers the input sigma.
        iv_c = implied_vol(c, S, K, T, r, q, "call")
        iv_p = implied_vol(p, S, K, T, r, q, "put")
        assert math.isfinite(iv_c) and abs(iv_c - sigma) < 1e-4, \
            f"call IV {iv_c} vs {sigma}"
        assert math.isfinite(iv_p) and abs(iv_p - sigma) < 1e-4, \
            f"put IV {iv_p} vs {sigma}"

        # 3) Delta ranges.
        dc = bs_delta(S, K, T, r, sigma, q, "call")
        dp = bs_delta(S, K, T, r, sigma, q, "put")
        assert 0.0 <= dc <= 1.0, f"call delta out of range: {dc}"
        assert -1.0 <= dp <= 0.0, f"put delta out of range: {dp}"

        # 4) Gamma and vega strictly positive for live options.
        g = bs_gamma(S, K, T, r, sigma, q)
        vg = bs_vega(S, K, T, r, sigma, q)
        assert g > 0.0, f"gamma not positive: {g}"
        assert vg > 0.0, f"vega not positive: {vg}"

        # 5) Finite-difference vs analytic first-order Greeks.
        assert abs(dc - _fd_delta(S, K, T, r, sigma, q, "call")) < 1e-3
        assert abs(dp - _fd_delta(S, K, T, r, sigma, q, "put")) < 1e-3
        assert abs(g - _fd_gamma(S, K, T, r, sigma, q, "call")) < 1e-3
        assert abs(vg - _fd_vega(S, K, T, r, sigma, q, "call")) < 1e-3
        # gamma/vega are identical across call & put - cross-check.
        assert abs(g - bs_gamma(S, K, T, r, sigma, q)) < 1e-12
        assert abs(vg - bs_vega(S, K, T, r, sigma, q)) < 1e-12

        # 6) Second-order Greeks vs finite differences.
        vn = bs_vanna(S, K, T, r, sigma, q)
        # vanna == dDelta/dsigma == dVega/dS (both definitions must match).
        assert abs(vn - _fd_vanna_dDelta_dsig(S, K, T, r, sigma, q)) < 1e-4, \
            f"vanna vs dDelta/dsigma: {vn}"
        assert abs(vn - _fd_vanna_dVega_dS(S, K, T, r, sigma, q)) < 1e-3, \
            f"vanna vs dVega/dS: {vn}"
        vl = bs_volga(S, K, T, r, sigma, q)
        assert abs(vl - _fd_volga(S, K, T, r, sigma, q)) < 1e-3, \
            f"volga vs fd: {vl}"
        # volga is call/put symmetric (depends only on vega, d1, d2).
        assert abs(vl - bs_volga(S, K, T, r, sigma, q)) < 1e-12
        ch_c = bs_charm(S, K, T, r, sigma, q, "call")
        ch_p = bs_charm(S, K, T, r, sigma, q, "put")
        assert abs(ch_c - _fd_charm(S, K, T, r, sigma, q, "call")) < 1e-4, \
            f"charm call vs fd: {ch_c}"
        assert abs(ch_p - _fd_charm(S, K, T, r, sigma, q, "put")) < 1e-4, \
            f"charm put vs fd: {ch_p}"

    # 7) Deep-ITM call ~ S e^{-qT} - K e^{-rT} (option ~ forward, N(d.)~1).
    S, K, T, r, sigma, q = 200.0, 50.0, 1.0, 0.03, 0.20, 0.02
    c = bs_price(S, K, T, r, sigma, q, "call")
    intrinsic_fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(c - intrinsic_fwd) < 1e-4, \
        f"deep-ITM call {c} vs forward intrinsic {intrinsic_fwd}"
    # ...and its delta ~ e^{-qT}.
    assert abs(bs_delta(S, K, T, r, sigma, q, "call")
               - math.exp(-q * T)) < 1e-6

    # 8) Theta of a long ATM option is negative (time decay) under our sign.
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.03, 0.20, 0.0
    th_c = bs_theta(S, K, T, r, sigma, q, "call")
    th_p = bs_theta(S, K, T, r, sigma, q, "put")
    assert th_c < 0.0, f"ATM call theta not negative: {th_c}"
    assert th_p < 0.0, f"ATM put theta not negative: {th_p}"

    # 9) forward_price identity and the parity RHS line up.
    F = forward_price(100.0, 0.05, 0.02, 0.75)
    assert abs(F - 100.0 * math.exp((0.05 - 0.02) * 0.75)) < 1e-12

    # 10) implied_vol returns nan for an arbitrage-violating quote
    #     (price below intrinsic) rather than a bogus root.
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.03, 0.0
    below_intrinsic = max(S - K, 0.0) - 1.0  # clearly below any valid price
    assert math.isnan(implied_vol(below_intrinsic, S, K, T, r, q, "call"))

    # 11) Degenerate inputs (T<=0, sigma<=0) return discounted intrinsic,
    #     not nan / ZeroDivisionError.
    assert abs(bs_price(110.0, 100.0, 0.0, 0.03, 0.20, 0.0, "call")
               - 10.0) < 1e-12
    assert bs_gamma(100.0, 100.0, 0.0, 0.03, 0.20) == 0.0
    assert bs_vega(100.0, 100.0, 0.0, 0.03, 0.20) == 0.0
    assert bs_vanna(100.0, 100.0, 0.0, 0.03, 0.20) == 0.0
    assert bs_volga(100.0, 100.0, 0.0, 0.03, 0.20) == 0.0
    assert bs_charm(100.0, 100.0, 0.0, 0.03, 0.20, 0.0, "call") == 0.0

    # 12) CRR European tree converges to BSM (<1e-2 at n=500), and an American
    #     call on a NON-dividend stock equals the European value (never early-
    #     exercise a call without dividends).
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    bsm_c = bs_price(S, K, T, r, sigma, q, "call")
    crr_e = crr_price(S, K, T, r, sigma, q, "call", 500, american=False)
    crr_a = crr_american(S, K, T, r, sigma, q, "call", 500)
    assert abs(crr_e - bsm_c) < 1e-2, f"CRR-Euro {crr_e} vs BSM {bsm_c}"
    assert abs(crr_a - crr_e) < 1e-9, \
        f"American call (no div) {crr_a} != European {crr_e}"

    # 13) American >= European always; and with r>0 an American PUT carries a
    #     strictly positive early-exercise premium over the European put.
    for kind in ("call", "put"):
        a = crr_american(80.0, 100.0, 1.0, 0.06, 0.25, 0.03, kind, 400)
        e = crr_price(80.0, 100.0, 1.0, 0.06, 0.25, 0.03, kind, 400, False)
        assert a >= e - 1e-9, f"American {kind} {a} < European {e}"
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.08, 0.30, 0.0
    ap = crr_american(S, K, T, r, sigma, q, "put", 500)
    ep = crr_price(S, K, T, r, sigma, q, "put", 500, american=False)
    assert ap > ep + 1e-3, \
        f"American put premium not positive: {ap} vs {ep}"
    # European CRR put still tracks BSM.
    assert abs(ep - bs_price(S, K, T, r, sigma, q, "put")) < 1e-2

    # 14) CRR raises (not silently biases) when p leaves [0,1] (huge carry,
    #     too few steps).
    raised = False
    try:
        crr_price(100.0, 100.0, 1.0, 5.0, 0.05, 0.0, "call", 2)
    except ValueError:
        raised = True
    assert raised, "CRR should raise when risk-neutral prob is out of [0,1]"

    # ------------------------------------------------------------------
    # Delta-hedging simulator. Seeded GBM paths; check the economics and the
    # gamma-theta reconciliation. r=q=0 so no financing term contaminates the
    # delta-hedged identity. Uses only the stdlib `random` (numpy not assumed).
    # ------------------------------------------------------------------
    import random
    import statistics as _st

    def _gbm_path(S0, vol, T, n, rng):
        """Driftless GBM path (mu=0) with realized vol `vol`."""
        dt = T / n
        s = S0
        out = [s]
        root = vol * math.sqrt(dt)
        half = 0.5 * vol * vol * dt
        for _ in range(n):
            z = rng.gauss(0.0, 1.0)
            s = s * math.exp(-half + root * z)
            out.append(s)
        return out

    K, T, r, q = 100.0, 1.0, 0.0, 0.0
    sig_imp = 0.20

    # 15) Long option hedged at implied=0.20 on HIGH realized-vol paths
    #     (0.40) makes positive P&L before costs; on LOW realized-vol paths
    #     (0.10) it loses. This is the realized-vs-implied variance bet.
    rng = random.Random(0)
    hi_pnls = [delta_hedge_pnl(_gbm_path(100.0, 0.40, T, 252, rng), K, T, r,
                               sig_imp, q, "call", 1, 0.0)["pnl_before_cost"]
               for _ in range(200)]
    assert _st.mean(hi_pnls) > 0.0, \
        f"long option on high-vol paths should profit pre-cost: {_st.mean(hi_pnls)}"

    rng = random.Random(7)
    lo_pnls = [delta_hedge_pnl(_gbm_path(100.0, 0.10, T, 252, rng), K, T, r,
                               sig_imp, q, "call", 1, 0.0)["pnl_before_cost"]
               for _ in range(200)]
    assert _st.mean(lo_pnls) < 0.0, \
        f"long option on low-vol paths should lose pre-cost: {_st.mean(lo_pnls)}"

    # 16) Gamma-theta attribution reconciles to realized P&L: it is UNBIASED
    #     across random paths (mean residual ~ 0), and the residual SHRINKS as
    #     dt is refined (the identity 0.5*Gamma*dS^2 + Theta*dt is the dt->0
    #     limit; the residual is the discrete-hedging error). Iron Law 1 & 3.
    rng = random.Random(0)
    resid = []
    for _ in range(300):
        path = _gbm_path(100.0, 0.40, T, 504, rng)
        res = delta_hedge_pnl(path, K, T, r, sig_imp, q, "call", 1, 0.0)
        resid.append(res["pnl_before_cost"] - res["attribution_total"])
    assert abs(_st.mean(resid)) < 0.05, \
        f"gamma-theta attribution biased: mean residual {_st.mean(resid)}"

    # Refinement: mean |residual| on a fixed smooth path roughly halves when dt
    # is quartered (per-step Taylor error -> 0). Deterministic, no RNG.
    def _smooth_path(n, amp=0.03):
        return [100.0 * math.exp(amp * math.sin(2.0 * math.pi * i / n))
                for i in range(n + 1)]
    errs = []
    for n in (500, 2000, 8000):
        res = delta_hedge_pnl(_smooth_path(n), K, T, r, sig_imp, q, "call", 1, 0.0)
        errs.append(abs(res["pnl_before_cost"] - res["attribution_total"]))
    assert errs[0] > errs[1] > errs[2], \
        f"reconciliation error should shrink as dt->0: {errs}"
    # At the finest grid the residual is a small fraction of the premium.
    prem = bs_price(100.0, K, T, r, sig_imp, q, "call")
    assert errs[-1] / prem < 0.02, \
        f"fine-grid reconciliation error too large: {errs[-1]/prem}"
    # And the attribution split has the right signs for a long option.
    res = delta_hedge_pnl(_gbm_path(100.0, 0.40, T, 252, random.Random(3)),
                          K, T, r, sig_imp, q, "call", 1, 0.0)
    assert res["attrib_gamma"] > 0.0 and res["attrib_theta"] < 0.0, \
        f"long-option attribution signs wrong: {res}"

    # 17) Costs (Iron Law 3): a positive per-share cost strictly reduces P&L,
    #     and rehedging MORE often costs MORE (more rebalances).
    path = _gbm_path(100.0, 0.40, T, 252, random.Random(11))
    free = delta_hedge_pnl(path, K, T, r, sig_imp, q, "call", 1, 0.0)
    charged = delta_hedge_pnl(path, K, T, r, sig_imp, q, "call", 1, 0.02)
    assert charged["cost_drag"] > 0.0
    assert abs(charged["pnl_before_cost"] - free["pnl_before_cost"]) < 1e-12
    assert charged["total_pnl"] < charged["pnl_before_cost"]
    daily = delta_hedge_pnl(path, K, T, r, sig_imp, q, "call", 1, 0.02)
    monthly = delta_hedge_pnl(path, K, T, r, sig_imp, q, "call", 21, 0.02)
    assert daily["cost_drag"] > monthly["cost_drag"], \
        "more frequent rehedging should cost more"

    # 18) greeks() bundle is internally consistent with the standalone funcs.
    gk = greeks(100.0, 100.0, 1.0, 0.03, 0.20, 0.0, "call")
    assert abs(gk["vanna"] - bs_vanna(100.0, 100.0, 1.0, 0.03, 0.20, 0.0)) < 1e-12
    assert abs(gk["charm"] - bs_charm(100.0, 100.0, 1.0, 0.03, 0.20, 0.0, "call")) < 1e-12

    print("options.py: all self-tests passed.")
