"""
Black-Scholes-Merton (BSM) option pricing, Greeks, and implied volatility.

Self-contained: standard library + (optional) numpy only. No scipy.
The normal CDF comes from statistics.NormalDist; the pdf is written out
analytically. Everything is closed-form except implied_vol, which is a
Newton iteration with a robust bisection fallback.

Conventions
-----------
S      : spot price of the underlying (>0)
K      : strike (>0)
T      : time to expiry in YEARS (Act/365 or Act/252 — pick one and be
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

Pitfalls (detect/fix)
---------------------
* T<=0 or sigma<=0: the formulas divide by sigma*sqrt(T). We handle the
  degenerate cases explicitly and return the discounted intrinsic / forward
  payoff rather than producing nan or a ZeroDivisionError.
* vega is reported per 1.00 vol, not per vol point. A "1% move" is 0.01.
  Mixing the two is the most common Greek-scaling bug.
* theta sign: a long ATM option loses value as time passes, so theta<0 here.
  If you store "theta per day" remember to keep the sign.
* implied_vol can fail to bracket when the quoted price violates no-arb
  bounds (below intrinsic or above the underlying). We return nan instead
  of a bogus root, so callers must check for nan.
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
# Self-tests: analytic identities + finite-difference checks on fixed
# parameter sets (deterministic, no RNG). Run `python options.py`.
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

        # 5) Finite-difference vs analytic Greeks.
        assert abs(dc - _fd_delta(S, K, T, r, sigma, q, "call")) < 1e-3
        assert abs(dp - _fd_delta(S, K, T, r, sigma, q, "put")) < 1e-3
        assert abs(g - _fd_gamma(S, K, T, r, sigma, q, "call")) < 1e-3
        assert abs(vg - _fd_vega(S, K, T, r, sigma, q, "call")) < 1e-3
        # gamma/vega are identical across call & put — cross-check.
        assert abs(g - bs_gamma(S, K, T, r, sigma, q)) < 1e-12
        assert abs(vg - bs_vega(S, K, T, r, sigma, q)) < 1e-12

    # 6) Deep-ITM call ~ S e^{-qT} - K e^{-rT} (option ~ forward, N(d.)~1).
    S, K, T, r, sigma, q = 200.0, 50.0, 1.0, 0.03, 0.20, 0.02
    c = bs_price(S, K, T, r, sigma, q, "call")
    intrinsic_fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(c - intrinsic_fwd) < 1e-4, \
        f"deep-ITM call {c} vs forward intrinsic {intrinsic_fwd}"
    # ...and its delta ~ e^{-qT}.
    assert abs(bs_delta(S, K, T, r, sigma, q, "call")
               - math.exp(-q * T)) < 1e-6

    # 7) Theta of a long ATM option is negative (time decay) under our sign.
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.03, 0.20, 0.0
    th_c = bs_theta(S, K, T, r, sigma, q, "call")
    th_p = bs_theta(S, K, T, r, sigma, q, "put")
    assert th_c < 0.0, f"ATM call theta not negative: {th_c}"
    assert th_p < 0.0, f"ATM put theta not negative: {th_p}"

    # 8) forward_price identity and the parity RHS line up.
    F = forward_price(100.0, 0.05, 0.02, 0.75)
    assert abs(F - 100.0 * math.exp((0.05 - 0.02) * 0.75)) < 1e-12

    # 9) implied_vol returns nan for an arbitrage-violating quote
    #    (price below intrinsic) rather than a bogus root.
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.03, 0.0
    below_intrinsic = max(S - K, 0.0) - 1.0  # clearly below any valid price
    assert math.isnan(implied_vol(below_intrinsic, S, K, T, r, q, "call"))

    # 10) Degenerate inputs (T<=0, sigma<=0) return discounted intrinsic,
    #     not nan / ZeroDivisionError.
    assert abs(bs_price(110.0, 100.0, 0.0, 0.03, 0.20, 0.0, "call")
               - 10.0) < 1e-12
    assert bs_gamma(100.0, 100.0, 0.0, 0.03, 0.20) == 0.0
    assert bs_vega(100.0, 100.0, 0.0, 0.03, 0.20) == 0.0

    print("options.py: all self-tests passed.")
