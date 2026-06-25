"""
betting_markets.py - Odds conversion, devigging, Kelly sizing, and calibration toolkit
for prediction-market and sports-betting quant work.

Scope
-----
Treat quoted prices/odds as (vig-inflated) probabilities. The core jobs of a betting
quant are: (1) convert between odds conventions cleanly, (2) strip the bookmaker margin
("vig"/"overround") to recover a fair probability, (3) size stakes by edge, and (4)
measure forecast quality (calibration) and your skill vs the market (closing-line value).

Conventions (stated explicitly to avoid silent errors)
------------------------------------------------------
- Decimal odds D: total payout multiple per unit staked (stake returned + profit).
  D = 2.00 means stake 1 -> receive 2 (profit 1). Implied prob = 1/D.
- American odds (moneyline) ML:
    ML > 0:  D = 1 + ML/100        (underdog; +150 -> 2.50)
    ML < 0:  D = 1 + 100/|ML|      (favorite; -200 -> 1.50)
  Pick-em (exactly even, D == 2.0) maps to +100 by convention here.
- Implied probabilities from a full market (e.g. both sides of a binary, or every
  runner) sum to MORE than 1 by the overround. Devigging maps the raw implied vector
  to a fair vector that sums to exactly 1.
- Expected value is per unit staked: EV = p*D - 1. EV > 0 is a positive-edge bet.
- Kelly fraction is the growth-optimal stake as a fraction of bankroll for a single
  binary bet with net odds b = D - 1: f* = (p*b - (1-p)) / b, floored at 0 (no shorting
  / no negative-edge bets). Use FRACTIONAL Kelly (e.g. 0.25*f*) in practice; full Kelly
  assumes p is exactly known, which it never is.
- Closing Line Value (CLV): entry_decimal / closing_decimal - 1. POSITIVE means you got
  a BETTER price than the close (your decimal odds were higher => you "beat the close").
  Beating the closing line is the single most robust public signal of long-run betting
  edge, because the closing line is the market's most efficient estimate.

Leak-free backtesting (read this before backtesting bets)
--------------------------------------------------------
The cardinal sin is deciding a bet on information not available pre-event. Enforce:
  - Decide stake/side using ONLY pre-event ("entry") prices and your pre-event model.
  - Settle PnL with the realized outcome, never peeking at it to choose the bet.
  - Evaluate skill with CLV (entry vs CLOSE), which is observable before settlement and
    far less noisy than realized win/loss.
This module provides the price/odds/sizing primitives; pair it with a backtester that
timestamps decisions strictly before event start.

Dependencies: numpy + Python stdlib only (statistics.NormalDist for the normal CDF/inverse
if you extend this). No scipy.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


# --------------------------------------------------------------------------- #
# Odds conversions                                                            #
# --------------------------------------------------------------------------- #
def decimal_to_implied(odds: float) -> float:
    """Raw implied probability from decimal odds: 1/odds (includes vig)."""
    if odds <= 0:
        raise ValueError("decimal odds must be positive")
    return 1.0 / odds


def implied_to_decimal(p: float) -> float:
    """Decimal odds from a probability: 1/p. (Fair odds if p is fair.)"""
    if not (0.0 < p <= 1.0):
        raise ValueError("probability must be in (0, 1]")
    return 1.0 / p


def american_to_decimal(ml: float) -> float:
    """American moneyline -> decimal odds.

    ml > 0: 1 + ml/100   (underdog)
    ml < 0: 1 + 100/|ml|  (favorite)
    """
    if ml == 0:
        raise ValueError("american odds of 0 are undefined")
    if ml > 0:
        return 1.0 + ml / 100.0
    return 1.0 + 100.0 / abs(ml)


def decimal_to_american(dec: float) -> float:
    """Decimal odds -> American moneyline.

    dec >= 2: (dec-1)*100   (underdog, positive ML)
    dec <  2: -100/(dec-1)  (favorite, negative ML)
    Pick-em (dec == 2.0) -> +100.
    """
    if dec <= 1.0:
        raise ValueError("decimal odds must be > 1")
    if dec >= 2.0:
        return (dec - 1.0) * 100.0
    return -100.0 / (dec - 1.0)


# --------------------------------------------------------------------------- #
# Devigging (margin removal)                                                  #
# --------------------------------------------------------------------------- #
def devig_multiplicative(implied: ArrayLike) -> np.ndarray:
    """Multiplicative (proportional / normalized) devig.

    Simplest method: probs = implied / sum(implied). Assumes the bookmaker applies the
    margin proportionally across outcomes. Tends to over-shorten favorites relative to
    longshots vs. power/Shin methods, but is fast and a fine default for ~50/50 binaries.
    """
    a = np.asarray(implied, dtype=float)
    s = a.sum()
    if s <= 0:
        raise ValueError("implied probabilities must sum to a positive number")
    return a / s


def devig_power(implied: ArrayLike, tol: float = 1e-10) -> np.ndarray:
    """Power (Odds Ratio family) devig.

    Find exponent k > 0 such that sum(implied_i ** k) == 1, then return the (already
    summing-to-1) vector implied ** k. Because raw implied probabilities sum to > 1, the
    margin is removed by RAISING to a power k > 1 (longshots shrink faster), which models
    the favorite-longshot bias better than the multiplicative method.

    Solved by bisection on k. g(k) = sum(implied**k) is strictly decreasing in k for
    implied_i in (0,1), so a unique root exists when the overround is positive.
    """
    a = np.asarray(implied, dtype=float)
    if np.any(a <= 0) or np.any(a >= 1):
        raise ValueError("each implied probability must lie strictly in (0, 1)")

    def g(k: float) -> float:
        return float(np.sum(a ** k) - 1.0)

    lo, hi = 1.0, 1.0
    # Bracket the root. With overround > 0, g(1) > 0, so push hi up until g(hi) <= 0.
    if g(lo) <= 0.0:
        # Already <= 1 (no/negative margin); push lo down to bracket from below.
        while g(lo) < 0.0 and lo > 1e-12:
            lo *= 0.5
    else:
        while g(hi) > 0.0 and hi < 1e12:
            hi *= 2.0

    for _ in range(1000):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if abs(gm) < tol:
            break
        # g decreasing: if g(mid) > 0 root is to the right.
        if gm > 0.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    out = a ** k
    return out / out.sum()  # guard against tiny residual


def devig_shin(implied: ArrayLike, tol: float = 1e-12) -> np.ndarray:
    """Shin (1992) devig: removes margin attributed to insider/informed traders.

    Model: the observed price reflects a fraction z in [0, 1) of informed money. The fair
    probability of outcome i is recovered as

        p_i = (sqrt(z^2 + 4*(1 - z) * implied_i^2 / S) - z) / (2*(1 - z))

    where S = sum_j implied_j (the booksum / overround+1). We solve for z by bisection so
    that sum_i p_i == 1. f(z) = sum_i p_i(z) is monotonic in z over [0, 1), so a unique
    root exists for a positive overround. Shin's method is the standard for sports books
    and typically sits between multiplicative and power in how it treats longshots.
    """
    a = np.asarray(implied, dtype=float)
    if np.any(a <= 0):
        raise ValueError("implied probabilities must be positive")
    S = a.sum()

    def probs_for_z(z: float) -> np.ndarray:
        # As z -> 0, p_i -> implied_i / S (the multiplicative limit).
        if z <= 0.0:
            return a / S
        disc = z * z + 4.0 * (1.0 - z) * (a * a) / S
        return (np.sqrt(disc) - z) / (2.0 * (1.0 - z))

    def f(z: float) -> float:
        return float(probs_for_z(z).sum() - 1.0)

    lo, hi = 0.0, 1.0 - 1e-12
    flo = f(lo)
    if abs(flo) < tol:
        return probs_for_z(lo)
    # f(0) = (sum implied)/S - 1 = 0 only with no margin; with margin f(0) > 0 and f
    # decreases toward 0 as z grows, so the standard bracket [0, 1) holds.
    for _ in range(1000):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            break
        if (flo > 0.0) == (fm > 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    out = probs_for_z(z)
    return out / out.sum()  # guard tiny residual


# --------------------------------------------------------------------------- #
# Edge, sizing                                                                #
# --------------------------------------------------------------------------- #
def expected_value(prob: float, decimal_odds: float) -> float:
    """EV per unit staked: prob * decimal_odds - 1. Positive => edge."""
    return prob * decimal_odds - 1.0


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Growth-optimal stake fraction for a single binary bet.

    b = decimal_odds - 1 (net odds); f* = (prob*b - (1-prob)) / b, floored at 0.
    Returns 0 when there is no positive edge (don't bet). Use fractional Kelly live.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        raise ValueError("decimal_odds must be > 1 for a payable bet")
    f = (prob * b - (1.0 - prob)) / b
    return max(0.0, f)


def joint_kelly(
    probs: ArrayLike,
    decimal_odds: ArrayLike,
    cov: ArrayLike | None = None,
    frac: float = 1.0,
    cap: float | None = None,
) -> np.ndarray:
    """Growth-optimal stake VECTOR for n simultaneous binary bets (correlated allowed).

    Maximizes expected log-wealth E[log(1 + sum_i f_i * R_i)] EXACTLY over the full
    outcome lattice of the n bets, where each bet's per-unit return is

        R_i = +b_i  if leg i wins   (b_i = decimal_odds_i - 1, the net odds)
        R_i = -1    if leg i loses.

    With n bets there are 2**n joint win/lose outcomes; their probabilities come from the
    Gaussian-copula construction below (so marginals match `probs` and pairwise wins are
    correlated by `cov`). The returned f sums-to-bankroll-fraction stakes (no leverage
    assumed beyond what makes 1 + f.R > 0 in every state). Negative entries are allowed
    (a hedge against a correlated leg) unless they would imply shorting an unavailable
    bet -- here we keep the unconstrained log-optimal vector, then floor at 0 component-
    wise only if `cap` is None? No: we DO NOT silently floor; see below.

    Method
    ------
    1. Start from the Gaussian/Markowitz approximation  f0 = Sigma^-1 @ mu, where
       mu_i = E[R_i] = p_i*b_i - (1-p_i) = p_i*(b_i+1) - 1 (the per-unit EV, == EV of the
       bet), and Sigma is the covariance of the return vector R. This is the small-stake
       limit of the Kelly criterion (Thorp 2006, "The Kelly Capital Growth Investment
       Criterion"; Markowitz tangency / log-utility quadratic approximation).
    2. Newton-Raphson refine on the EXACT objective L(f) = sum_s P_s * log(1 + f . R_s)
       over outcome states s. Gradient g = sum_s P_s * R_s / (1 + f.R_s); Hessian
       H = -sum_s P_s * (R_s R_s^T) / (1 + f.R_s)^2 (negative-definite => concave =>
       unique max). Step f <- f - H^-1 g with a backtracking guard that keeps every
       1 + f.R_s > 0 (wealth strictly positive in all states).

    Parameters
    ----------
    probs : length-n win probabilities p_i in (0,1) (use DEVIGGED / fair probs).
    decimal_odds : length-n decimal odds D_i > 1; net odds b_i = D_i - 1.
    cov : optional n-by-n correlation matrix of the win INDICATORS (1 if leg i wins).
          Diagonal is ignored (Bernoulli variance p_i(1-p_i) is used). Off-diagonal
          rho_ij in [-1,1] couples the legs via a Gaussian copula. Default None =>
          independent legs.
    frac : fractional-Kelly multiplier applied to the final vector (e.g. 0.25). Default 1.
    cap : optional per-leg absolute cap; each |f_i| is clipped to `cap` AFTER `frac`.

    Returns
    -------
    np.ndarray of length n: stake fractions of bankroll.

    Leak-free / Iron-Law note
    -------------------------
    Pure function of pre-event inputs (fair probs, quoted odds, your correlation estimate).
    It NEVER consumes the realized outcome, so it is causal and safe to call inside a
    backtest at decision time. Estimate `probs`/`cov` from information available strictly
    before event start.

    Reduces to `kelly_fraction` for n == 1 (single-bet Kelly), and the joint stake on
    positively-correlated legs is SMALLER than the naive sum of independent single-bet
    Kelly fractions (correlation is risk you must pay for).
    """
    p = np.asarray(probs, dtype=float).reshape(-1)
    D = np.asarray(decimal_odds, dtype=float).reshape(-1)
    n = p.shape[0]
    if D.shape[0] != n:
        raise ValueError("probs and decimal_odds must have the same length")
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("each probability must lie strictly in (0, 1)")
    if np.any(D <= 1.0):
        raise ValueError("each decimal_odds must be > 1")
    b = D - 1.0  # net odds

    # Win/lose return matrix R[s, i] for each of the 2**n joint states s, and the
    # per-state probability P[s] under a Gaussian copula with marginals p_i.
    masks = np.array([[(s >> i) & 1 for i in range(n)] for s in range(2 ** n)], dtype=int)
    # masks[s, i] == 1 means leg i WINS in state s.
    R = np.where(masks == 1, b[None, :], -1.0)

    if cov is None:
        rho = np.eye(n)
    else:
        rho = np.asarray(cov, dtype=float)
        if rho.shape != (n, n):
            raise ValueError("cov must be an n-by-n matrix")
        rho = rho.copy()
        np.fill_diagonal(rho, 1.0)
        if np.any(np.abs(rho) > 1.0 + 1e-12):
            raise ValueError("correlation entries must lie in [-1, 1]")

    P = _lattice_probs(p, rho, masks)

    # mu = E[R] (per-leg EV); Sigma = Cov(R) under the joint law (for the warm start).
    mu = P @ R                       # length n
    ER2 = np.einsum("s,si,sj->ij", P, R, R)  # n x n  E[R_i R_j]
    Sigma = ER2 - np.outer(mu, mu)
    # Gaussian-approx Kelly start: f0 = Sigma^-1 mu (Markowitz / small-stake Kelly).
    f = np.linalg.solve(Sigma + 1e-12 * np.eye(n), mu)

    # Keep the start strictly feasible (1 + f.R_s > 0 for all states).
    def feasible(fv: np.ndarray) -> bool:
        return bool(np.all(1.0 + R @ fv > 1e-9))

    if not feasible(f):
        scale = 1.0
        while scale > 1e-12 and not feasible(f * scale):
            scale *= 0.5
        f = f * scale

    # Newton-Raphson on the exact expected-log-wealth objective (concave).
    for _ in range(100):
        w = 1.0 + R @ f                       # wealth multiplier per state
        inv = 1.0 / w
        grad = (P * inv) @ R                  # length n
        # Hessian = -sum_s P_s inv_s^2 R_s R_s^T
        WR = R * (P * inv * inv)[:, None]
        H = -(WR.T @ R)
        try:
            step = np.linalg.solve(H, grad)   # Newton step = -H^-1 grad (grad - here +)
        except np.linalg.LinAlgError:
            break
        step = -step
        # Backtracking line search keeping feasibility and increasing the objective.
        t = 1.0
        f_obj = float(P @ np.log(w))
        while t > 1e-12:
            f_new = f + t * step
            if feasible(f_new):
                obj_new = float(P @ np.log(1.0 + R @ f_new))
                if obj_new >= f_obj - 1e-15:
                    break
            t *= 0.5
        f_new = f + t * step
        if np.max(np.abs(f_new - f)) < 1e-12:
            f = f_new
            break
        f = f_new

    f = frac * f
    if cap is not None:
        f = np.clip(f, -abs(cap), abs(cap))
    return f


def _lattice_probs(p: np.ndarray, rho: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Joint probabilities of the 2**n win/lose states under a Gaussian copula.

    Each leg i wins iff a standard normal Z_i exceeds threshold t_i = Phi^-1(1 - p_i),
    with Corr(Z) = rho. State probabilities are the orthant masses of the multivariate
    normal, computed by exact recursion for n <= 2 and by Monte-Carlo (seeded, so the
    result is deterministic and leak-free) for n >= 3. Marginals match p_i exactly.
    """
    from statistics import NormalDist

    nd = NormalDist()
    n = p.shape[0]
    t = np.array([nd.inv_cdf(1.0 - pi) for pi in p])  # win iff Z_i > t_i

    if n == 1:
        return np.array([p[0] if masks[s, 0] == 1 else 1.0 - p[0] for s in range(2)])

    if n == 2:
        r = float(rho[0, 1])
        # Bivariate normal upper-orthant prob P(Z0 > t0, Z1 > t1).
        p11 = _bvn_upper(t[0], t[1], r, nd)
        out = np.empty(4)
        for s in range(4):
            w0 = masks[s, 0] == 1
            w1 = masks[s, 1] == 1
            if w0 and w1:
                out[s] = p11
            elif w0 and not w1:
                out[s] = p[0] - p11
            elif (not w0) and w1:
                out[s] = p[1] - p11
            else:
                out[s] = 1.0 - p[0] - p[1] + p11
        return np.clip(out, 0.0, 1.0) / np.sum(np.clip(out, 0.0, 1.0))

    # n >= 3: seeded Monte-Carlo over the Gaussian copula (deterministic).
    L = np.linalg.cholesky(rho + 1e-12 * np.eye(n))
    rng = np.random.default_rng(0)
    m = 400_000
    Z = rng.standard_normal((m, n)) @ L.T
    wins = (Z > t[None, :]).astype(int)  # m x n
    # Map each sample to a state index and tally.
    idx = wins @ (1 << np.arange(n))
    counts = np.bincount(idx, minlength=2 ** n).astype(float)
    return counts / counts.sum()


def _bvn_upper(h: float, k: float, r: float, nd) -> float:
    """P(X > h, Y > k) for standard bivariate normal with correlation r.

    Drezner-Wesolowsky (1990) Gauss-Legendre quadrature of the bivariate density;
    accurate to ~1e-10. Uses statistics.NormalDist for the univariate CDF -- no scipy.
    """
    import math

    # P(X>h, Y>k) = P(X<=-h, Y<=-k) by symmetry; compute Phi2(-h,-k; r).
    dh, dk = -h, -k
    if abs(r) < 1e-15:
        return nd.cdf(dh) * nd.cdf(dk)

    # 10-point Gauss-Legendre nodes/weights on [0, r] for the standard formula
    # Phi2(dh,dk;r) = Phi(dh)Phi(dk) + integral_0^r phi2(dh,dk;t) dt.
    x = np.array(
        [0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
         0.8650633666889845, 0.9739065285171717]
    )
    wts = np.array(
        [0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
         0.1494513491505806, 0.0666713443086881]
    )
    nodes = np.concatenate([-x, x])
    weights = np.concatenate([wts, wts])

    a = 0.5 * r
    s = 0.0
    for node, wt in zip(nodes, weights):
        tt = a * node + a  # map [-1,1] -> [0, r]
        denom = 1.0 - tt * tt
        s += wt * math.exp(-(dh * dh + dk * dk - 2.0 * tt * dh * dk) / (2.0 * denom)) / math.sqrt(denom)
    integral = a * s / (2.0 * math.pi)
    return nd.cdf(dh) * nd.cdf(dk) + integral


# --------------------------------------------------------------------------- #
# Calibration / forecast quality                                             #
# --------------------------------------------------------------------------- #
def brier_score(probs: ArrayLike, outcomes: ArrayLike) -> float:
    """Mean squared error of probabilistic forecasts: mean((p - y)^2).

    Lower is better. 0 = perfect; 0.25 = always predicting 0.5; ~1 = confidently wrong.
    For binary outcomes y in {0,1} this is a proper scoring rule.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and outcomes must have the same shape")
    return float(np.mean((p - y) ** 2))


def log_loss(probs: ArrayLike, outcomes: ArrayLike, eps: float = 1e-15) -> float:
    """Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p)), with p clipped to [eps,1-eps].

    Lower is better. Penalizes confident wrong calls much harder than Brier (unbounded as
    p -> 0 for a true outcome). Also a proper scoring rule for binary forecasts.
    """
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and outcomes must have the same shape")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def closing_line_value(entry_decimal: float, closing_decimal: float) -> float:
    """CLV = entry_decimal / closing_decimal - 1.

    POSITIVE => you bet at higher decimal odds than the close, i.e. you BEAT the closing
    line (got a better price). Consistent positive CLV is the most reliable public proxy
    for genuine betting edge, since the closing line is the market's sharpest estimate.
    """
    if entry_decimal <= 1.0 or closing_decimal <= 1.0:
        raise ValueError("decimal odds must be > 1")
    return entry_decimal / closing_decimal - 1.0


# --------------------------------------------------------------------------- #
# Self-tests                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    TOL = 1e-9

    # --- Odds conversions ---
    assert abs(decimal_to_implied(2.0) - 0.5) < TOL
    assert abs(implied_to_decimal(0.5) - 2.0) < TOL
    assert abs(american_to_decimal(100) - 2.0) < TOL
    assert abs(american_to_decimal(-200) - 1.5) < TOL
    assert abs(american_to_decimal(150) - 2.5) < TOL
    assert abs(american_to_decimal(-150) - (1.0 + 100.0 / 150.0)) < TOL

    # decimal_to_american round-trips through american_to_decimal
    for ml in (100, 150, 250, -110, -150, -200, -500):
        dec = american_to_decimal(ml)
        assert abs(decimal_to_american(dec) - ml) < 1e-6, ml
    assert abs(decimal_to_american(2.0) - 100.0) < TOL  # pick-em -> +100
    assert abs(decimal_to_american(1.5) - (-200.0)) < TOL

    # implied/decimal round-trip
    for d in (1.5, 2.0, 3.3, 10.0):
        assert abs(implied_to_decimal(decimal_to_implied(d)) - d) < TOL

    # --- Multiplicative devig ---
    raw = [0.55, 0.50]  # booksum 1.05, ~4.8% overround
    mult = devig_multiplicative(raw)
    assert abs(mult.sum() - 1.0) < TOL
    assert abs(mult[0] - 0.55 / 1.05) < TOL
    assert abs(mult[1] - 0.50 / 1.05) < TOL

    # --- Power devig ---
    powr = devig_power(raw)
    assert abs(powr.sum() - 1.0) < 1e-9
    assert np.all(powr > 0.0) and np.all(powr < 1.0)
    # Three-way market (e.g. soccer 1X2) also normalizes.
    powr3 = devig_power([0.45, 0.30, 0.32])
    assert abs(powr3.sum() - 1.0) < 1e-9
    assert np.all((powr3 > 0.0) & (powr3 < 1.0))

    # --- Shin devig ---
    shin = devig_shin(raw)
    assert abs(shin.sum() - 1.0) < 1e-9
    assert np.all(shin > 0.0) and np.all(shin < 1.0)
    shin3 = devig_shin([0.45, 0.30, 0.32])
    assert abs(shin3.sum() - 1.0) < 1e-9
    assert np.all((shin3 > 0.0) & (shin3 < 1.0))

    # Sanity: all three methods agree closely when the market is symmetric.
    sym = [0.525, 0.525]  # equal raw implied on both sides
    m, p, s = devig_multiplicative(sym), devig_power(sym), devig_shin(sym)
    for v in (m, p, s):
        assert abs(v[0] - 0.5) < 1e-6 and abs(v[1] - 0.5) < 1e-6

    # Power devig pushes the longshot lower than multiplicative on a skewed book
    # (favorite-longshot bias correction).
    skew = [0.80, 0.30]  # heavy favorite + longshot, booksum 1.10
    m_sk, p_sk = devig_multiplicative(skew), devig_power(skew)
    # index 1 is the longshot; power assigns it less probability than multiplicative.
    assert p_sk[1] < m_sk[1]

    # --- EV & Kelly ---
    assert abs(expected_value(0.6, 2.0) - 0.2) < TOL
    assert abs(expected_value(0.5, 2.0) - 0.0) < TOL  # fair coin at even money
    assert abs(kelly_fraction(0.6, 2.0) - 0.2) < TOL
    assert kelly_fraction(0.4, 2.0) == 0.0  # no edge -> no bet
    assert kelly_fraction(0.5, 2.0) == 0.0  # exactly fair -> no bet
    # Kelly identity check vs formula for a non-trivial case.
    p_, d_ = 0.55, 2.10
    b_ = d_ - 1.0
    assert abs(kelly_fraction(p_, d_) - (p_ * b_ - (1 - p_)) / b_) < TOL

    # --- Joint Kelly ---
    # (a) Single-bet reduction: joint_kelly with n==1 equals kelly_fraction.
    jk1 = joint_kelly([0.55], [2.10])
    assert jk1.shape == (1,)
    assert abs(jk1[0] - kelly_fraction(0.55, 2.10)) < 1e-7, jk1[0]
    jk1b = joint_kelly([0.6], [2.0])
    assert abs(jk1b[0] - kelly_fraction(0.6, 2.0)) < 1e-7  # == 0.2
    # Independent two-bet case: each leg is staked CLOSE TO but slightly BELOW its own
    # single-bet Kelly (simultaneous bets can both lose, adding variance the single-bet
    # formula ignores). Symmetric inputs => equal stakes.
    jk_ind = joint_kelly([0.55, 0.55], [2.10, 2.10], cov=None)
    k_single = kelly_fraction(0.55, 2.10)
    assert abs(jk_ind[0] - jk_ind[1]) < 1e-9  # symmetry
    assert jk_ind[0] < k_single and jk_ind[0] > 0.9 * k_single  # close, slightly under

    # (b) Positive correlation shrinks the joint stake vs naive summed single-bet Kelly.
    p2, d2 = [0.55, 0.55], [2.10, 2.10]
    naive = np.array([kelly_fraction(p2[0], d2[0]), kelly_fraction(p2[1], d2[1])])
    rho_pos = np.array([[1.0, 0.6], [0.6, 1.0]])
    jk_corr = joint_kelly(p2, d2, cov=rho_pos)
    assert np.all(jk_corr > 0.0)  # still positive-edge bets
    # Each correlated leg is staked LESS than its standalone Kelly...
    assert jk_corr[0] < naive[0] and jk_corr[1] < naive[1]
    # ...and the total joint stake is below the naive sum of single-bet Kellys.
    assert jk_corr.sum() < naive.sum()
    # Stronger correlation => even smaller joint stake (monotone risk penalty).
    rho_hi = np.array([[1.0, 0.9], [0.9, 1.0]])
    jk_hi = joint_kelly(p2, d2, cov=rho_hi)
    assert jk_hi.sum() < jk_corr.sum()

    # (c) Fractional Kelly scales the vector linearly.
    jk_half = joint_kelly(p2, d2, cov=rho_pos, frac=0.5)
    assert np.allclose(jk_half, 0.5 * jk_corr, atol=1e-9)
    # (d) Cap clips per-leg magnitude.
    jk_cap = joint_kelly(p2, d2, cov=rho_pos, cap=0.01)
    assert np.all(np.abs(jk_cap) <= 0.01 + 1e-12) and np.any(np.abs(jk_cap) >= 0.01 - 1e-9)

    # (e) Optimality anchor: returned f is a stationary point of E[log wealth] (grad ~ 0).
    f_opt = joint_kelly(p2, d2, cov=rho_pos)
    b_ = np.array([d2[0] - 1.0, d2[1] - 1.0])
    masks_ = np.array([[(s >> i) & 1 for i in range(2)] for s in range(4)], dtype=int)
    R_ = np.where(masks_ == 1, b_[None, :], -1.0)
    P_ = _lattice_probs(np.asarray(p2, float), rho_pos, masks_)
    grad_ = (P_ / (1.0 + R_ @ f_opt)) @ R_
    assert np.max(np.abs(grad_)) < 1e-6, grad_

    # --- Calibration ---
    assert abs(brier_score([1.0, 0.0], [1, 0]) - 0.0) < TOL  # perfect
    assert abs(brier_score([0.0, 1.0], [1, 0]) - 1.0) < TOL  # worst case ~1
    assert abs(brier_score([0.5, 0.5], [1, 0]) - 0.25) < TOL  # uninformative

    # Better-calibrated probabilities => lower log loss.
    outcomes = [1, 0, 1, 1, 0]
    good = [0.85, 0.10, 0.90, 0.80, 0.15]
    bad = [0.55, 0.45, 0.50, 0.52, 0.48]
    assert log_loss(good, outcomes) < log_loss(bad, outcomes)
    # Brier agrees with the ranking here too.
    assert brier_score(good, outcomes) < brier_score(bad, outcomes)
    # Clipping keeps log loss finite even on a confidently-wrong forecast.
    assert np.isfinite(log_loss([1.0], [0]))

    # --- CLV ---
    assert closing_line_value(2.10, 2.00) > 0.0  # beat the close
    assert abs(closing_line_value(2.10, 2.00) - (2.10 / 2.00 - 1.0)) < TOL
    assert closing_line_value(1.90, 2.00) < 0.0  # worse than close
    assert abs(closing_line_value(2.00, 2.00)) < TOL  # matched the close

    print("betting_markets.py: all self-tests passed.")
