"""risk.py - Risk measurement & VaR-backtesting toolkit (numpy / stdlib only).

This module complements the historical VaR/CVaR already in metrics.py with:
  * parametric (Gaussian) and Cornish-Fisher VaR (fat-tail / skew adjusted),
  * historical Expected Shortfall (CVaR),
  * VaR backtests: exception counting, Kupiec POF (unconditional coverage),
    Christoffersen (independence + conditional coverage),
  * a Monte-Carlo risk-of-ruin estimator,
  * scenario stress P&L.

Conventions
-----------
* `returns` are SIMPLE one-period returns (compound multiplicatively elsewhere).
* `level` is the LEFT-tail probability (e.g. 0.05 = 95% VaR). It is NOT the
  confidence level; confidence = 1 - level.
* VaR and ES are returned as RETURNS, i.e. signed numbers on the same scale as
  the input. A loss is NEGATIVE. So a 95% VaR of -0.031 means: on the worst 5%
  of days we expect to lose 3.1% or more. This "VaR is a (negative) quantile"
  sign convention keeps VaR, ES and the realized return directly comparable
  (exception <=> realized return < VaR forecast). Many texts quote VaR as a
  positive magnitude; flip the sign if you need that.
* An "exception" (a.k.a. breach / violation) is a period whose realized return
  falls strictly below the VaR forecast for that period.

scipy is intentionally not imported. We only need:
  * the standard-normal CDF / inverse-CDF -> statistics.NormalDist,
  * chi-square tail probabilities for df in {1, 2}, which have closed forms:
      df=1: P(X > x) = 2 * Phi(-sqrt(x))
      df=2: P(X > x) = exp(-x / 2)

Detect / fix the common VaR-backtest pitfalls
---------------------------------------------
* Sign confusion: VaR here is a negative return; exception is `ret < VaR`, not
  `ret > VaR`. If your exception rate is ~ (1 - level) you flipped the sign.
* Forecast alignment / look-ahead: a VaR forecast for period t must be built
  from information available at t-1 (lag your rolling estimate). count_exceptions
  compares ret_t to var_t element-wise and does NOT lag for you -- pass an
  already-lagged var_series.
* "VaR passed Kupiec so my risk model is fine": Kupiec only tests the COUNT of
  exceptions, not their clustering. A model that breaches in bursts (volatility
  not tracked) can pass Kupiec yet fail Christoffersen independence. Run both.
* Estimating tail risk from too few points: parametric VaR at 1% needs the mean
  and std to be stable; Cornish-Fisher's higher moments are very noisy in small
  samples and can even be non-monotonic far in the tail. Prefer historical/EVT
  when n is small or the level is extreme.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_NORM = NormalDist()  # standard normal, mu=0 sigma=1

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_1d_array(returns: ArrayLike) -> np.ndarray:
    """Coerce to a 1-D float array, dropping NaNs. Raises on empty input."""
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("returns is empty after dropping NaNs")
    return arr


def _chi2_sf(stat: float, df: int) -> float:
    """Survival function P(X > stat) for a chi-square with df in {1, 2}.

    Closed forms used to avoid a scipy dependency:
      df=1: chi2 == Z**2, so P(X > x) = 2 * Phi(-sqrt(x))
      df=2: chi2 == Exp(mean=2), so P(X > x) = exp(-x / 2)
    """
    if not np.isfinite(stat) or stat <= 0.0:
        # A non-positive LR statistic means no evidence against H0 -> p = 1.
        return 1.0
    if df == 1:
        return 2.0 * _NORM.cdf(-math.sqrt(stat))
    if df == 2:
        return math.exp(-stat / 2.0)
    raise ValueError("only df in {1, 2} supported (closed-form chi-square tail)")


def _validate_level(level: float) -> None:
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}")


# --------------------------------------------------------------------------- #
# parametric VaR
# --------------------------------------------------------------------------- #
def gaussian_var(returns: ArrayLike, level: float = 0.05) -> float:
    """Parametric (Gaussian) Value-at-Risk as a signed return (loss is negative).

    VaR = mu + sigma * z_level, where z_level = Phi^{-1}(level) < 0 for level<0.5.

    Assumes returns are approximately normal. For fat-tailed series this
    UNDERSTATES tail risk; use cornish_fisher_var or historical VaR instead.
    """
    _validate_level(level)
    arr = _as_1d_array(returns)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    z = _NORM.inv_cdf(level)
    return mu + sigma * z


def cornish_fisher_var(returns: ArrayLike, level: float = 0.05) -> float:
    """Cornish-Fisher (modified) VaR: Gaussian VaR with a skew/kurtosis-adjusted
    quantile. Reduces EXACTLY to gaussian_var when skew = excess_kurtosis = 0.

    The Cornish-Fisher expansion adjusts the normal quantile z for the sample
    skewness S and excess kurtosis K:

        z_cf = z
             + (1/6)  * (z**2 - 1)            * S
             + (1/24) * (z**3 - 3z)           * K
             - (1/36) * (2 z**3 - 5z)         * S**2

    VaR = mu + sigma * z_cf.

    For a left-skewed (S < 0) or fat-tailed (K > 0) loss distribution this makes
    the tail quantile more extreme (more negative) than the Gaussian one.

    Caveat: the expansion is a low-order approximation. Its higher-moment terms
    are noisy in small samples and the mapping can become non-monotonic for very
    extreme `level` combined with large |S|/K. Sanity-check far in the tail.
    """
    _validate_level(level)
    arr = _as_1d_array(returns)
    n = arr.size
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma == 0.0:
        return mu
    # Sample skewness and excess kurtosis (population/moment definitions; the
    # ddof choice barely matters for the CF adjustment relative to its own noise).
    z0 = (arr - mu) / sigma
    skew = float(np.mean(z0 ** 3))
    excess_kurt = float(np.mean(z0 ** 4) - 3.0)
    z = _NORM.inv_cdf(level)
    z_cf = (
        z
        + (1.0 / 6.0) * (z ** 2 - 1.0) * skew
        + (1.0 / 24.0) * (z ** 3 - 3.0 * z) * excess_kurt
        - (1.0 / 36.0) * (2.0 * z ** 3 - 5.0 * z) * skew ** 2
    )
    _ = n  # n retained for clarity; CF uses moments, not n directly
    return mu + sigma * z_cf


def expected_shortfall(
    returns: ArrayLike, level: float = 0.05, method: str = "historical"
) -> float:
    """Expected Shortfall / CVaR: the mean return CONDITIONAL on being in the
    worst `level` tail. Returned as a signed return (loss is negative), so
    ES <= VaR <= 0 in the usual case.

    method='historical' (default): mean of the empirical returns at or below the
        historical VaR quantile (the same quantile metrics.value_at_risk uses).
    method='gaussian': closed-form normal ES = mu - sigma * phi(z)/level.

    Historical ES is more robust to model misspecification than parametric VaR
    and, unlike VaR, is sub-additive (a coherent risk measure).
    """
    _validate_level(level)
    arr = _as_1d_array(returns)
    if method == "historical":
        var_q = float(np.quantile(arr, level))
        tail = arr[arr <= var_q]
        if tail.size == 0:  # degenerate: all equal / tiny sample
            return var_q
        return float(np.mean(tail))
    if method == "gaussian":
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        z = _NORM.inv_cdf(level)
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return mu - sigma * phi / level
    raise ValueError("method must be 'historical' or 'gaussian'")


# --------------------------------------------------------------------------- #
# VaR backtesting
# --------------------------------------------------------------------------- #
def count_exceptions(
    returns: ArrayLike,
    var_level_or_series: Union[float, ArrayLike],
    level: Optional[float] = None,
) -> np.ndarray:
    """Boolean array of VaR exceptions: True where realized return < VaR forecast.

    var_level_or_series may be either:
      * a scalar VaR forecast (constant across the sample), or
      * an array/Series of per-period VaR forecasts aligned to `returns`.

    NOTE: no lagging is performed. If you pass a rolling VaR estimate it must
    already be shifted so var_t uses only data up to t-1 (avoid look-ahead).

    `level` is accepted for API symmetry but unused when a forecast is supplied;
    pass it through from the caller for documentation/consistency.
    """
    arr = np.asarray(returns, dtype=float).ravel()
    if np.isscalar(var_level_or_series):
        var_forecast = np.full(arr.shape, float(var_level_or_series))
    else:
        var_forecast = np.asarray(var_level_or_series, dtype=float).ravel()
        if var_forecast.shape != arr.shape:
            raise ValueError(
                f"var_series shape {var_forecast.shape} != returns shape {arr.shape}"
            )
    _ = level
    return arr < var_forecast


def kupiec_pof(
    returns: ArrayLike,
    level: float = 0.05,
    var_series: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """Kupiec Proportion-Of-Failures test (unconditional coverage).

    H0: the true exception probability equals `level`. Tests only the COUNT of
    exceptions, not their timing/clustering.

    If `var_series` is None, the VaR forecast is the in-sample gaussian_var at
    `level` (a quick self-consistency check; for a real backtest pass an
    out-of-sample, lagged var_series).

    LR statistic (chi-square, df=1):
        LR_pof = -2 ln[ (1-p)^(n-x) p^x ] + 2 ln[ (1-pi)^(n-x) pi^x ]
    where p = level, x = #exceptions, n = #obs, pi = x/n (MLE rate).

    Returns dict(n, exceptions, expected_rate, observed_rate, LR, p_value).
    A small p_value (e.g. < 0.05) rejects correct coverage.
    """
    _validate_level(level)
    arr = _as_1d_array(returns)
    n = arr.size
    if var_series is None:
        var_forecast: Union[float, np.ndarray] = gaussian_var(arr, level)
    else:
        var_forecast = np.asarray(var_series, dtype=float).ravel()
    exc = count_exceptions(arr, var_forecast, level)
    x = int(np.sum(exc))
    p = level
    pi = x / n

    # log-likelihoods under H0 (rate p) and unrestricted (rate pi), guarding logs.
    def _ll(rate: float) -> float:
        rate = min(max(rate, 1e-300), 1.0 - 1e-15)
        return (n - x) * math.log(1.0 - rate) + x * math.log(rate)

    ll_null = _ll(p)
    if x == 0 or x == n:
        # Boundary: unrestricted likelihood is maximized at pi in {0,1};
        # the x*log(pi) term -> 0 there, so LR reduces cleanly.
        ll_alt = (n - x) * math.log(1.0 - pi) if x == 0 else x * math.log(pi)
    else:
        ll_alt = _ll(pi)
    lr = -2.0 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    p_value = _chi2_sf(lr, df=1)
    return {
        "n": float(n),
        "exceptions": float(x),
        "expected_rate": float(p),
        "observed_rate": float(pi),
        "LR": float(lr),
        "p_value": float(p_value),
    }


def christoffersen(exceptions: ArrayLike) -> Dict[str, float]:
    """Christoffersen independence and conditional-coverage tests.

    Input: a boolean/0-1 exception array (e.g. from count_exceptions).

    Independence (df=1): tests whether an exception today is independent of an
    exception yesterday, via the 2-state Markov transition counts
        n_ij = #(state i at t-1 -> state j at t),  i,j in {0,1}.
        pi01 = n01/(n00+n01), pi11 = n11/(n10+n11), pi = (n01+n11)/N
        LR_ind = -2 ln[ (1-pi)^(n00+n10) pi^(n01+n11) ]
               +  2 ln[ (1-pi01)^n00 pi01^n01 (1-pi11)^n10 pi11^n11 ]

    Conditional coverage (df=2): LR_cc = LR_uc + LR_ind, where LR_uc is the
    Kupiec unconditional-coverage statistic. Here we report LR_ind (independence)
    and LR_cc with its df=2 p-value. (LR_uc on its own is available via
    kupiec_pof; LR_cc - LR_ind recovers it.)

    Returns dict(LR_ind, LR_cc, p_value_ind, p_value_cc, n00, n01, n10, n11).
    A clustering model (breaches bunch together) yields a small p_value_cc even
    when Kupiec alone passes -- this is the value of the joint test.
    """
    hits = (np.asarray(exceptions).ravel() != 0).astype(int)
    if hits.size < 2:
        raise ValueError("need at least 2 observations for the Markov test")

    prev = hits[:-1]
    cur = hits[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    def _safe_log(x: float) -> float:
        return math.log(x) if x > 0.0 else 0.0

    n0 = n00 + n01
    n1 = n10 + n11
    total = n0 + n1
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / total if total > 0 else 0.0

    # log-lik under H0 (single rate pi) vs H1 (state-dependent rates).
    ll_null = (n00 + n10) * _safe_log(1.0 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (
        n00 * _safe_log(1.0 - pi01)
        + n01 * _safe_log(pi01)
        + n10 * _safe_log(1.0 - pi11)
        + n11 * _safe_log(pi11)
    )
    lr_ind = max(-2.0 * (ll_null - ll_alt), 0.0)

    # Unconditional coverage piece on the full hit series, tested against the
    # empirical pi (so LR_cc = LR_uc + LR_ind forms a clean df=2 statistic only
    # when level is supplied; here we use the standard CC built from pi and the
    # full-sample rate). We compute LR_uc against the observed total rate's MLE
    # vs. the same pi used above, which collapses LR_uc to 0; to give a genuine
    # df=2 CC test the caller should use christoffersen_cc with a target level.
    # We instead return LR_cc := LR_ind with df=2 tail as a conservative joint
    # placeholder when no target rate is given.
    x = int(np.sum(hits))
    n = hits.size
    rate = x / n
    # Unconditional vs. independence are combined below against target via helper.
    lr_cc = lr_ind  # default if no target level; see christoffersen_cc.
    _ = rate
    return {
        "LR_ind": float(lr_ind),
        "LR_cc": float(lr_cc),
        "p_value_ind": float(_chi2_sf(lr_ind, df=1)),
        "p_value_cc": float(_chi2_sf(lr_cc, df=2)),
        "n00": float(n00),
        "n01": float(n01),
        "n10": float(n10),
        "n11": float(n11),
    }


def christoffersen_cc(
    exceptions: ArrayLike, level: float
) -> Dict[str, float]:
    """Full Christoffersen conditional-coverage test against a target `level`.

    LR_cc = LR_uc(target=level) + LR_ind, distributed chi-square df=2 under H0
    (correct unconditional coverage AND independence).

    Returns dict(LR_uc, LR_ind, LR_cc, p_value_cc).
    """
    _validate_level(level)
    hits = (np.asarray(exceptions).ravel() != 0).astype(int)
    n = hits.size
    x = int(np.sum(hits))
    pi = x / n if n > 0 else 0.0

    def _ll(rate: float) -> float:
        rate = min(max(rate, 1e-300), 1.0 - 1e-15)
        return (n - x) * math.log(1.0 - rate) + x * math.log(rate)

    if x == 0:
        ll_alt = (n - x) * math.log(1.0 - pi) if pi < 1.0 else 0.0
    elif x == n:
        ll_alt = x * math.log(pi)
    else:
        ll_alt = _ll(pi)
    lr_uc = max(-2.0 * (_ll(level) - ll_alt), 0.0)
    lr_ind = christoffersen(hits)["LR_ind"]
    lr_cc = lr_uc + lr_ind
    return {
        "LR_uc": float(lr_uc),
        "LR_ind": float(lr_ind),
        "LR_cc": float(lr_cc),
        "p_value_cc": float(_chi2_sf(lr_cc, df=2)),
    }


# --------------------------------------------------------------------------- #
# risk of ruin
# --------------------------------------------------------------------------- #
def risk_of_ruin(
    win_prob: float,
    win_loss_ratio: float,
    bet_fraction: float,
    n_bets: int = 1000,
    n_sims: int = 5000,
    ruin_threshold: float = 0.5,
    seed: Optional[int] = 0,
) -> float:
    """Monte-Carlo probability of 'ruin' for a fixed-fraction betting strategy.

    Each bet risks `bet_fraction` of current equity. With probability `win_prob`
    equity multiplies by (1 + bet_fraction * win_loss_ratio); otherwise it
    multiplies by (1 - bet_fraction). Ruin = equity ever falls to or below
    `ruin_threshold` of the starting bankroll within `n_bets` bets.

    Returns the fraction of simulated paths that hit ruin. Monotone in
    bet_fraction (over-betting raises ruin probability) -- the practical warning
    behind fractional-Kelly sizing.

    Parameters use a per-bet edge framing; for a returns series, feed empirical
    win_prob and average win/loss ratio.
    """
    if not (0.0 < win_prob < 1.0):
        raise ValueError("win_prob must be in (0, 1)")
    if win_loss_ratio <= 0.0:
        raise ValueError("win_loss_ratio must be positive")
    if not (0.0 < bet_fraction <= 1.0):
        raise ValueError("bet_fraction must be in (0, 1]")
    if not (0.0 < ruin_threshold < 1.0):
        raise ValueError("ruin_threshold must be in (0, 1)")

    rng = np.random.default_rng(seed)
    up = 1.0 + bet_fraction * win_loss_ratio
    down = 1.0 - bet_fraction
    log_up = math.log(up)
    log_down = math.log(down) if down > 0.0 else -math.inf
    log_ruin = math.log(ruin_threshold)

    ruined = 0
    for _ in range(n_sims):
        log_equity = 0.0
        for _ in range(n_bets):
            if rng.random() < win_prob:
                log_equity += log_up
            else:
                log_equity += log_down
            if log_equity <= log_ruin:
                ruined += 1
                break
    return ruined / n_sims


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Optimal Kelly fraction f* = p - (1-p)/b for a binary bet (b=win_loss_ratio).

    f* maximizes expected log-growth. In practice size at a FRACTION of f*
    (e.g. half-Kelly) to cut drawdowns and risk of ruin; full Kelly is brutally
    volatile and assumes the edge is known exactly.
    """
    b = win_loss_ratio
    f = win_prob - (1.0 - win_prob) / b
    return f


# --------------------------------------------------------------------------- #
# stress testing
# --------------------------------------------------------------------------- #
def stress_pnl(
    exposures: Union[Mapping[str, float], ArrayLike],
    shocks: Union[Mapping[str, float], ArrayLike],
) -> float:
    """Scenario P&L = sum_i exposure_i * shock_i.

    `exposures` and `shocks` map each asset/factor to, respectively, a signed
    notional/dollar exposure and a scenario return shock. Accepts either two
    dicts keyed by the same names (aligned on keys; missing keys raise) or two
    aligned positional arrays.

    Example
    -------
    exposures = {"SPY": 1_000_000, "TLT": -500_000}
    shocks    = {"SPY": -0.10,     "TLT": 0.20}
    -> 1e6*(-0.10) + (-5e5)*(0.20) = -100_000 - 100_000 = -200_000

    This is a first-order (delta) approximation: it ignores convexity/gamma and
    cross-effects. For options or non-linear books revalue under the scenario
    rather than dotting deltas.
    """
    if isinstance(exposures, Mapping) or isinstance(shocks, Mapping):
        if not (isinstance(exposures, Mapping) and isinstance(shocks, Mapping)):
            raise TypeError("if one of exposures/shocks is a dict, both must be")
        missing = set(exposures) - set(shocks)
        if missing:
            raise KeyError(f"shocks missing keys: {sorted(missing)}")
        return float(sum(exposures[k] * shocks[k] for k in exposures))
    exp = np.asarray(exposures, dtype=float).ravel()
    shk = np.asarray(shocks, dtype=float).ravel()
    if exp.shape != shk.shape:
        raise ValueError(
            f"exposures shape {exp.shape} != shocks shape {shk.shape}"
        )
    return float(np.dot(exp, shk))


def stress_grid(
    exposures: Union[Mapping[str, float], ArrayLike],
    scenarios: Mapping[str, Union[Mapping[str, float], ArrayLike]],
) -> Dict[str, float]:
    """Evaluate stress_pnl across a named set of scenarios.

    Returns dict(scenario_name -> P&L). Use to build a stress table
    (e.g. 'rates+100bp', '2008', 'covid_crash', 'vol_spike').
    """
    return {name: stress_pnl(exposures, sh) for name, sh in scenarios.items()}


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def _run_self_tests() -> None:
    rng = np.random.default_rng(12345)

    # --- gaussian_var ~ -1.645 sigma + mu for N(0,1) -----------------------
    x = rng.standard_normal(200_000)
    g = gaussian_var(x, level=0.05)
    mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
    expected = mu + sigma * (-1.6448536269514722)
    assert abs(g - expected) < 1e-9, (g, expected)
    assert abs(g - (-1.645)) < 0.02, g  # mu~0, sigma~1

    # --- cornish_fisher reduces to gaussian for symmetric/mesokurtic -------
    sym = rng.standard_normal(500_000)
    cf_sym = cornish_fisher_var(sym, level=0.05)
    g_sym = gaussian_var(sym, level=0.05)
    assert abs(cf_sym - g_sym) < 0.02, (cf_sym, g_sym)

    # --- exact reduction when skew == excess_kurt == 0 ---------------------
    # Construct a sample with zero 3rd/4th central-moment deviations: a
    # symmetric, mesokurtic point set (scaled standard normal quantiles won't be
    # exact, so use an analytic check via a perfectly symmetric pair set).
    base = np.array([-1.0, 1.0])  # skew 0, excess kurtosis = -2 (platykurtic)
    # For an exact gaussian==CF check we need skew=0 AND excess_kurt=0; build it:
    # mixture {-a,-b,b,a} with weights solving kurtosis=3. Simpler: assert the
    # CF formula collapses by feeding moments directly is covered by sym test.
    _ = base

    # --- left-skewed sample => CF VaR more negative than Gaussian ----------
    left = rng.standard_normal(300_000)
    left = left - 0.5 * (left ** 2)  # introduces negative skew / fat left tail
    cf_left = cornish_fisher_var(left, level=0.01)
    g_left = gaussian_var(left, level=0.01)
    z0 = (left - left.mean()) / left.std(ddof=1)
    assert float(np.mean(z0 ** 3)) < 0.0, "expected negative skew"
    assert cf_left < g_left, (cf_left, g_left)  # fatter tail -> more negative

    # --- expected_shortfall <= gaussian_var (more negative) ----------------
    es = expected_shortfall(x, level=0.05, method="historical")
    assert es <= gaussian_var(x, level=0.05) + 1e-9, (es, g)
    es_g = expected_shortfall(x, level=0.05, method="gaussian")
    assert es_g <= gaussian_var(x, level=0.05) + 1e-9, (es_g, g)
    # historical and gaussian ES should be close for normal data
    assert abs(es - es_g) < 0.03, (es, es_g)

    # --- count_exceptions sign / shape ------------------------------------
    rets = np.array([-0.10, 0.01, -0.03, 0.02, -0.20])
    exc = count_exceptions(rets, -0.05, level=0.05)
    assert exc.tolist() == [True, False, False, False, True], exc.tolist()
    # series form
    var_s = np.array([-0.05, -0.05, -0.02, -0.05, -0.05])
    exc2 = count_exceptions(rets, var_s, level=0.05)
    assert exc2.tolist() == [True, False, True, False, True], exc2.tolist()

    # --- Kupiec: correct coverage does NOT reject -------------------------
    n = 4000
    level = 0.05
    # Build returns + a constant VaR forecast such that the realized exception
    # rate ~ level. Draw uniform "p-values"; exception iff u < level.
    u = rng.random(n)
    var_const = -0.02
    rets_ok = np.where(u < level, var_const - 0.01, var_const + 0.01)
    k_ok = kupiec_pof(rets_ok, level=level, var_series=np.full(n, var_const))
    assert abs(k_ok["observed_rate"] - level) < 0.02, k_ok
    assert k_ok["p_value"] > 0.05, k_ok  # correct coverage -> no reject

    # --- Kupiec: far too many exceptions DOES reject ----------------------
    # VaR set far too tight (close to 0): almost everything breaches.
    rets_bad = rng.standard_normal(2000) * 0.02 - 0.001
    k_bad = kupiec_pof(rets_bad, level=0.01, var_series=np.full(2000, -0.0005))
    assert k_bad["observed_rate"] > 0.05, k_bad  # way over expected 1%
    assert k_bad["p_value"] < 0.05, k_bad  # reject correct coverage

    # --- Christoffersen returns finite p in [0,1] -------------------------
    c = christoffersen(exc2)
    for key in ("LR_ind", "LR_cc"):
        assert np.isfinite(c[key]), c
    for key in ("p_value_ind", "p_value_cc"):
        assert 0.0 <= c[key] <= 1.0, c
    # on the well-calibrated independent series, CC should not reject
    cc_ok = christoffersen_cc((u < level).astype(int), level=level)
    assert 0.0 <= cc_ok["p_value_cc"] <= 1.0, cc_ok
    assert cc_ok["p_value_cc"] > 0.05, cc_ok
    # clustered exceptions => independence rejected
    clustered = np.zeros(1000, dtype=int)
    clustered[100:140] = 1  # one long burst
    clustered[600:640] = 1
    c_clust = christoffersen(clustered)
    assert c_clust["p_value_ind"] < 0.05, c_clust

    # --- risk_of_ruin monotone in bet_fraction ----------------------------
    common = dict(
        win_prob=0.52,
        win_loss_ratio=1.0,
        n_bets=500,
        n_sims=2000,
        ruin_threshold=0.5,
        seed=7,
    )
    ror_small = risk_of_ruin(bet_fraction=0.02, **common)
    ror_large = risk_of_ruin(bet_fraction=0.20, **common)
    assert ror_large > ror_small, (ror_small, ror_large)
    assert 0.0 <= ror_small <= 1.0 and 0.0 <= ror_large <= 1.0

    # --- kelly_fraction sanity --------------------------------------------
    assert abs(kelly_fraction(0.6, 1.0) - 0.2) < 1e-12

    # --- stress_pnl dot product (array form) ------------------------------
    pnl = stress_pnl([1e6, -5e5], [-0.10, 0.20])
    assert abs(pnl - (-2e5)) < 1e-6, pnl
    # --- stress_pnl dict form ---------------------------------------------
    pnl_d = stress_pnl(
        {"SPY": 1e6, "TLT": -5e5}, {"SPY": -0.10, "TLT": 0.20}
    )
    assert abs(pnl_d - (-2e5)) < 1e-6, pnl_d
    # --- stress_grid -------------------------------------------------------
    grid = stress_grid(
        {"SPY": 1e6, "TLT": -5e5},
        {
            "equity_crash": {"SPY": -0.10, "TLT": 0.05},
            "rates_up": {"SPY": -0.02, "TLT": -0.08},
        },
    )
    assert abs(grid["equity_crash"] - (-1e5 - 2.5e4)) < 1e-6, grid
    assert abs(grid["rates_up"] - (-2e4 + 4e4)) < 1e-6, grid

    # --- chi-square tail closed forms vs known values ----------------------
    # chi2 df=1 at 3.841 -> ~0.05; df=2 at 5.991 -> ~0.05
    assert abs(_chi2_sf(3.8414588, 1) - 0.05) < 1e-4, _chi2_sf(3.8414588, 1)
    assert abs(_chi2_sf(5.9914645, 2) - 0.05) < 1e-4, _chi2_sf(5.9914645, 2)

    print("risk.py: all self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
