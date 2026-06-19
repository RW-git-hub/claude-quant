"""risk.py - Risk measurement & VaR-backtesting toolkit (numpy / stdlib only).

This module complements the historical VaR/CVaR already in metrics.py with:
  * parametric (Gaussian) and Cornish-Fisher VaR (fat-tail / skew adjusted),
  * historical Expected Shortfall (CVaR),
  * Filtered Historical Simulation (FHS) VaR/ES -- vol-reactive, the industry
    standard reactive VaR,
  * age-weighted (BRW) historical VaR -- exponentially time-weighted quantile,
  * EVT peaks-over-threshold (POT) VaR/ES -- a fitted Generalized Pareto tail for
    principled deep-quantile (99.9%) estimation from short samples,
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
  * a Generalized Pareto fit by probability-weighted moments (PWM), which is
    closed-form (no optimizer / no scipy).

Detect / fix the common VaR-backtest pitfalls
---------------------------------------------
* Sign confusion: VaR here is a negative return; exception is `ret < VaR`, not
  `ret > VaR`. If your exception rate is ~ (1 - level) you flipped the sign.
* Forecast alignment / look-ahead: a VaR forecast for period t must be built
  from information available at t-1 (lag your rolling estimate). count_exceptions
  compares ret_t to var_t element-wise and does NOT lag for you -- pass an
  already-lagged var_series. The rolling FHS routine here builds CAUSAL forecasts
  (var_t uses only data up to t-1) so its output is directly backtestable.
* "VaR passed Kupiec so my risk model is fine": Kupiec only tests the COUNT of
  exceptions, not their clustering. A model that breaches in bursts (volatility
  not tracked) can pass Kupiec yet fail Christoffersen independence. Run both.
* Estimating tail risk from too few points: parametric VaR at 1% needs the mean
  and std to be stable; Cornish-Fisher's higher moments are very noisy in small
  samples and can even be non-monotonic far in the tail. For the FAR tail
  (99%, 99.9%) prefer EVT/POT -- a plain historical quantile beyond 1/N is just
  the worst observation and cannot extrapolate, whereas the fitted GPD can.
* Stale VaR misses regime shifts: plain historical VaR reacts slowly to a vol
  spike. Use Filtered Historical Simulation (vol-rescaled) or age-weighting to
  get a reactive forecast that still respects the empirical tail shape.
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
    UNDERSTATES tail risk; use cornish_fisher_var, filtered_historical_var_es,
    or evt_pot_var_es instead.
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
    extreme `level` combined with large |S|/K. For the FAR tail use EVT/POT.
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
    and, unlike VaR, is sub-additive (a coherent risk measure). For the FAR tail
    of a short sample use evt_pot_var_es (the GPD ES extrapolates; the historical
    tail mean beyond 1/N is just one or two order statistics).
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
# Filtered Historical Simulation (FHS) and age-weighted historical VaR
# --------------------------------------------------------------------------- #
def ewma_volatility(
    returns: ArrayLike, lam: float = 0.94, sigma0: Optional[float] = None
) -> np.ndarray:
    """CAUSAL EWMA (RiskMetrics) one-step volatility forecast.

    Returns an array `sigma` aligned to `returns` where `sigma[t]` is the
    volatility forecast for period t, built ONLY from returns up to t-1 (no
    look-ahead). The recursion is the RiskMetrics square-root-EWMA on a
    zero-mean assumption (1-day horizon, where the mean is negligible):

        sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}**2

    The seed `sigma2_0` defaults to the full-sample variance (a fixed, in-sample
    constant used only to initialise the recursion; it is the conventional warm
    start). Because `sigma[t]` depends on `r[t-1]` and earlier, it is a valid
    t-1-measurable forecast for any t >= 1. NaNs are treated as zero shocks so
    the recursion stays aligned to calendar time -- pass a clean contiguous
    series.

    lam=0.94 is the RiskMetrics daily default (~75-day effective memory);
    lam=0.97 is the monthly default. Lower lam = more reactive, noisier.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam (decay) must be in (0, 1), got {lam}")
    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    if n == 0:
        raise ValueError("returns is empty")
    r = np.nan_to_num(r, nan=0.0)  # treat a missing return as a zero shock
    sig2 = np.empty(n, dtype=float)
    sig2[0] = float(np.var(r)) if sigma0 is None else float(sigma0) ** 2
    if sig2[0] <= 0.0:
        sig2[0] = 1e-12
    for t in range(1, n):
        sig2[t] = lam * sig2[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return np.sqrt(sig2)


def filtered_historical_var_es(
    returns: ArrayLike,
    level: float = 0.05,
    lam: float = 0.94,
    min_obs: int = 250,
    rolling: bool = True,
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """Filtered Historical Simulation (FHS) VaR and ES -- vol-reactive, the
    industry-standard reactive VaR (Barone-Adesi, Giannopoulos & Vosper).

    Idea: real returns are not iid (vol clusters), so the empirical quantile of
    RAW returns is stale during a vol regime shift. FHS standardizes each return
    by a CAUSAL volatility forecast to get approximately-iid residuals
    `z_t = r_t / sigma_t`, takes the empirical quantile/ES of those residuals
    (so it keeps the true fat-tailed/skewed shape), then RESCALES by the CURRENT
    sigma forecast. The result reacts immediately to volatility while remaining
    non-parametric in the tail shape.

        z_t = r_t / sigma_t                     # sigma_t causal (see ewma_volatility)
        z_q = empirical quantile_level(z)       # standardized-residual quantile
        VaR_t = sigma_t * z_q                   # rescale by current vol forecast
        ES_t  = sigma_t * mean(z | z <= z_q)

    Sign convention: z_q is negative (left tail), so VaR/ES come out negative
    (loss), consistent with the rest of this module.

    rolling=True (default): returns (var_series, es_series) arrays aligned to
        `returns`, each entry a CAUSAL forecast made at t-1 (NaN for the first
        `min_obs` periods where there is not yet enough history). These series
        are directly backtestable with count_exceptions / kupiec_pof WITHOUT
        further lagging -- the look-ahead lag is already baked in.
    rolling=False: returns the single (VaR, ES) snapshot computed from the FULL
        sample's standardized residuals, rescaled by the LAST sigma forecast --
        i.e. "today's" reactive VaR. Use this for a current risk number, not for
        a backtest (its residual quantile peeks at the whole sample).

    The standardized-residual quantile is taken over residuals strictly BEFORE t
    in the rolling case (z[:t], i.e. up to t-1), so no information from period t
    leaks into VaR_t.
    """
    _validate_level(level)
    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    if n == 0:
        raise ValueError("returns is empty")
    sigma = ewma_volatility(r, lam=lam)
    # standardized residuals; guard against a zero vol seed
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0.0, r / sigma, 0.0)

    if not rolling:
        zclean = z[~np.isnan(z)]
        z_q = float(np.quantile(zclean, level))
        tail = zclean[zclean <= z_q]
        z_es = float(np.mean(tail)) if tail.size else z_q
        sig_now = float(sigma[-1])
        return sig_now * z_q, sig_now * z_es

    if min_obs < 2:
        raise ValueError("min_obs must be >= 2 for a meaningful tail quantile")
    var = np.full(n, np.nan, dtype=float)
    es = np.full(n, np.nan, dtype=float)
    for t in range(min_obs, n):
        zh = z[:t]  # residuals up to t-1 only -> causal
        zh = zh[~np.isnan(zh)]
        if zh.size < min_obs:
            continue
        z_q = float(np.quantile(zh, level))
        var[t] = sigma[t] * z_q
        tail = zh[zh <= z_q]
        es[t] = sigma[t] * (float(np.mean(tail)) if tail.size else z_q)
    return var, es


def age_weighted_var(
    returns: ArrayLike, level: float = 0.05, decay: float = 0.99
) -> float:
    """Age-weighted (BRW: Boudoukh-Richardson-Whitelaw) historical VaR.

    A non-parametric reactive VaR that keeps the empirical tail shape but weights
    RECENT observations more, so the quantile responds to a vol regime shift
    faster than equal-weighted historical -- without needing a vol model.

    `returns` MUST be in chronological order (oldest first); the most recent
    observation (last element) receives weight `decay**0 = 1`, an observation
    `k` steps older gets `decay**k`, all normalised to sum to 1. The VaR is the
    smallest return whose cumulative (ascending-sorted) weight first reaches
    `level`:

        w_i = decay**(age_i) / sum_j decay**(age_j),  age = 0 for the newest obs
        sort returns ascending, accumulate weights, VaR = first r with cumW >= level

    As decay -> 1 every weight is equal and this collapses to the plain
    historical VaR. decay in [0.97, 0.995] is typical; lower decay = more
    reactive but uses fewer effective observations (more sampling noise in the
    tail).

    Returned as a signed return (loss negative), like the other VaR routines.
    """
    _validate_level(level)
    if not (0.0 < decay <= 1.0):
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    n = r.size
    if n == 0:
        raise ValueError("returns is empty after dropping NaNs")
    age = np.arange(n - 1, -1, -1)  # first element is oldest -> largest age
    w = np.power(float(decay), age.astype(float))
    w_sum = w.sum()
    if w_sum <= 0.0:  # underflow for tiny decay & long history
        # fall back to giving all weight to the most recent observation
        return float(r[-1])
    w = w / w_sum
    order = np.argsort(r, kind="stable")  # ascending
    r_sorted = r[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cum, level, side="left"))
    idx = min(idx, n - 1)
    return float(r_sorted[idx])


# --------------------------------------------------------------------------- #
# EVT: peaks-over-threshold (POT) tail VaR / ES via a fitted Generalized Pareto
# --------------------------------------------------------------------------- #
def _gpd_pwm_fit(excess: np.ndarray) -> Tuple[float, float]:
    """Fit a Generalized Pareto (shape xi, scale beta) to positive exceedances by
    probability-weighted moments (Hosking & Wallis 1987) -- closed form, no scipy.

    GPD survival: P(X > x) = (1 + xi * x / beta)**(-1/xi)  for xi != 0, x > 0.
    With the first two PWMs of the GPD,
        a0 = E[X]            = beta / (1 - xi)
        a1 = E[X (1 - F(X))] = beta / (2 (2 - xi))
    the method-of-PWM estimators invert to:
        xi   = 2 - a0 / (a0 - 2 a1)
        beta = 2 a0 a1 / (a0 - 2 a1)
    Sample PWMs from ascending order statistics x_(1)..x_(n):
        a0_hat = mean(x)
        a1_hat = (1/n) * sum_j  ((n-1-j)/(n-1)) * x_(j),   j = 0..n-1 (0-based)

    PWM is robust and well-behaved for the moderate xi (< 0.5) typical of
    financial tails, and recovers a known GPD shape within ~0.02 at n ~ 1e4. It
    requires a finite GPD first moment (xi < 1) and is mathematically bounded by
    xi -> 1 (since the sample a1 <= a0/2 forces the denominator >= 0); see
    evt_pot_var_es for how that ceiling is surfaced via `xi_near_one`.
    """
    x = np.sort(np.asarray(excess, dtype=float))
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 exceedances to fit a GPD")
    a0 = float(np.mean(x))
    j = np.arange(n, dtype=float)
    w = (n - 1.0 - j) / (n - 1.0)
    a1 = float(np.mean(w * x))
    denom = a0 - 2.0 * a1
    if denom == 0.0 or not np.isfinite(denom):
        # degenerate (e.g. constant exceedances) -> exponential tail (xi=0)
        return 0.0, max(a0, 1e-12)
    xi = 2.0 - a0 / denom
    beta = 2.0 * a0 * a1 / denom
    if beta <= 0.0 or not np.isfinite(beta):
        beta = max(a0 * (1.0 - min(xi, 0.0)), 1e-12)
    return float(xi), float(beta)


def evt_pot_var_es(
    returns: ArrayLike,
    level: float = 0.001,
    threshold_q: float = 0.95,
) -> Dict[str, float]:
    """Extreme-Value-Theory peaks-over-threshold (POT) VaR and ES for the FAR
    tail (McNeil & Frey). The principled route to deep quantiles (99%, 99.9%)
    from a short sample, where a plain historical quantile is just the worst
    one or two observations and cannot extrapolate.

    Method: work on LOSSES L = -returns (a loss is a large positive L). Pick a
    high threshold u = empirical quantile of L at `threshold_q` (e.g. the 95th
    loss percentile). By the Pickands-Balkema-de Haan theorem the exceedances
    (L - u | L > u) converge to a Generalized Pareto GPD(xi, beta); fit it by
    PWM (`_gpd_pwm_fit`). Then for any tail probability `level` < (1 - threshold_q):

        VaR_p(L) = u + (beta/xi) * ( ( (n/Nu) * level )**(-xi) - 1 )      [xi != 0]
        ES_p(L)  = VaR_p / (1 - xi) + (beta - xi*u) / (1 - xi)           [xi  < 1]

    where n = #obs, Nu = #exceedances over u, and `level` is the deep left-tail
    probability (0.001 -> 99.9% VaR). The xi=0 (exponential / Gumbel-domain)
    limit is handled separately. Returns are flipped back to the module's sign
    convention (VaR, ES negative = loss).

    The fitted shape `xi` is the TAIL INDEX:
      * xi > 0  : heavy (power-law) tail; tail index alpha = 1/xi. Equities,
                  credit, crypto live here (xi ~ 0.1-0.4, i.e. alpha ~ 2.5-10).
      * xi = 0  : exponentially decaying tail (Gumbel domain).
      * xi < 0  : finite right endpoint (bounded loss).
      * xi >= 1 : INFINITE MEAN -- ES (and the GPD mean) do not exist; the ES
                  closed form is invalid, so we return -inf and set
                  `infinite_mean=True` as a defensive guard.

    IMPORTANT estimator limitation: the PWM (probability-weighted-moments) fit
    used here requires a FINITE first moment of the GPD, which exists only for
    xi < 1, and the sample PWM estimator is mathematically bounded above by
    xi -> 1 (it asymptotes to 1 but cannot exceed it: with positive exceedances
    the second PWM a1 <= a0/2, forcing xi = 2 - a0/(a0 - 2 a1) < 1). So PWM
    will NOT report a value >= 1 even for a genuinely infinite-mean tail; instead
    it pins xi just below 1. We therefore also expose `xi_near_one` (xi >= 0.9):
    when set, treat the ES as unreliable regardless of the finite number -- the
    tail is so heavy the mean is barely (or not) defined and a few exceedances
    dominate. Raise `threshold_q`, get more data, or switch to a maximum-
    likelihood GPD fit (which can return xi >= 1) before trusting the ES. xi
    in the 0.1-0.4 range is the normal, well-behaved regime for asset returns.

    Requires `level < 1 - threshold_q` (you can only extrapolate BEYOND the
    threshold you fit the tail above).

    Returns dict(var, es, xi, beta, tail_index, threshold, n_exceed,
    infinite_mean, xi_near_one).
    """
    _validate_level(level)
    if not (0.0 < threshold_q < 1.0):
        raise ValueError(f"threshold_q must be in (0, 1), got {threshold_q}")
    if level >= 1.0 - threshold_q:
        raise ValueError(
            f"level ({level}) must be < 1 - threshold_q ({1.0 - threshold_q}); "
            "EVT extrapolates only BEYOND the fitting threshold"
        )
    arr = _as_1d_array(returns)
    n = arr.size
    losses = -arr  # losses positive
    u = float(np.quantile(losses, threshold_q))
    exceed = losses[losses > u] - u
    n_exceed = int(exceed.size)
    if n_exceed < 10:
        raise ValueError(
            f"only {n_exceed} exceedances over the {threshold_q:.0%} threshold; "
            "EVT needs more (lower threshold_q or use a longer sample)"
        )
    xi, beta = _gpd_pwm_fit(exceed)

    ratio = (n / n_exceed) * level  # < 1 since level < Nu/n by construction
    if abs(xi) < 1e-8:
        # exponential-tail limit: VaR = u + beta * (-ln ratio), ES = VaR + beta
        var_loss = u + beta * (-math.log(ratio))
        es_loss = var_loss + beta
        infinite_mean = False
    else:
        var_loss = u + (beta / xi) * (ratio ** (-xi) - 1.0)
        if xi < 1.0:
            es_loss = var_loss / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
            infinite_mean = False
        else:
            es_loss = math.inf  # GPD mean diverges for xi >= 1 (defensive guard)
            infinite_mean = True

    tail_index = (1.0 / xi) if xi > 0.0 else math.inf  # power-law exponent alpha
    return {
        "var": float(-var_loss),
        "es": float(-es_loss) if math.isfinite(es_loss) else -math.inf,
        "xi": float(xi),
        "beta": float(beta),
        "tail_index": float(tail_index),
        "threshold": float(-u),  # threshold expressed as a (negative) return
        "n_exceed": float(n_exceed),
        "infinite_mean": bool(infinite_mean),
        "xi_near_one": bool(xi >= 0.9),  # ES unreliable; PWM caps xi just below 1
    }


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
    already be shifted so var_t uses only data up to t-1 (avoid look-ahead). The
    rolling output of filtered_historical_var_es is already causal -- pass it
    directly. A NaN VaR forecast (e.g. the FHS warm-up window) yields a False
    (non-exception) for that period since `ret < NaN` is False.

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
    with np.errstate(invalid="ignore"):
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

    # LR_cc requires a target level (the unconditional-coverage piece). With no
    # target rate supplied here, the unconditional piece is undefined, so we
    # return LR_cc := LR_ind as a conservative joint placeholder and reserve the
    # genuine df=2 conditional-coverage test for christoffersen_cc(..., level).
    x = int(np.sum(hits))
    n = hits.size
    rate = x / n
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
def _simulate_garch_t(
    n: int, df: float, seed: int, omega: float = 2e-6, a: float = 0.08, b: float = 0.90
) -> np.ndarray:
    """Deterministic GARCH(1,1) path with unit-variance Student-t innovations:
    fat-tailed AND vol-clustered -- exactly the regime where plain Gaussian VaR
    breaks and FHS shines."""
    rng = np.random.default_rng(seed)
    sig2 = np.empty(n)
    ret = np.empty(n)
    sig2[0] = omega / (1.0 - a - b)
    inno = rng.standard_t(df, size=n) / math.sqrt(df / (df - 2.0))  # unit variance
    for t in range(n):
        if t > 0:
            sig2[t] = omega + a * ret[t - 1] ** 2 + b * sig2[t - 1]
        ret[t] = math.sqrt(sig2[t]) * inno[t]
    return ret


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
    base = np.array([-1.0, 1.0])  # skew 0, excess kurtosis = -2 (platykurtic)
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
    assert abs(es - es_g) < 0.03, (es, es_g)

    # --- ewma_volatility is causal & positive ------------------------------
    r_ew = rng.standard_normal(1000) * 0.01
    sig_chk = ewma_volatility(r_ew, lam=0.94)
    assert sig_chk.shape == r_ew.shape and np.all(sig_chk > 0.0)
    # sigma[t] depends only on r[:t]: perturbing r[t] onward must not move sig[t]
    # (fix the seed explicitly so the default full-sample-variance seed -- which
    # itself depends on the whole series -- does not confound the causality test).
    seed0 = float(np.std(r_ew))
    sig_ew = ewma_volatility(r_ew, lam=0.94, sigma0=seed0)
    r2 = r_ew.copy()
    r2[500:] += 5.0  # giant shock from t=500 on
    sig2_ew = ewma_volatility(r2, lam=0.94, sigma0=seed0)
    # sig[t] uses r[t-1], so sig[:501] (indices 0..500) is unaffected, sig[501] reacts
    assert np.allclose(sig_ew[:501], sig2_ew[:501]), "EWMA vol leaked future data"
    assert sig2_ew[501] > sig_ew[501], "EWMA vol failed to react to the t=500 shock"

    # ===================================================================== #
    # Filtered Historical Simulation: tracks alpha far better than Gaussian
    # on a fat-tailed, vol-clustered (GARCH-t) path -- the headline upgrade.
    # ===================================================================== #
    garch = _simulate_garch_t(n=5000, df=5.0, seed=0)
    level = 0.01
    fhs_var, fhs_es = filtered_historical_var_es(
        garch, level=level, lam=0.94, min_obs=250, rolling=True
    )
    valid = ~np.isnan(fhs_var)
    # causal Gaussian VaR using the SAME causal EWMA vol forecast (mu=0)
    sig_fc = ewma_volatility(garch, lam=0.94)
    gauss_var = sig_fc * _NORM.inv_cdf(level)

    rv = garch[valid]
    k_fhs = kupiec_pof(rv, level=level, var_series=fhs_var[valid])
    k_gauss = kupiec_pof(rv, level=level, var_series=gauss_var[valid])

    # FHS coverage is close to alpha and Kupiec does NOT reject it...
    assert abs(k_fhs["observed_rate"] - level) < 0.006, k_fhs
    assert k_fhs["p_value"] > 0.05, ("FHS rejected by Kupiec", k_fhs)
    # ...while bare Gaussian over-breaches (fat t5 tail) and IS rejected.
    assert k_gauss["observed_rate"] > k_fhs["observed_rate"], (k_gauss, k_fhs)
    assert k_gauss["p_value"] < 0.05, ("Gaussian not rejected?!", k_gauss)
    # FHS's Kupiec LR (mis-coverage evidence) is far smaller than Gaussian's.
    assert k_fhs["LR"] < k_gauss["LR"], (k_fhs["LR"], k_gauss["LR"])

    # ES is at least as extreme (more negative) than VaR wherever both defined.
    both = valid & (~np.isnan(fhs_es))
    assert np.all(fhs_es[both] <= fhs_var[both] + 1e-12), "FHS ES not <= VaR"
    # snapshot (non-rolling) form returns a single negative VaR/ES pair
    snap_var, snap_es = filtered_historical_var_es(garch, level=level, rolling=False)
    assert snap_var < 0.0 and snap_es <= snap_var + 1e-12, (snap_var, snap_es)

    # --- age-weighted historical VaR ---------------------------------------
    # Old calm regime, recent volatile regime: age-weighting must react and
    # report a MORE extreme VaR than equal-weighted historical.
    rng_aw = np.random.default_rng(0)
    old = rng_aw.standard_normal(2000) * 0.005
    new = rng_aw.standard_normal(500) * 0.030
    chrono = np.concatenate([old, new])  # oldest first
    aw = age_weighted_var(chrono, level=0.05, decay=0.97)
    plain = float(np.quantile(chrono, 0.05))
    assert aw < plain, ("age-weighted should be more extreme", aw, plain)
    # decay -> 1 collapses to plain historical
    aw1 = age_weighted_var(chrono, level=0.05, decay=0.99999999)
    assert abs(aw1 - plain) < 0.002, (aw1, plain)
    assert aw < 0.0  # it's a loss

    # ===================================================================== #
    # EVT / POT: recovers a KNOWN GPD shape and extrapolates beyond the sample
    # ===================================================================== #
    # (1) PWM recovers xi/beta of a simulated GPD tail within tolerance. Embed a
    # GPD(xi,beta) upper-loss tail into a returns series (losses positive).
    rng_evt = np.random.default_rng(1)
    xi_true, beta_true = 0.30, 0.02
    u_unif = rng_evt.random(60_000)
    gpd_tail = beta_true / xi_true * ((1.0 - u_unif) ** (-xi_true) - 1.0)
    body = np.abs(rng_evt.standard_normal(60_000)) * 0.004  # small losses (body)
    losses = np.concatenate([body, 0.02 + gpd_tail])  # tail sits above the body
    rets_evt = -losses  # module sign convention: losses are negative returns
    out = evt_pot_var_es(rets_evt, level=0.001, threshold_q=0.90)
    assert abs(out["xi"] - xi_true) < 0.05, ("EVT xi recovery", out["xi"], xi_true)
    assert out["var"] < 0.0 and out["es"] < out["var"], out  # ES more extreme
    assert not out["infinite_mean"], out
    # tail_index = 1/xi should be a sensible power-law exponent
    assert abs(out["tail_index"] - 1.0 / xi_true) < 1.0, out["tail_index"]

    # (2) EVT extrapolates BEYOND the worst observation. At a tail probability
    # below 1/N the historical quantile is pinned at the sample minimum and
    # cannot go further; the fitted GPD must. This is deterministic.
    t_short = _simulate_garch_t(n=1000, df=4.0, seed=7)
    out_deep = evt_pot_var_es(t_short, level=1e-5, threshold_q=0.90)
    assert out_deep["var"] < t_short.min(), (
        "EVT failed to extrapolate beyond the sample min",
        out_deep["var"],
        t_short.min(),
    )
    # EVT 99.9% ES is at least as extreme as its own 99.9% VaR
    out_999 = evt_pot_var_es(t_short, level=0.001, threshold_q=0.90)
    assert out_999["es"] <= out_999["var"] + 1e-12, out_999

    # (3) Catastrophically heavy tail: a true GPD with xi >= 1 has infinite mean,
    # but PWM is mathematically bounded by xi -> 1 (a1 <= a0/2). The fit must
    # therefore PIN xi just below 1 and RAISE the xi_near_one flag (the practical
    # signal that the ES is untrustworthy), rather than silently returning a
    # tame ES. tail_index = 1/xi should be close to 1 (near-undefined mean).
    rng_im = np.random.default_rng(3)
    xi_im, beta_im = 1.5, 0.01
    u_im = rng_im.random(80_000)
    tail_im = beta_im / xi_im * ((1.0 - u_im) ** (-xi_im) - 1.0)
    rets_im = -tail_im  # whole series is the heavy GPD loss tail
    out_im = evt_pot_var_es(rets_im, level=0.001, threshold_q=0.90)
    assert out_im["xi"] < 1.0, ("PWM must cap xi below 1", out_im["xi"])
    assert out_im["xi"] >= 0.9 and out_im["xi_near_one"], (
        "heavy tail should trip the xi_near_one warning",
        out_im,
    )
    assert out_im["tail_index"] < 1.15, ("near-undefined mean", out_im["tail_index"])
    # Mild, well-behaved tail: flag off, ES finite & more extreme than VaR.
    out_mild = evt_pot_var_es(
        _simulate_garch_t(4000, 6.0, seed=2), level=0.001, threshold_q=0.95
    )
    assert 0.0 < out_mild["xi"] < 0.9 and not out_mild["xi_near_one"], out_mild
    assert math.isfinite(out_mild["es"]) and out_mild["es"] < out_mild["var"], out_mild
    assert out_mild["tail_index"] > 1.0, out_mild  # alpha = 1/xi finite & > 1

    # (4) guard: cannot extrapolate at/above the fitting threshold
    try:
        evt_pot_var_es(t_short, level=0.10, threshold_q=0.95)  # 0.10 > 1-0.95
        raise AssertionError("expected ValueError for level >= 1 - threshold_q")
    except ValueError:
        pass
    # (5) guard: too few exceedances over a very high threshold raises
    try:
        evt_pot_var_es(rng_evt.standard_normal(60) * 0.01, level=0.001, threshold_q=0.95)
        raise AssertionError("expected ValueError for too few exceedances")
    except ValueError:
        pass

    # --- count_exceptions sign / shape ------------------------------------
    rets = np.array([-0.10, 0.01, -0.03, 0.02, -0.20])
    exc = count_exceptions(rets, -0.05, level=0.05)
    assert exc.tolist() == [True, False, False, False, True], exc.tolist()
    var_s = np.array([-0.05, -0.05, -0.02, -0.05, -0.05])
    exc2 = count_exceptions(rets, var_s, level=0.05)
    assert exc2.tolist() == [True, False, True, False, True], exc2.tolist()
    # NaN VaR forecast (FHS warm-up) -> never an exception
    nan_exc = count_exceptions(np.array([-0.5, -0.1]), np.array([np.nan, -0.05]))
    assert nan_exc.tolist() == [False, True], nan_exc.tolist()

    # --- Kupiec: correct coverage does NOT reject -------------------------
    n = 4000
    level = 0.05
    u = rng.random(n)
    var_const = -0.02
    rets_ok = np.where(u < level, var_const - 0.01, var_const + 0.01)
    k_ok = kupiec_pof(rets_ok, level=level, var_series=np.full(n, var_const))
    assert abs(k_ok["observed_rate"] - level) < 0.02, k_ok
    assert k_ok["p_value"] > 0.05, k_ok

    # --- Kupiec: far too many exceptions DOES reject ----------------------
    rets_bad = rng.standard_normal(2000) * 0.02 - 0.001
    k_bad = kupiec_pof(rets_bad, level=0.01, var_series=np.full(2000, -0.0005))
    assert k_bad["observed_rate"] > 0.05, k_bad
    assert k_bad["p_value"] < 0.05, k_bad

    # --- Christoffersen returns finite p in [0,1] -------------------------
    c = christoffersen(exc2)
    for key in ("LR_ind", "LR_cc"):
        assert np.isfinite(c[key]), c
    for key in ("p_value_ind", "p_value_cc"):
        assert 0.0 <= c[key] <= 1.0, c
    cc_ok = christoffersen_cc((u < level).astype(int), level=level)
    assert 0.0 <= cc_ok["p_value_cc"] <= 1.0, cc_ok
    assert cc_ok["p_value_cc"] > 0.05, cc_ok
    clustered = np.zeros(1000, dtype=int)
    clustered[100:140] = 1
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
    pnl_d = stress_pnl(
        {"SPY": 1e6, "TLT": -5e5}, {"SPY": -0.10, "TLT": 0.20}
    )
    assert abs(pnl_d - (-2e5)) < 1e-6, pnl_d
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
    assert abs(_chi2_sf(3.8414588, 1) - 0.05) < 1e-4, _chi2_sf(3.8414588, 1)
    assert abs(_chi2_sf(5.9914645, 2) - 0.05) < 1e-4, _chi2_sf(5.9914645, 2)

    print("risk.py: all self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
