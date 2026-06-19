"""
metrics.py - performance and risk metrics for quant strategies.

CONVENTIONS (applied consistently everywhere):
- Inputs are *simple* periodic returns unless stated otherwise. Simple returns
  compound multiplicatively: total = prod(1+r) - 1. Log returns add.
- Sample std/var use ddof=1 (unbiased).
- `periods_per_year` (ppy) annualizes: daily ~252, weekly 52, monthly 12;
  crypto / 24-7 often 365.
- `risk_free` is an ANNUAL simple rate, converted to per-period internally via
  (1+rf)**(1/ppy) - 1.
- Returns may be a pandas Series / ndarray / list. NaNs are DROPPED (treated as
  "no observation", never forward-filled to 0%).
- VaR / CVaR are returned as RETURNS: a 5% VaR of -0.03 means "the worst 5% of
  periods lose at least 3%". Losses are negative numbers.

References:
- Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier"
  (Probabilistic & Deflated Sharpe Ratio).
- Lo (2002), "The Statistics of Sharpe Ratios" (autocorrelation caveat: the
  sqrt(ppy) scaling assumes iid returns; serial correlation biases it).
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_NORM = NormalDist()
_EULER = 0.5772156649015329  # Euler-Mascheroni constant
_EPS = 1e-12  # std below this is numerically zero (a constant return series leaves
              # ~1e-19 float residue, far under any real return volatility)


def _clean(returns) -> np.ndarray:
    """Coerce to a 1-D float array with NaNs dropped."""
    return pd.Series(returns, dtype="float64").dropna().to_numpy()


# --------------------------------------------------------------------------- #
# Returns / equity
# --------------------------------------------------------------------------- #
def returns_from_prices(prices, kind: str = "simple") -> pd.Series:
    """Periodic returns from a price series. kind in {'simple', 'log'}."""
    p = pd.Series(prices, dtype="float64")
    if kind == "log":
        return np.log(p).diff().dropna()
    return p.pct_change(fill_method=None).dropna()  # fill_method=None avoids leaking ffill


def cumulative_returns(returns, kind: str = "simple") -> pd.Series:
    r = pd.Series(returns, dtype="float64").dropna()
    return r.cumsum() if kind == "log" else (1 + r).cumprod() - 1


def equity_curve(returns, initial: float = 1.0) -> pd.Series:
    r = pd.Series(returns, dtype="float64").dropna()
    return initial * (1 + r).cumprod()


# --------------------------------------------------------------------------- #
# Risk-adjusted performance
# --------------------------------------------------------------------------- #
def _per_period_rf(risk_free_annual: float, ppy: int) -> float:
    return (1 + risk_free_annual) ** (1 / ppy) - 1


def annualized_return(returns, periods_per_year: int = 252) -> float:
    """Geometric (CAGR-style) annualized return."""
    a = _clean(returns)
    if a.size == 0:
        return float("nan")
    growth = float(np.prod(1 + a))
    if growth <= 0:
        return -1.0  # capital wiped out
    return growth ** (periods_per_year / a.size) - 1


def annualized_volatility(returns, periods_per_year: int = 252) -> float:
    a = _clean(returns)
    if a.size < 2:
        return float("nan")
    return float(np.std(a, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(returns, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe = mean(excess)/std(excess, ddof=1) * sqrt(ppy)."""
    a = _clean(returns)
    if a.size < 2:
        return float("nan")
    ex = a - _per_period_rf(risk_free, periods_per_year)
    sd = np.std(ex, ddof=1)
    if sd <= _EPS:
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(returns, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino. The hurdle (minimum acceptable return) is the per-period
    risk-free rate; downside deviation counts only returns below the hurdle but
    divides by the FULL sample size. (A single hurdle - do not also pass a separate
    target, which would double-count it.)"""
    a = _clean(returns)
    if a.size < 2:
        return float("nan")
    mar = _per_period_rf(risk_free, periods_per_year)
    excess = a - mar
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(np.mean(downside ** 2))
    if dd <= _EPS:
        return float("nan")
    return float(excess.mean() / dd * math.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# Sharpe significance (Lo 2002)
# --------------------------------------------------------------------------- #
def sharpe_tstat(returns) -> float:
    """t-statistic of the Sharpe ratio under iid returns (Lo 2002):
        t = SR_pp / sqrt((1 + 0.5*SR_pp^2) / n)
    with SR_pp the per-period Sharpe. NOTE: this is NOT SR*sqrt(n) - that naive
    form drops the (1 + 0.5*SR^2) finite-Sharpe correction."""
    a = _clean(returns)
    n = a.size
    if n < 3:
        return float("nan")
    sr = _per_period_sharpe(a)
    if math.isnan(sr):
        return float("nan")
    return sr / math.sqrt((1 + 0.5 * sr ** 2) / n)


def sharpe_se(returns, periods_per_year: int = 252) -> float:
    """Standard error of the ANNUALIZED Sharpe under iid returns (Lo 2002):
        SE = sqrt(ppy) * sqrt((1 + 0.5*SR_pp^2) / n)."""
    a = _clean(returns)
    n = a.size
    if n < 3:
        return float("nan")
    sr = _per_period_sharpe(a)
    if math.isnan(sr):
        return float("nan")
    return math.sqrt(periods_per_year) * math.sqrt((1 + 0.5 * sr ** 2) / n)


def lo_annualization_factor(returns, q: int) -> float:
    """Lo (2002) autocorrelation-adjusted Sharpe annualization factor:
        eta(q) = q / sqrt( q + 2*sum_{k=1}^{q-1} (q-k)*rho_k )
    where rho_k is the lag-k return autocorrelation. Use
    annualized_SR = eta(q) * per_period_SR instead of the naive sqrt(q) when
    returns are serially correlated (positive autocorrelation -> eta < sqrt(q)).

    Small-sample guard: high lags whose overlap would have < MIN_PAIRS points are
    dropped (they carry tiny (q-k) weight anyway), so the function never returns
    NaN from estimating lag-(q-1) autocorrelation on a couple of points - the bug
    in a naive implementation when q ~ n. Falls back to sqrt(q) if no lag is
    estimable."""
    a = _clean(returns)
    n = a.size
    if n < 3 or q < 1:
        return float("nan")
    MIN_PAIRS = 20
    max_lag = min(q - 1, n - MIN_PAIRS)
    if max_lag < 1:
        return math.sqrt(q)
    mean = a.mean()
    var = float(np.mean((a - mean) ** 2))
    if var <= _EPS:
        return math.sqrt(q)
    s = 0.0
    for k in range(1, max_lag + 1):
        rho = float(np.mean((a[:-k] - mean) * (a[k:] - mean))) / var
        s += (q - k) * rho
    denom = q + 2.0 * s
    if denom <= _EPS:
        return math.sqrt(q)  # degenerate noisy estimate (e.g. q >> n): fall back to iid
    return q / math.sqrt(denom)


def max_drawdown(returns) -> float:
    """Largest peak-to-trough decline of the compounded equity curve (negative)."""
    a = _clean(returns)
    if a.size == 0:
        return float("nan")
    eq = np.cumprod(1 + a)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def calmar_ratio(returns, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if mdd is None or math.isnan(mdd) or mdd == 0:
        return float("nan")
    return annualized_return(returns, periods_per_year) / abs(mdd)


def hit_rate(returns) -> float:
    a = _clean(returns)
    return float(np.mean(a > 0)) if a.size else float("nan")


def profit_factor(returns) -> float:
    a = _clean(returns)
    gains = a[a > 0].sum()
    losses = a[a < 0].sum()
    if losses == 0:
        return float("nan") if gains == 0 else float("inf")
    return float(gains / abs(losses))


def turnover(weights) -> float:
    """Average one-way turnover = mean over time of 0.5*sum|w_t - w_{t-1}|.
    A full rebalance of a fully-invested book is ~100% (1.0)."""
    w = pd.DataFrame(weights).fillna(0.0)
    changes = 0.5 * w.diff().abs().sum(axis=1)
    tail = changes.iloc[1:]
    return float(tail.mean()) if len(tail) else float("nan")


def information_ratio(returns, benchmark, periods_per_year: int = 252) -> float:
    """Annualized IR of active (return - benchmark) returns."""
    df = pd.concat([pd.Series(returns, dtype="float64"),
                    pd.Series(benchmark, dtype="float64")], axis=1).dropna()
    if len(df) < 2:
        return float("nan")
    active = (df.iloc[:, 0] - df.iloc[:, 1]).to_numpy()
    sd = np.std(active, ddof=1)
    if sd <= _EPS:
        return float("nan")
    return float(active.mean() / sd * math.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# Overfitting-aware Sharpe statistics (Bailey & Lopez de Prado)
# --------------------------------------------------------------------------- #
def _skew_kurt(a: np.ndarray) -> tuple[float, float]:
    """Sample skewness (g1) and NON-excess kurtosis (g2; normal -> 3.0)."""
    s = np.std(a, ddof=0)
    if s == 0:
        return 0.0, 3.0
    z = (a - a.mean()) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def _per_period_sharpe(a: np.ndarray) -> float:
    sd = np.std(a, ddof=1)
    return float(a.mean() / sd) if sd > _EPS else float("nan")


def probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0,
                               risk_free: float = 0.0,
                               periods_per_year: int = 252) -> float:
    """Probability that the true PER-PERIOD Sharpe exceeds `benchmark_sr`
    (a per-period Sharpe, default 0), correcting for skew/kurtosis and sample
    length. Bailey & Lopez de Prado (2012)."""
    a = _clean(returns) - _per_period_rf(risk_free, periods_per_year)
    n = a.size
    if n < 3:
        return float("nan")
    sr = _per_period_sharpe(a)
    if math.isnan(sr):
        return float("nan")
    g1, g2 = _skew_kurt(a)
    denom = math.sqrt(max(1e-12, 1 - g1 * sr + (g2 - 1) / 4.0 * sr ** 2))
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / denom
    return _NORM.cdf(z)


def expected_max_sharpe(trial_sharpe_std: float, n_trials: int) -> float:
    """Expected maximum of n_trials iid Sharpe estimates under the null
    (per-period units). Used by the Deflated Sharpe Ratio."""
    if n_trials <= 1:
        return 0.0
    z1 = _NORM.inv_cdf(1 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1 - 1.0 / (n_trials * math.e))
    return trial_sharpe_std * ((1 - _EULER) * z1 + _EULER * z2)


def deflated_sharpe_ratio(returns, n_trials: int, trial_sharpe_std: float,
                          risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """PSR measured against the expected maximum Sharpe from `n_trials`
    independent strategies. `trial_sharpe_std` = std (across trials) of the
    PER-PERIOD Sharpe estimates. A DSR < 0.95 means the result is plausibly the
    product of multiple testing. Bailey & Lopez de Prado (2012).

    Note: assumes trials are roughly independent; correlated grid searches have a
    smaller *effective* n_trials, making this conservative."""
    sr0 = expected_max_sharpe(trial_sharpe_std, n_trials)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr0,
                                      risk_free=risk_free, periods_per_year=periods_per_year)


# --------------------------------------------------------------------------- #
# Tail risk
# --------------------------------------------------------------------------- #
def value_at_risk(returns, level: float = 0.05, method: str = "historical") -> float:
    """VaR as a return (negative = loss). level is the lower-tail probability."""
    a = _clean(returns)
    if a.size == 0:
        return float("nan")
    if method == "historical":
        return float(np.quantile(a, level))
    if method == "gaussian":
        return float(a.mean() + np.std(a, ddof=1) * _NORM.inv_cdf(level))
    raise ValueError(f"unknown method: {method}")


def conditional_value_at_risk(returns, level: float = 0.05,
                              method: str = "historical") -> float:
    """Expected return conditional on being at/below the VaR threshold (CVaR /
    expected shortfall). Returned as a return (more negative than VaR)."""
    a = _clean(returns)
    if a.size == 0:
        return float("nan")
    var = value_at_risk(a, level, method)
    tail = a[a <= var]
    return float(tail.mean()) if tail.size else float(var)


# --------------------------------------------------------------------------- #
# Self-tests (analytic cases) - run: python metrics.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # geometric annualized return: two +10% periods, ppy=2 -> 1.1*1.1 - 1 = 0.21
    assert np.isclose(annualized_return([0.1, 0.1], periods_per_year=2), 0.21)

    # annualized volatility: std(ddof=1) of [+1%,-1%] is sqrt(2)*1% -> *sqrt(252)
    assert np.isclose(annualized_volatility([0.01, -0.01], 252),
                      math.sqrt(0.0002) * math.sqrt(252))

    # Sharpe self-consistency (rf=0): annualized / sqrt(252) == per-period Sharpe
    a = rng.normal(0.001, 0.01, 1000)
    assert np.isclose(sharpe_ratio(a, 0.0, 252) / math.sqrt(252),
                      a.mean() / np.std(a, ddof=1))

    # Lo (2002) Sharpe significance: t-stat and SE use the (1+0.5*SR^2) correction
    sr_pp = a.mean() / np.std(a, ddof=1)
    assert np.isclose(sharpe_tstat(a), sr_pp / math.sqrt((1 + 0.5 * sr_pp ** 2) / len(a)))
    assert np.isclose(sharpe_se(a) / math.sqrt(252),
                      math.sqrt((1 + 0.5 * sr_pp ** 2) / len(a)))
    # autocorrelation factor: AR(1) with positive autocorr -> eta < sqrt(q)
    e = rng.normal(0, 0.01, 5000)
    ar = np.zeros(5000)
    for i in range(1, 5000):
        ar[i] = 0.4 * ar[i - 1] + e[i]
    assert lo_annualization_factor(ar, 12) < math.sqrt(12)
    # white noise ~ sqrt(q); small-sample guard never returns NaN
    wn = rng.normal(0, 0.01, 5000)
    assert abs(lo_annualization_factor(wn, 12) - math.sqrt(12)) < 0.4 * math.sqrt(12)
    assert not math.isnan(lo_annualization_factor(rng.normal(0, 0.01, 30), 252))

    # max drawdown on a known path 100 -> 110 -> 90: trough is 90/110 - 1
    assert np.isclose(max_drawdown(returns_from_prices([100, 110, 90, 120])),
                      90 / 110 - 1)

    # discrete metrics
    assert np.isclose(hit_rate([1, -1, 1, 0]), 0.5)
    assert np.isclose(profit_factor([1, -1, 2, -1]), 1.5)
    assert np.isclose(turnover([[0, 0], [0.5, 0.5], [0.5, 0.5]]), 0.25)

    # undefined cases -> nan (zero vol Sharpe; all-positive Sortino)
    assert math.isnan(sharpe_ratio([0.001] * 10))
    assert math.isnan(sortino_ratio([0.01, 0.02, 0.03]))

    # PSR / DSR are probabilities in [0,1]; deflating by SR0>0 cannot raise it
    g = rng.normal(0.0015, 0.01, 750)
    psr0 = probabilistic_sharpe_ratio(g, 0.0)
    dsr = deflated_sharpe_ratio(g, n_trials=50, trial_sharpe_std=0.5 / math.sqrt(252))
    assert 0.0 <= psr0 <= 1.0 and 0.0 <= dsr <= 1.0
    assert dsr <= psr0 + 1e-9

    # VaR/CVaR sign convention: losses negative; CVaR no greater than VaR
    var = value_at_risk(g, 0.05)
    cvar = conditional_value_at_risk(g, 0.05)
    assert var < 0 and cvar <= var

    print("metrics.py: all self-tests passed")
