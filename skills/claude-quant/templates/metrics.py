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
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" (effective number
  of independent trials when grid points are correlated).
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


def expected_max_sharpe(trial_sharpe_std: float, n_trials: float) -> float:
    """Expected maximum of n_trials iid Sharpe estimates under the null
    (per-period units). Used by the Deflated Sharpe Ratio.

    `n_trials` may be a non-integer EFFECTIVE count (see
    effective_number_of_trials); the order-statistic approximation is smooth in N
    and well-defined for real N > 1."""
    if n_trials <= 1:
        return 0.0
    z1 = _NORM.inv_cdf(1 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1 - 1.0 / (n_trials * math.e))
    return trial_sharpe_std * ((1 - _EULER) * z1 + _EULER * z2)


# --------------------------------------------------------------------------- #
# Effective number of trials (correlated-trial deflation, LdP 2014)
# --------------------------------------------------------------------------- #
def _trial_corr_matrix(trial_returns_matrix) -> np.ndarray:
    """Correlation matrix of the N per-trial return series.

    Input is a (T x N) array: T time periods (rows) by N trials/configs (cols),
    each column the per-period return stream of one strategy variant. Any row with
    a NaN is dropped so the matrix is computed on a common, aligned sample.
    Zero-variance (constant) columns are kept with zero off-diagonal correlation
    and unit diagonal so they neither crash corrcoef nor inflate N_eff."""
    m = np.asarray(pd.DataFrame(trial_returns_matrix, dtype="float64").dropna(),
                   dtype=float)
    if m.ndim != 2 or m.shape[1] < 1:
        raise ValueError("trial_returns_matrix must be 2-D (T x N) with N >= 1")
    T, N = m.shape
    if T < 3:
        raise ValueError("need at least 3 aligned time periods to estimate correlation")
    sd = m.std(axis=0, ddof=1)
    # Correlate only the non-degenerate columns; constants get identity rows/cols.
    good = sd > _EPS
    C = np.eye(N)
    if good.sum() >= 2:
        sub = np.corrcoef(m[:, good], rowvar=False)
        idx = np.where(good)[0]
        C[np.ix_(idx, idx)] = sub
    # Guard tiny numerical drift from corrcoef so eigenvalues stay clean.
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C


def effective_number_of_trials(trial_returns_matrix, method: str = "cluster") -> float:
    """Effective number of INDEPENDENT trials, N_eff, from the correlation
    structure of N candidate strategies' per-period return streams.

    Why: the Deflated Sharpe Ratio penalizes the best of N trials by the expected
    maximum of N iid draws. A 1000-point grid of near-duplicate variants is NOT
    1000 independent bets - their Sharpe estimates are highly correlated, so the
    expected max is far smaller and using the raw count N=1000 over-deflates and
    can bury a genuine edge. Estimate the EFFECTIVE count and pass that to
    deflated_sharpe_ratio (LdP 2014).

    `trial_returns_matrix`: (T x N) per-period returns, one column per trial.

    method:
      'spectral' / 'cluster' (default) - participation ratio of the eigenvalue
          spectrum of the NxN trial correlation matrix:
              N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2
          For a correlation matrix sum_i lambda_i = trace = N, so equivalently
              N_eff = N^2 / sum_i lambda_i^2 = N^2 / ||C||_F^2
          (||.||_F = Frobenius norm). This is the spectral / participation-ratio
          count of effective independent dimensions: it equals N when all trials
          are mutually orthogonal (C = I, every lambda = 1) and collapses toward 1
          as trials become perfectly correlated (one large eigenvalue ~ N, the
          rest ~ 0). Dependency-free and monotone in pairwise correlation. This is
          the recommended estimator to feed the DSR.

      'threshold' - a conservative LOWER-bound count: cluster trials so that any
          two with |corr| >= 0.95 land in the same cluster (connected components),
          and return the number of clusters. Counts blocks of near-identical
          variants as one. Use as a sanity floor, not as the DSR input. For a
          tunable threshold use effective_number_of_trials_threshold.

    Returns a float in [1, N].
    """
    C = _trial_corr_matrix(trial_returns_matrix)
    N = C.shape[0]
    if N == 1:
        return 1.0
    if method == "threshold":
        return float(_threshold_cluster_count(C, corr_threshold=0.95))
    if method not in ("spectral", "cluster"):
        raise ValueError(f"unknown method: {method!r} (use 'spectral'/'cluster' or 'threshold')")
    eig = np.linalg.eigvalsh(C)
    eig = np.clip(eig, 0.0, None)  # symmetric PSD-ish; kill tiny negative round-off
    denom = float(np.sum(eig ** 2))
    if denom <= _EPS:
        return 1.0
    n_eff = float(np.sum(eig)) ** 2 / denom
    # Bound to [1, N]: the participation ratio is mathematically in [1, N].
    return float(min(max(n_eff, 1.0), N))


def _threshold_cluster_count(C: np.ndarray, corr_threshold: float = 0.95) -> int:
    """Number of clusters when trials with |corr| >= threshold are merged
    (connected components / single-linkage at the threshold). Conservative
    lower bound on effective N: blocks of near-duplicates collapse to one."""
    N = C.shape[0]
    adj = np.abs(C) >= corr_threshold
    seen = np.zeros(N, dtype=bool)
    clusters = 0
    for i in range(N):
        if seen[i]:
            continue
        clusters += 1
        stack = [i]
        seen[i] = True
        while stack:
            j = stack.pop()
            nbrs = np.where(adj[j] & ~seen)[0]
            for k in nbrs:
                seen[k] = True
                stack.append(int(k))
    return clusters


def effective_number_of_trials_threshold(trial_returns_matrix,
                                         corr_threshold: float = 0.95) -> int:
    """Conservative lower-bound effective trial count: number of correlation
    clusters at |corr| >= corr_threshold (see effective_number_of_trials,
    method='threshold', but with a tunable threshold)."""
    C = _trial_corr_matrix(trial_returns_matrix)
    return _threshold_cluster_count(C, corr_threshold=corr_threshold)


def trial_sharpe_std_from_matrix(trial_returns_matrix,
                                 periods_per_year: int = 252,
                                 annualized: bool = False) -> float:
    """Cross-trial std of the PER-PERIOD Sharpe estimates, computed directly from
    the (T x N) trial-returns matrix - the OTHER input the DSR needs and the one
    users most often get wrong.

    For each trial (column) compute its per-period Sharpe mean/std(ddof=1), then
    take the std (ddof=1) ACROSS the N trial Sharpes. Degenerate (zero-variance)
    trials are dropped. Returns a PER-PERIOD figure by default (the units
    deflated_sharpe_ratio / expected_max_sharpe expect); set annualized=True to
    multiply by sqrt(ppy)."""
    m = np.asarray(pd.DataFrame(trial_returns_matrix, dtype="float64").dropna(),
                   dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("need a 2-D (T x N) matrix with N >= 2 trials")
    mean = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    good = sd > _EPS
    if good.sum() < 2:
        return float("nan")
    sr_pp = mean[good] / sd[good]
    out = float(np.std(sr_pp, ddof=1))
    return out * math.sqrt(periods_per_year) if annualized else out


def deflated_sharpe_ratio(returns, n_trials: float, trial_sharpe_std: float,
                          risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """PSR measured against the expected maximum Sharpe from `n_trials`
    strategies. `trial_sharpe_std` = std (across trials) of the PER-PERIOD Sharpe
    estimates (see trial_sharpe_std_from_matrix). A DSR < 0.95 means the result is
    plausibly the product of multiple testing. Bailey & Lopez de Prado (2012).

    `n_trials` may be either:
      - the raw integer count of configurations tried, OR
      - a non-integer EFFECTIVE count from effective_number_of_trials(...).

    Prefer the EFFECTIVE count when the trials are a correlated grid: passing the
    raw count treats near-duplicate variants as independent bets and OVER-deflates
    (it can bury a real edge). Estimate N_eff from the trial-return correlation
    matrix and pass it here. The order-statistic expected-max is smooth in N, so a
    float effective count is valid."""
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

    # ----------------------------------------------------------------------- #
    # Effective number of trials (correlated-trial deflation)
    # ----------------------------------------------------------------------- #
    T, N = 2000, 40

    # (1) Orthogonal (independent) trials: N_eff ~ N (within sampling noise).
    indep = rng.normal(0.0, 0.01, size=(T, N))
    n_eff_indep = effective_number_of_trials(indep)
    assert 1.0 <= n_eff_indep <= N
    assert n_eff_indep > 0.8 * N, n_eff_indep   # ~independent => most dims survive

    # (2) Perfectly duplicated trials: one underlying series copied N times
    #     collapses N_eff toward 1 (a single effective bet).
    base = rng.normal(0.0, 0.01, size=(T, 1))
    dup = np.tile(base, (1, N))
    n_eff_dup = effective_number_of_trials(dup)
    assert n_eff_dup < 1.05, n_eff_dup
    # threshold cluster count agrees: all duplicates are one cluster
    assert effective_number_of_trials_threshold(dup) == 1

    # (3) Monotonicity: injecting MORE common-factor correlation lowers N_eff.
    #     x_i = sqrt(rho)*F + sqrt(1-rho)*eps_i  => pairwise corr = rho.
    def make_corr_block(rho, seed):
        r = np.random.default_rng(seed)
        F = r.normal(0.0, 1.0, size=(T, 1))
        eps = r.normal(0.0, 1.0, size=(T, N))
        return math.sqrt(rho) * F + math.sqrt(1 - rho) * eps

    neffs = [effective_number_of_trials(make_corr_block(rho, 100 + i))
             for i, rho in enumerate([0.0, 0.3, 0.6, 0.9])]
    # strictly decreasing in injected correlation (seeded, so deterministic)
    assert all(neffs[i] > neffs[i + 1] for i in range(len(neffs) - 1)), neffs
    assert neffs[0] > 0.8 * N and neffs[-1] < 0.25 * N, neffs

    # (4) effective-N fix for over-deflation: with correlated trials, using N_eff
    #     gives a HIGHER (less punitive) DSR than using the raw count N.
    corr_trials = make_corr_block(0.8, 7)
    winner = rng.normal(0.0012, 0.01, 750)            # a genuinely strong stream
    tsr_std = trial_sharpe_std_from_matrix(corr_trials)   # per-period units
    n_eff = effective_number_of_trials(corr_trials)
    assert 1.0 < n_eff < N
    dsr_raw = deflated_sharpe_ratio(winner, n_trials=N, trial_sharpe_std=tsr_std)
    dsr_eff = deflated_sharpe_ratio(winner, n_trials=n_eff, trial_sharpe_std=tsr_std)
    assert 0.0 <= dsr_raw <= 1.0 and 0.0 <= dsr_eff <= 1.0
    assert dsr_eff >= dsr_raw - 1e-12, (dsr_eff, dsr_raw)  # fewer trials => less deflation

    # (5) trial_sharpe_std_from_matrix: matches a manual per-column computation,
    #     and annualized == per-period * sqrt(ppy).
    cols_sr = corr_trials.mean(0) / corr_trials.std(0, ddof=1)
    assert np.isclose(trial_sharpe_std_from_matrix(corr_trials), np.std(cols_sr, ddof=1))
    assert np.isclose(trial_sharpe_std_from_matrix(corr_trials, 252, annualized=True),
                      trial_sharpe_std_from_matrix(corr_trials) * math.sqrt(252))

    # (6) expected_max_sharpe accepts a float effective count and is monotone:
    #     more trials => larger expected max under the null.
    assert expected_max_sharpe(0.5, 1) == 0.0
    assert expected_max_sharpe(0.5, 100) > expected_max_sharpe(0.5, 10.5) > 0

    # VaR/CVaR sign convention: losses negative; CVaR no greater than VaR
    var = value_at_risk(g, 0.05)
    cvar = conditional_value_at_risk(g, 0.05)
    assert var < 0 and cvar <= var

    print("metrics.py: all self-tests passed")
