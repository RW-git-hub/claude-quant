"""regime.py - Time-series & regime-modeling toolkit for quant research.

Pure numpy / pandas / standard library. No scipy, no statsmodels: the GARCH
fit, Kalman filters, HMM (Baum-Welch + Viterbi), CUSUM, HAR-RV and GJR-GARCH
are implemented by hand so the file is dependency-light and auditable.
`statistics.NormalDist` supplies the Gaussian pdf/cdf where needed.

Conventions
-----------
- `returns` are SIMPLE per-period returns (decimal, e.g. 0.01 == 1%). Volatility
  helpers operate on the return level directly; demean only where stated. For
  daily data the natural annualization of a vol is `vol * sqrt(252)` (not done
  here -- these functions return per-period vol so the caller controls ppy).
- Volatility = standard deviation (sqrt of variance), per period. Realized
  *variance* (RV) functions return variance (the square); HAR-RV is fit on RV.
- All filters are CAUSAL: the value at index t uses information up to and
  including t (filtered, not smoothed). To use a regime/vol signal as a position
  you must still lag it vs the return it earns (pnl_t = pos.shift(1) * ret_t),
  exactly as elsewhere in this skill. These functions do NOT lag for you, with
  one exception clearly documented: `har_rv` returns a STRICTLY trailing
  one-step-ahead forecast whose entry at index t uses only RV through t-1.

Detect/fix framing for the common pitfalls is in the docstrings; the
`__main__` block self-verifies every function on analytic/synthetic cases.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_NORM = NormalDist()
_LOG2PI = float(np.log(2.0 * np.pi))

ArrayLike = Union[pd.Series, np.ndarray, Sequence[float]]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_array(x: ArrayLike) -> Tuple[np.ndarray, Optional[pd.Index]]:
    """Return (float ndarray, index-or-None). Preserves a pandas index so we can
    hand results back with the caller's labels."""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float), x.index
    arr = np.asarray(x, dtype=float)
    return arr, None


def _wrap(values: np.ndarray, index: Optional[pd.Index], name: str) -> pd.Series:
    """Re-attach an index (default RangeIndex) and a name to a result array."""
    if index is None:
        index = pd.RangeIndex(len(values))
    return pd.Series(values, index=index, name=name)


def _check_finite(arr: np.ndarray, what: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} contains non-finite values (NaN/inf); "
                         "clean or impute before calling.")


# --------------------------------------------------------------------------- #
# 1. EWMA volatility (RiskMetrics-style)
# --------------------------------------------------------------------------- #
def ewma_vol(returns: ArrayLike, lam: float = 0.94) -> pd.Series:
    """EWMA volatility series (sqrt of EWMA variance), RiskMetrics convention.

    Recursion on variance:  s2_t = lam * s2_{t-1} + (1 - lam) * r_{t-1}^2,
    i.e. today's variance estimate uses returns up to t-1 plus the realized
    squared return; we report the in-sample EWMA variance aligned to t. Returns
    are treated as zero-mean (standard for daily risk; demeaning a noisy daily
    mean usually hurts). lam=0.94 is the RiskMetrics daily default (~11-day
    half-life ln(0.5)/ln(lam); center-of-mass lam/(1-lam) ~16 days); use ~0.97
    for monthly.

    Initialization: seed s2_0 with r_0^2 so the first value is defined rather
    than 0 (a common pitfall: seeding with 0 biases the early series low and
    creates a spurious "vol ramp" at the start of every backtest).

    Detect/fix: if your EWMA vol looks suspiciously smooth and lagging, check
    lam -- values >0.97 on daily data react slowly and will under-respond to
    jumps; if it spikes only on day 2 of a shock, you forgot the (1-lam) weight.

    Parameters
    ----------
    returns : per-period simple returns.
    lam : decay in (0, 1); larger = smoother / longer memory.

    Returns
    -------
    pd.Series of per-period volatility (same length & index as input).
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0, 1).")
    r, idx = _as_array(returns)
    _check_finite(r, "returns")
    n = r.size
    if n == 0:
        return _wrap(np.array([], dtype=float), idx, "ewma_vol")

    var = np.empty(n, dtype=float)
    var[0] = r[0] ** 2  # seed with first squared return
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return _wrap(np.sqrt(var), idx, "ewma_vol")


# --------------------------------------------------------------------------- #
# 2. GARCH(1,1) filter + variance-targeting fit
# --------------------------------------------------------------------------- #
def garch11_filter(returns: ArrayLike, omega: float, alpha: float,
                   beta: float) -> pd.Series:
    """Conditional volatility series for a GARCH(1,1) given parameters.

    Model (zero-mean returns):
        r_t = sqrt(h_t) * z_t,   z_t ~ iid (0, 1)
        h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}

    Stationarity requires omega > 0, alpha >= 0, beta >= 0 and persistence
    alpha + beta < 1; the unconditional variance is omega / (1 - alpha - beta),
    used to seed h_0 (better than the sample variance because it is internally
    consistent with the supplied params).

    Returns sqrt(h_t), strictly positive by construction (omega > 0).
    """
    if omega <= 0:
        raise ValueError("omega must be > 0 for positive variance.")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha, beta must be >= 0.")
    persistence = alpha + beta
    r, idx = _as_array(returns)
    _check_finite(r, "returns")
    n = r.size
    if n == 0:
        return _wrap(np.array([], dtype=float), idx, "garch_vol")

    if persistence < 1.0:
        uncond = omega / (1.0 - persistence)
    else:
        # non-stationary params: fall back to sample variance for the seed
        uncond = float(np.var(r)) if n > 1 else r[0] ** 2
        uncond = max(uncond, 1e-12)

    h = np.empty(n, dtype=float)
    h[0] = uncond
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    return _wrap(np.sqrt(h), idx, "garch_vol")


def _garch_negloglik(r: np.ndarray, omega: float, alpha: float,
                     beta: float) -> float:
    """Gaussian negative log-likelihood of GARCH(1,1) conditional variances.
    Lower is better. Returns +inf for degenerate variance paths."""
    n = r.size
    persistence = alpha + beta
    if persistence < 1.0:
        uncond = omega / (1.0 - persistence)
    else:
        uncond = float(np.var(r)) if n > 1 else r[0] ** 2
    if not np.isfinite(uncond) or uncond <= 0:
        return np.inf
    h_prev = uncond
    nll = 0.0
    for t in range(n):
        if t == 0:
            h = uncond
        else:
            h = omega + alpha * r[t - 1] ** 2 + beta * h_prev
        if h <= 0 or not np.isfinite(h):
            return np.inf
        nll += 0.5 * (_LOG2PI + np.log(h) + r[t] ** 2 / h)
        h_prev = h
    return nll


def garch11_fit(returns: ArrayLike,
                grid: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Estimate GARCH(1,1) params by variance targeting + coarse grid search.

    Variance targeting fixes the unconditional variance to the sample variance
    of (demeaned) returns:
        omega = sample_var * (1 - alpha - beta).
    We then grid-search (alpha, beta) over the stationarity region
    (alpha + beta < 1) maximizing the Gaussian log-likelihood. This is a
    standard, robust quasi-MLE shortcut: it removes one parameter, guarantees a
    sensible long-run vol, and avoids the flat-likelihood ridge that trips up
    naive full optimizers.

    Caveats / detect-fix:
    - Grid search is coarse by design; for production use refine the grid near
      the optimum or hand off to a proper optimizer. Persistence near 1.0
      (alpha+beta -> 1, IGARCH-like) is common for daily equity vol and is fine,
      but if you estimate persistence > 1 you've lost stationarity -- usually a
      sign of a structural break or outliers; winsorize and re-fit.
    - Variance targeting is biased if the sample mean is a poor unconditional
      var estimate (short or non-stationary samples). It is a feature for
      stability, not a free lunch.

    Parameters
    ----------
    returns : per-period simple returns.
    grid : optional 1-D grid of candidate values shared by alpha and beta
        (defaults to np.linspace(0.0, 0.98, 50)). Only pairs with
        alpha + beta < 1 are evaluated.

    Returns
    -------
    dict(omega, alpha, beta, persistence=alpha+beta).
    """
    r, _ = _as_array(returns)
    _check_finite(r, "returns")
    n = r.size
    if n < 10:
        raise ValueError("Need at least ~10 observations to fit GARCH(1,1).")

    r = r - r.mean()  # demean: GARCH models variance of the mean-zero residual
    sample_var = float(np.var(r))
    if sample_var <= 0:
        raise ValueError("Zero-variance returns; nothing to fit.")

    if grid is None:
        grid = np.linspace(0.0, 0.98, 50)
    grid = np.asarray(grid, dtype=float)

    best = (np.inf, 0.05, 0.90)  # (nll, alpha, beta) fallback
    for a in grid:
        for b in grid:
            persistence = a + b
            if persistence >= 1.0 or persistence <= 0.0:
                continue
            omega = sample_var * (1.0 - persistence)
            if omega <= 0:
                continue
            nll = _garch_negloglik(r, omega, a, b)
            if nll < best[0]:
                best = (nll, float(a), float(b))

    _, alpha, beta = best
    persistence = alpha + beta
    omega = sample_var * (1.0 - persistence)
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(persistence),
    }


# --------------------------------------------------------------------------- #
# 2b. GJR-GARCH(1,1): asymmetric (leverage) volatility
# --------------------------------------------------------------------------- #
def gjr_garch11_filter(returns: ArrayLike, omega: float, alpha: float,
                       gamma: float, beta: float) -> pd.Series:
    """Conditional volatility for an asymmetric GJR-GARCH(1,1) given params.

    Model (Glosten-Jagannathan-Runkle, zero-mean returns):
        r_t = sqrt(h_t) * z_t,   z_t ~ iid (0, 1)
        h_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2 + beta * h_{t-1}

    The extra term gamma * 1[r_{t-1} < 0] * r_{t-1}^2 lets NEGATIVE shocks raise
    next-period variance more than equal-magnitude positive shocks -- the
    "leverage effect" that empirically dominates equity-index volatility and is
    the single most useful generalization of plain Gaussian GARCH there.
    gamma > 0 is the asymmetry; gamma = 0 collapses exactly to GARCH(1,1).

    Stationarity (assuming a symmetric z, so P(z<0)=1/2) requires
        alpha + gamma/2 + beta < 1,
    because the expected ARCH multiplier is E[alpha + gamma*1[r<0]] = alpha +
    gamma/2. The unconditional variance is
        omega / (1 - alpha - gamma/2 - beta),
    used to seed h_0 (internally consistent with the supplied params). Requires
    omega > 0, alpha >= 0, beta >= 0, and alpha + gamma >= 0 (so the negative-
    shock multiplier is non-negative).

    Returns sqrt(h_t), strictly positive by construction (omega > 0).
    """
    if omega <= 0:
        raise ValueError("omega must be > 0 for positive variance.")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha, beta must be >= 0.")
    if alpha + gamma < 0:
        raise ValueError("alpha + gamma must be >= 0 (non-negative neg-shock "
                         "multiplier).")
    r, idx = _as_array(returns)
    _check_finite(r, "returns")
    n = r.size
    if n == 0:
        return _wrap(np.array([], dtype=float), idx, "gjr_garch_vol")

    eff_persist = alpha + 0.5 * gamma + beta
    if eff_persist < 1.0:
        uncond = omega / (1.0 - eff_persist)
    else:
        uncond = float(np.var(r)) if n > 1 else r[0] ** 2
        uncond = max(uncond, 1e-12)

    h = np.empty(n, dtype=float)
    h[0] = uncond
    for t in range(1, n):
        shock = r[t - 1] ** 2
        lev = gamma if r[t - 1] < 0.0 else 0.0
        h[t] = omega + (alpha + lev) * shock + beta * h[t - 1]
    return _wrap(np.sqrt(h), idx, "gjr_garch_vol")


def _gjr_negloglik(r: np.ndarray, omega: float, alpha: float, gamma: float,
                   beta: float) -> float:
    """Gaussian negative log-likelihood for GJR-GARCH(1,1). Lower is better;
    +inf on degenerate variance paths. Mirrors `_garch_negloglik` so the two
    models are compared on an identical (Gaussian) likelihood basis -- with
    gamma=0 this returns exactly _garch_negloglik(r, omega, alpha, beta)."""
    n = r.size
    eff_persist = alpha + 0.5 * gamma + beta
    if eff_persist < 1.0:
        uncond = omega / (1.0 - eff_persist)
    else:
        uncond = float(np.var(r)) if n > 1 else r[0] ** 2
    if not np.isfinite(uncond) or uncond <= 0:
        return np.inf
    h_prev = uncond
    nll = 0.0
    for t in range(n):
        if t == 0:
            h = uncond
        else:
            shock = r[t - 1] ** 2
            lev = gamma if r[t - 1] < 0.0 else 0.0
            h = omega + (alpha + lev) * shock + beta * h_prev
        if h <= 0 or not np.isfinite(h):
            return np.inf
        nll += 0.5 * (_LOG2PI + np.log(h) + r[t] ** 2 / h)
        h_prev = h
    return nll


def gjr_garch11_fit(returns: ArrayLike,
                    grid: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Estimate GJR-GARCH(1,1) by variance targeting + coarse grid search.

    Same machinery as `garch11_fit`, extended with the asymmetry parameter
    gamma. Variance targeting fixes
        omega = sample_var * (1 - alpha - gamma/2 - beta)
    (matching the GJR unconditional variance under a symmetric innovation), so
    only (alpha, gamma, beta) are searched over the stationarity region
    alpha + gamma/2 + beta < 1 (with alpha + gamma >= 0). The negative log-
    likelihood is the SAME Gaussian objective as GARCH(1,1), so the two fits are
    directly comparable: on equities the GJR optimum almost always has gamma > 0
    and a lower NLL than symmetric GARCH (the leverage effect is real).

    Detect/fix:
    - gamma ~ 0 (or the GJR NLL barely below GARCH's) => little asymmetry in
      this sample; don't pay the extra parameter. gamma < 0 is unusual for
      equities (would mean up-moves raise vol more) -- sanity-check for sign
      conventions or a commodity/currency series where it can legitimately
      differ.
    - The 3-D grid is coarse by design (deterministic, dependency-free). For
      production refine near the optimum or hand to a proper optimizer
      (e.g. the `arch` package's GJR with dist='t').

    Parameters
    ----------
    returns : per-period simple returns.
    grid : optional 1-D grid of candidate values shared by alpha, gamma, beta
        (defaults to np.linspace(0.0, 0.9, 13)). Only triples with
        alpha + gamma/2 + beta < 1 are evaluated.

    Returns
    -------
    dict(omega, alpha, gamma, beta, persistence=alpha+gamma/2+beta, negloglik).
    """
    r, _ = _as_array(returns)
    _check_finite(r, "returns")
    n = r.size
    if n < 10:
        raise ValueError("Need at least ~10 observations to fit GJR-GARCH(1,1).")

    r = r - r.mean()
    sample_var = float(np.var(r))
    if sample_var <= 0:
        raise ValueError("Zero-variance returns; nothing to fit.")

    if grid is None:
        grid = np.linspace(0.0, 0.9, 13)
    grid = np.asarray(grid, dtype=float)

    best = (np.inf, 0.03, 0.05, 0.90)  # (nll, alpha, gamma, beta) fallback
    for a in grid:
        for g in grid:
            for b in grid:
                eff = a + 0.5 * g + b
                if eff >= 1.0 or eff <= 0.0:
                    continue
                omega = sample_var * (1.0 - eff)
                if omega <= 0:
                    continue
                nll = _gjr_negloglik(r, omega, a, g, b)
                if nll < best[0]:
                    best = (nll, float(a), float(g), float(b))

    nll, alpha, gamma, beta = best
    eff = alpha + 0.5 * gamma + beta
    omega = sample_var * (1.0 - eff)
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta": float(beta),
        "persistence": float(eff),
        "negloglik": float(nll),
    }


# --------------------------------------------------------------------------- #
# 2c. Realized variance + HAR-RV forecaster
# --------------------------------------------------------------------------- #
def realized_variance(intraday_returns_by_day: Sequence[ArrayLike],
                      overnight_returns: Optional[ArrayLike] = None,
                      index: Optional[pd.Index] = None) -> pd.Series:
    """Daily realized VARIANCE from intraday LOG returns, RV_t = sum_i r_{t,i}^2.

    Realized variance is a far more accurate proxy for that day's latent
    variance than a single squared daily return r_t^2 (it uses all the intraday
    information), and it is the natural input to HAR-RV (`har_rv`).

    Parameters
    ----------
    intraday_returns_by_day : sequence of length n_days; element d is the array
        of intraday LOG returns within day d (the open-to-close session). Each
        day's RV is the sum of squared intraday log returns.
    overnight_returns : optional length-n_days array of close-to-open LOG
        returns. Intraday RV over the session MISSES the overnight close-to-open
        jump; for single-name equities and anything with scheduled overnight
        events this systematically UNDER-states variance. If supplied, RV_t
        becomes overnight_t^2 + sum_i r_{t,i}^2. (24h markets -- crypto, FX --
        have no gap but do have intraday/weekend seasonality; deseasonalize
        before modeling. This helper does not deseasonalize.)
    index : optional pandas index (length n_days) for the returned series.

    Returns
    -------
    pd.Series of realized VARIANCE per day (variance, not vol). NaN-guarded:
    raises if any intraday/overnight return is non-finite.

    Detect/fix: if RV is implausibly large at the finest sampling, you are
    eating microstructure noise (bid-ask bounce) -- sample coarser (5-min is the
    classic compromise) or use a realized-kernel / two-scale estimator. If your
    RV-based vol is systematically below a daily-return GARCH vol for a gappy
    name, you forgot the overnight term.
    """
    n_days = len(intraday_returns_by_day)
    rv = np.empty(n_days, dtype=float)
    for d in range(n_days):
        ri = np.asarray(intraday_returns_by_day[d], dtype=float)
        _check_finite(ri, f"intraday_returns_by_day[{d}]")
        rv[d] = float(np.sum(ri ** 2))
    if overnight_returns is not None:
        ov, _ = _as_array(overnight_returns)
        _check_finite(ov, "overnight_returns")
        if ov.size != n_days:
            raise ValueError("overnight_returns must have one entry per day.")
        rv = rv + ov ** 2
    return _wrap(rv, index, "realized_variance")


def har_rv(rv: ArrayLike, horizon: int = 1, use_log: bool = False
           ) -> Dict[str, object]:
    """HAR-RV: Corsi (2009) Heterogeneous AutoRegressive realized-variance model.

    A strong, simple daily-vol forecaster and the standard baseline to beat
    GARCH. It regresses realized variance on three trailing averages capturing
    daily / weekly / monthly horizons:

        RV^{(d)}_{t-1} = RV_{t-1}                              (yesterday)
        RV^{(w)}_{t-1} = mean(RV_{t-1}, ..., RV_{t-5})         (trailing week)
        RV^{(m)}_{t-1} = mean(RV_{t-1}, ..., RV_{t-22})        (trailing month)

        RV_t = c + b_d * RV^{(d)}_{t-1} + b_w * RV^{(w)}_{t-1}
                   + b_m * RV^{(m)}_{t-1} + eps_t

    fit by OLS (numpy lstsq). The three lagged aggregates approximate the long
    memory of RV with only ~4 parameters (no fractional differencing). With
    horizon h > 1 the LHS target is the AVERAGE realized variance over the next
    h days, mean(RV_{t}, ..., RV_{t+h-1}); the RHS is unchanged and still ends at
    t-1 -- so the design is strictly trailing for any horizon. (The h-step target
    legitimately uses RV_{t..t+h-1}; it is the *thing being predicted*, never a
    regressor.)

    NO LOOK-AHEAD -- the central discipline of this file. Every regressor for
    target index t is built from RV strictly through t-1. The returned
    `forecast` Series at index t is the one-step (or h-average) prediction that
    a trader could have formed at the close of t-1, so to size day t you can use
    forecast[t] directly (it is already trailing); to earn the return of day t
    you still apply the usual position lag (pnl_t = pos.shift(1) * ret_t). We
    assert internally that no contemporaneous RV_t enters the RHS.

    use_log=True fits the model in log-RV (regress log RV_t on log trailing
    averages) and exponentiates the forecast back to the RV level. Logs tame the
    right-skew/heteroskedasticity of RV and often forecast better; the
    exponentiation introduces a small (here uncorrected) retransformation bias --
    acceptable for ranking/sizing, correct with a smearing factor if you need an
    unbiased level.

    Parameters
    ----------
    rv : realized-variance series (per day), e.g. from `realized_variance`. Must
        be strictly positive when use_log=True.
    horizon : forecast horizon h >= 1. h=1 is the standard one-step forecast.
    use_log : fit in log-RV if True.

    Returns
    -------
    dict with:
        coef       : np.ndarray [c, b_d, b_w, b_m] (in log space if use_log).
        forecast   : pd.Series aligned to `rv`'s index. Entry at t is the
                     trailing one-step / h-average RV forecast for t formed at
                     t-1; the first 22 entries (insufficient lookback) are NaN.
        fitted     : pd.Series of in-sample fitted target (RV level), NaN where
                     either RHS lookback (<22) or LHS horizon window is missing.
        r2         : in-sample R^2 on the rows used for fitting (level space).
        n_obs      : number of rows used in the OLS fit.

    Detect/fix: if b_d + b_w + b_m >= 1 the implied process is near-explosive
    (RV barely mean-reverts) -- usually outliers/jumps; consider HAR-RV-J
    (separate a jump component) or winsorize. If `forecast` ever beats a static
    forecast by an implausible margin, check you did not accidentally feed a
    contemporaneous RV (the assertion below guards the canonical path).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    x, idx = _as_array(rv)
    _check_finite(x, "rv")
    n = x.size
    if use_log:
        if np.any(x <= 0):
            raise ValueError("use_log=True requires strictly positive RV.")
        series = np.log(x)
    else:
        series = x

    LAG_MAX = 22  # monthly window -> first usable target index is 22
    # Build the trailing design. Row t uses series[t-1] (daily), the mean of
    # series[t-5 .. t-1] (weekly) and series[t-22 .. t-1] (monthly). ALL strictly
    # < t, so the contemporaneous series[t] never appears on the RHS.
    daily = np.full(n, np.nan)
    weekly = np.full(n, np.nan)
    monthly = np.full(n, np.nan)
    for t in range(LAG_MAX, n):
        daily[t] = series[t - 1]
        weekly[t] = series[t - 5:t].mean()      # 5 obs: t-5 .. t-1
        monthly[t] = series[t - 22:t].mean()    # 22 obs: t-22 .. t-1

    # --- explicit no-look-ahead assertion (mirrors the file's lag discipline) ---
    # Verify each RHS term equals a value computed from a window that ENDS at
    # t-1; i.e. the RHS does not read series[t] (or anything later).
    for t in range(LAG_MAX, n):
        assert daily[t] == series[t - 1], "daily lag leaked contemporaneous RV"
        assert np.isclose(weekly[t], series[t - 5:t].mean())
        assert np.isclose(monthly[t], series[t - 22:t].mean())

    # h-step target: mean of series over [t, t+h-1] (in the modeled space).
    target = np.full(n, np.nan)
    for t in range(n):
        if t + horizon <= n:
            target[t] = series[t:t + horizon].mean()

    # rows usable for FITTING: need full RHS lookback (t >= LAG_MAX) and a full
    # target window (t + horizon <= n).
    fit_mask = np.zeros(n, dtype=bool)
    for t in range(LAG_MAX, n):
        if t + horizon <= n:
            fit_mask[t] = True
    if fit_mask.sum() < 5:
        raise ValueError("Not enough observations after the 22-day lookback "
                         "and horizon window to fit HAR-RV.")

    rows = np.where(fit_mask)[0]
    X = np.column_stack([np.ones(rows.size), daily[rows], weekly[rows],
                         monthly[rows]])
    y = target[rows]
    coef, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)

    # in-sample fit + R^2, reported in LEVEL space for interpretability
    yhat_modeled = X @ coef
    if use_log:
        yhat_level = np.exp(yhat_modeled)
        y_level = np.exp(y)
    else:
        yhat_level = yhat_modeled
        y_level = y
    ss_res = float(np.sum((y_level - yhat_level) ** 2))
    ss_tot = float(np.sum((y_level - y_level.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fitted = np.full(n, np.nan)
    fitted[rows] = yhat_level

    # forecast at EVERY t with a full RHS lookback (t >= LAG_MAX), including the
    # tail rows whose target window is incomplete -- those are exactly the live
    # forecasts you would trade on. RHS is strictly trailing, so this is causal.
    forecast = np.full(n, np.nan)
    for t in range(LAG_MAX, n):
        rhs = np.array([1.0, daily[t], weekly[t], monthly[t]])
        pred_modeled = float(rhs @ coef)
        forecast[t] = np.exp(pred_modeled) if use_log else pred_modeled

    return {
        "coef": coef,
        "forecast": _wrap(forecast, idx, "har_rv_forecast"),
        "fitted": _wrap(fitted, idx, "har_rv_fitted"),
        "r2": float(r2),
        "n_obs": int(rows.size),
    }


# --------------------------------------------------------------------------- #
# 3. Kalman filter: local-level (random walk + noise)
# --------------------------------------------------------------------------- #
def kalman_local_level(obs: ArrayLike, q: float, r: float) -> pd.Series:
    """Filtered state for the local-level (random-walk-plus-noise) model.

    State space:
        x_t = x_{t-1} + w_t,   w_t ~ N(0, q)     (process / level drift)
        y_t = x_t     + v_t,   v_t ~ N(0, r)     (observation noise)

    The steady-state gain depends only on the signal-to-noise ratio q/r: large
    q/r -> the filter tracks observations quickly (responsive, noisy); small
    q/r -> heavy smoothing toward a slow-moving level. This is the canonical
    adaptive-mean / trend-extraction filter for prices and spreads.

    Returns the filtered mean x_{t|t} (uses data through t -- causal). For a
    trading signal, remember it is causal but still must be lagged vs the return
    it predicts.

    Detect/fix: if the filtered state is glued to the raw series, q/r is too
    high; if it lags badly through real level shifts, q/r is too low. A
    near-flat output through an obvious regime change means q ~ 0.
    """
    if q < 0 or r <= 0:
        raise ValueError("q must be >= 0 and r must be > 0.")
    y, idx = _as_array(obs)
    _check_finite(y, "obs")
    n = y.size
    if n == 0:
        return _wrap(np.array([], dtype=float), idx, "kalman_level")

    x = np.empty(n, dtype=float)
    x_hat = y[0]            # initialize state at first observation
    p = r                  # initial state variance ~ obs noise
    for t in range(n):
        # predict
        x_prior = x_hat
        p_prior = p + q
        # update
        k = p_prior / (p_prior + r)           # Kalman gain
        x_hat = x_prior + k * (y[t] - x_prior)
        p = (1.0 - k) * p_prior
        x[t] = x_hat
    return _wrap(x, idx, "kalman_level")


# --------------------------------------------------------------------------- #
# 4. Kalman filter: dynamic (time-varying) regression beta
# --------------------------------------------------------------------------- #
def kalman_dynamic_beta(y: ArrayLike, x: ArrayLike, q: float,
                        r: float) -> pd.Series:
    """Time-varying regression coefficient beta_t via a state-space regression.

    Model (single regressor, no intercept -- demean inputs if you need one):
        beta_t = beta_{t-1} + w_t,   w_t ~ N(0, q)      (random-walk coefficient)
        y_t    = x_t * beta_t + v_t, v_t ~ N(0, r)      (observation)

    This is the workhorse for dynamic hedge ratios (pairs/stat-arb), rolling
    factor loadings, and adaptive market beta -- a principled alternative to a
    fixed rolling-window OLS beta. q controls how fast beta is allowed to drift;
    r is the measurement-noise scale.

    Returns the filtered beta_{t|t} (causal). For a hedge ratio, lag it before
    forming next period's spread/position to avoid look-ahead.

    Detect/fix: a beta that jumps to absurd magnitudes on small |x_t| means r is
    too small relative to q (the filter over-trusts noisy obs); shrink q or raise
    r. To add an intercept, stack x with a column of ones and generalize to the
    2-state vector form.
    """
    if q < 0 or r <= 0:
        raise ValueError("q must be >= 0 and r must be > 0.")
    yv, idx = _as_array(y)
    xv, _ = _as_array(x)
    _check_finite(yv, "y")
    _check_finite(xv, "x")
    if yv.size != xv.size:
        raise ValueError("y and x must have the same length.")
    n = yv.size
    if n == 0:
        return _wrap(np.array([], dtype=float), idx, "kalman_beta")

    beta = np.empty(n, dtype=float)
    b_hat = 0.0
    p = 1.0e6  # diffuse prior: we know little about beta initially
    for t in range(n):
        # predict
        b_prior = b_hat
        p_prior = p + q
        # update (scalar obs with time-varying "design" x_t)
        xt = xv[t]
        s = xt * p_prior * xt + r           # innovation variance
        k = p_prior * xt / s                # Kalman gain
        innov = yv[t] - xt * b_prior
        b_hat = b_prior + k * innov
        p = (1.0 - k * xt) * p_prior
        beta[t] = b_hat
    return _wrap(beta, idx, "kalman_beta")


# --------------------------------------------------------------------------- #
# 5. 2-state Gaussian HMM (Baum-Welch EM + Viterbi)
# --------------------------------------------------------------------------- #
def _gauss_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def hmm_gaussian_2state(returns: ArrayLike, n_iter: int = 50,
                        seed: int = 0) -> Dict[str, object]:
    """Fit a 2-state Gaussian HMM by Baum-Welch (EM); decode with Viterbi.

    Emissions: in state k, r_t ~ N(mean_k, var_k). Latent state follows a
    first-order Markov chain with 2x2 transition matrix A (A[i, j] = P(s_t=j |
    s_{t-1}=i)). The classic application is a "calm vs turbulent" volatility
    regime: one state with low variance, one with high variance.

    Implementation notes:
    - Scaled forward-backward (per-time normalization) for numerical stability
      instead of full log-space, which is adequate for the 2-state daily case.
    - Means/vars initialized by splitting the sample at its median (a robust,
      deterministic warm start that reliably separates a vol regime); `seed`
      perturbs the split only enough to break exact ties.
    - States are returned via Viterbi (most-likely PATH), which respects the
      transition dynamics rather than independently arg-maxing each posterior.

    Pitfalls / detect-fix:
    - LABEL SWITCHING: state 0 vs 1 is arbitrary. Here we sort states so index 0
      is the LOW-variance regime and index 1 the HIGH-variance regime; rely on
      that ordering, not on raw EM labels.
    - LOOK-AHEAD: this fit uses the WHOLE sample (smoothed parameters). For a
      backtest you must re-fit on an expanding window and decode only up to t, or
      you leak future information into the regime label. Treat the in-sample
      `states` here as research/diagnostic, not a tradable signal as-is.
    - Variance collapse: a state can grab a single point and drive var -> 0; we
      floor variances to avoid singular likelihoods.

    Returns
    -------
    dict with:
        means       : np.ndarray shape (2,)  -- [low-var mean, high-var mean]
        variances   : np.ndarray shape (2,)  -- sorted ascending (var0 < var1)
        transition  : np.ndarray shape (2,2) -- rows sum to 1, reordered to match
        states      : pd.Series of Viterbi state labels in {0, 1}
        loglik      : final scaled log-likelihood (float)
    """
    rv, idx = _as_array(returns)
    _check_finite(rv, "returns")
    n = rv.size
    if n < 4:
        raise ValueError("Need at least 4 observations for a 2-state HMM.")

    rng = np.random.default_rng(seed)
    # deterministic, robust init: split at median into low/high groups
    med = np.median(rv)
    lo = rv[rv <= med]
    hi = rv[rv > med]
    if lo.size == 0 or hi.size == 0:  # degenerate split
        lo, hi = rv[: n // 2], rv[n // 2:]
    jitter = 1e-6 * rng.standard_normal(2)
    means = np.array([lo.mean(), hi.mean()], dtype=float) + jitter
    variances = np.array([max(lo.var(), 1e-8), max(hi.var(), 1e-8)], dtype=float)
    # init so state 0 = low variance, state 1 = high variance
    if variances[0] > variances[1]:
        means = means[::-1].copy()
        variances = variances[::-1].copy()
    pi = np.array([0.5, 0.5], dtype=float)
    A = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)

    K = 2
    loglik = -np.inf
    for _ in range(n_iter):
        # ---- E-step: scaled forward-backward ----
        B = np.column_stack([_gauss_pdf(rv, means[k], variances[k])
                             for k in range(K)])  # (n, K) emission likelihoods
        B = np.maximum(B, 1e-300)

        alpha = np.zeros((n, K))
        c = np.zeros(n)  # scaling factors
        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum()
        if c[0] <= 0:
            break
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum()
            if c[t] <= 0:
                c[t] = 1e-300
            alpha[t] /= c[t]

        beta = np.zeros((n, K))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1.0
        gamma /= gamma_sum  # (n, K) posterior P(s_t=k | data)

        # xi: pairwise posteriors summed over t -> expected transition counts
        xi_sum = np.zeros((K, K))
        for t in range(n - 1):
            denom = c[t + 1]
            num = (alpha[t][:, None] * A *
                   (B[t + 1] * beta[t + 1])[None, :]) / denom
            xi_sum += num

        new_loglik = float(np.sum(np.log(c)))

        # ---- M-step ----
        pi = gamma[0] / gamma[0].sum()
        row = xi_sum.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        A = xi_sum / row
        gsum = gamma.sum(axis=0)
        gsum[gsum == 0] = 1e-300
        means = (gamma * rv[:, None]).sum(axis=0) / gsum
        variances = (gamma * (rv[:, None] - means[None, :]) ** 2).sum(axis=0) / gsum
        variances = np.maximum(variances, 1e-10)

        if np.isfinite(new_loglik) and abs(new_loglik - loglik) < 1e-8:
            loglik = new_loglik
            break
        loglik = new_loglik

    # ---- canonical ordering: state 0 = low variance, 1 = high variance ----
    order = np.argsort(variances)
    means = means[order]
    variances = variances[order]
    pi = pi[order]
    A = A[np.ix_(order, order)]
    A = A / A.sum(axis=1, keepdims=True)

    # ---- Viterbi decode (in log space) with reordered params ----
    B = np.column_stack([_gauss_pdf(rv, means[k], variances[k])
                         for k in range(K)])
    B = np.maximum(B, 1e-300)
    logB = np.log(B)
    logA = np.log(np.maximum(A, 1e-300))
    logpi = np.log(np.maximum(pi, 1e-300))

    delta = np.zeros((n, K))
    psi = np.zeros((n, K), dtype=int)
    delta[0] = logpi + logB[0]
    for t in range(1, n):
        for k in range(K):
            seq = delta[t - 1] + logA[:, k]
            psi[t, k] = int(np.argmax(seq))
            delta[t, k] = seq[psi[t, k]] + logB[t, k]
    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return {
        "means": means,
        "variances": variances,
        "transition": A,
        "states": _wrap(states.astype(int), idx, "hmm_state"),
        "loglik": float(loglik),
    }


# --------------------------------------------------------------------------- #
# 6. CUSUM change-point detection
# --------------------------------------------------------------------------- #
def cusum_changepoints(series: ArrayLike, threshold: float,
                       drift: float = 0.0) -> List[int]:
    """Two-sided CUSUM change-point detector for shifts in the level of a series.

    Tabular CUSUM on e_t = x_t - mu, where mu is the IN-CONTROL mean estimated
    from the current segment (re-estimated after each detection), NOT the full-
    sample mean (centering on mean(x) makes both sides of a shift look like
    deviations and fires spuriously at t=0):
        g_pos_t = max(0, g_pos_{t-1} + e_t - drift)
        g_neg_t = max(0, g_neg_{t-1} - e_t - drift)
    A change is flagged when either accumulator exceeds `threshold`; both
    accumulators reset to 0 after a detection so multiple shifts can be found.

    `drift` is the slack / allowance (often k = half the smallest shift you care
    to detect, in the same units as the series): larger drift -> fewer false
    alarms but slower detection. Choose `threshold` for the desired
    sensitivity / average-run-length tradeoff.

    Use for detecting regime breaks in a spread mean, rolling Sharpe, factor
    return, or realized-vol series. It detects shifts AFTER they accumulate, so
    the reported index lags the true change by roughly threshold / shift_size
    bars -- account for this latency, do not treat the index as the exact break.

    Returns a list of 0-based indices where a change was detected (possibly
    empty).
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0.")
    if drift < 0:
        raise ValueError("drift (slack) must be >= 0.")
    x, _ = _as_array(series)
    _check_finite(x, "series")
    n = x.size
    if n == 0:
        return []

    warmup = max(10, n // 20)
    g_pos = 0.0
    g_neg = 0.0
    seg_sum = 0.0
    seg_count = 0
    changes: List[int] = []
    for t in range(n):
        seg_sum += float(x[t])
        seg_count += 1
        mu = seg_sum / seg_count          # in-control mean of current segment
        if seg_count <= warmup:
            continue                      # establish baseline before testing
        e = x[t] - mu
        g_pos = max(0.0, g_pos + e - drift)
        g_neg = max(0.0, g_neg - e - drift)
        if g_pos > threshold or g_neg > threshold:
            changes.append(t)
            g_pos = 0.0
            g_neg = 0.0
            seg_sum = 0.0
            seg_count = 0             # restart baseline for the new regime
    return changes


# --------------------------------------------------------------------------- #
# 7. Volatility targeting
# --------------------------------------------------------------------------- #
def vol_target_scale(forecast_vol: ArrayLike, target_vol: float,
                     max_leverage: float = 3.0) -> Union[pd.Series, float]:
    """Position scaler for volatility targeting: scale = target_vol / forecast_vol.

    Sizes a position so its EX-ANTE volatility equals `target_vol`: when forecast
    vol doubles, exposure halves (the inverse relationship that produces the
    well-known vol-targeting risk smoothing). The scaler is capped at
    `max_leverage` to avoid blowing up when forecast vol -> 0.

    IMPORTANT: `forecast_vol` and `target_vol` must be on the SAME horizon/units
    (both per-period, or both annualized). Mixing a daily forecast with an
    annual target is a frequent, silent sizing bug -- it scales positions by
    ~sqrt(252). The forecast must also be CAUSAL (e.g. ewma_vol/garch filtered
    through t, or har_rv's trailing forecast) and the resulting position lagged
    before earning returns.

    Parameters
    ----------
    forecast_vol : scalar or series of forecast per-period volatility (> 0).
    target_vol   : desired per-period volatility (same units), > 0.
    max_leverage : cap on the absolute scaler (default 3.0).

    Returns
    -------
    Same shape as `forecast_vol` (float if scalar in, Series if series in),
    clipped to [0, max_leverage].
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be > 0.")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be > 0.")

    scalar_in = np.isscalar(forecast_vol)
    fv, idx = _as_array(forecast_vol)
    # guard against divide-by-zero; tiny floor -> scaler hits the cap, not inf
    safe = np.where(fv > 0, fv, 1e-300)  # zero/degenerate vol -> hits the cap (vol->0+ limit), not 0
    scale = np.clip(target_vol / safe, 0.0, max_leverage)
    if scalar_in:
        return float(scale.reshape(-1)[0])
    return _wrap(scale, idx, "vol_scale")


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def _test_ewma_vol() -> None:
    rng = np.random.default_rng(0)
    low = rng.normal(0.0, 0.005, 300)   # calm regime
    high = rng.normal(0.0, 0.05, 300)   # turbulent regime
    r = pd.Series(np.concatenate([low, high]))
    v = ewma_vol(r, lam=0.94)
    assert len(v) == len(r)
    assert np.all(np.isfinite(v.values))
    early = v.iloc[250:300].mean()      # end of calm period
    late = v.iloc[550:600].mean()       # well into turbulent period
    assert late > early, (early, late)
    assert late > 3 * early, "EWMA vol should respond strongly to a 10x jump"


def _test_garch_filter() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, 500)
    v = garch11_filter(r, omega=1e-6, alpha=0.08, beta=0.90)
    assert np.all(v.values > 0), "GARCH conditional vol must be strictly positive"
    assert np.all(np.isfinite(v.values))


def _simulate_garch(n: int, omega: float, alpha: float, beta: float,
                    seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 500)  # burn-in
    h = omega / (1.0 - alpha - beta)
    r = np.empty(n + 500)
    for t in range(n + 500):
        r[t] = np.sqrt(h) * z[t]
        h = omega + alpha * r[t] ** 2 + beta * h
    return r[500:]


def _test_garch_fit() -> None:
    true_omega, true_alpha, true_beta = 2e-6, 0.10, 0.85
    r = _simulate_garch(5000, true_omega, true_alpha, true_beta, seed=7)
    fit = garch11_fit(r)
    true_persist = true_alpha + true_beta
    assert abs(fit["persistence"] - true_persist) < 0.15, fit
    # omega order-of-magnitude check
    assert 1e-7 < fit["omega"] < 1e-4, fit["omega"]
    assert 0.0 <= fit["alpha"] <= 1.0 and 0.0 <= fit["beta"] <= 1.0
    assert fit["persistence"] < 1.0


def _simulate_gjr(n: int, omega: float, alpha: float, gamma: float,
                  beta: float, seed: int) -> np.ndarray:
    """Simulate GJR-GARCH(1,1) with a known leverage effect (gamma)."""
    rng = np.random.default_rng(seed)
    burn = 1000
    z = rng.standard_normal(n + burn)
    eff = alpha + 0.5 * gamma + beta
    h = omega / (1.0 - eff)
    r = np.empty(n + burn)
    r_prev = 0.0
    h_prev = h
    for t in range(n + burn):
        lev = gamma if r_prev < 0.0 else 0.0
        h_t = omega + (alpha + lev) * r_prev ** 2 + beta * h_prev
        r[t] = np.sqrt(h_t) * z[t]
        r_prev = r[t]
        h_prev = h_t
    return r[burn:]


def _test_gjr_filter() -> None:
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 0.01, 500)
    v = gjr_garch11_filter(r, omega=1e-6, alpha=0.03, gamma=0.08, beta=0.88)
    assert np.all(v.values > 0), "GJR vol must be strictly positive"
    assert np.all(np.isfinite(v.values))
    # gamma=0 must collapse EXACTLY to symmetric GARCH(1,1)
    vg = gjr_garch11_filter(r, omega=1e-6, alpha=0.08, gamma=0.0, beta=0.90)
    vs = garch11_filter(r, omega=1e-6, alpha=0.08, beta=0.90)
    assert np.allclose(vg.values, vs.values), "gamma=0 should equal GARCH(1,1)"
    # negloglik parity at gamma=0
    nll_gjr0 = _gjr_negloglik(r, 1e-6, 0.08, 0.0, 0.90)
    nll_g = _garch_negloglik(r, 1e-6, 0.08, 0.90)
    assert np.isclose(nll_gjr0, nll_g), (nll_gjr0, nll_g)


def _test_gjr_fit_recovers_leverage() -> None:
    # data WITH a leverage effect: gamma_true > 0
    true = dict(omega=3e-6, alpha=0.02, gamma=0.12, beta=0.85)
    r = _simulate_gjr(2500, seed=23, **true)
    gjr = gjr_garch11_fit(r)
    # fitted asymmetry is positive (recovers the leverage effect)
    assert gjr["gamma"] > 0.02, gjr
    # effective persistence recovered to the right neighborhood
    true_eff = true["alpha"] + 0.5 * true["gamma"] + true["beta"]
    assert abs(gjr["persistence"] - true_eff) < 0.12, (gjr["persistence"], true_eff)
    # GJR beats symmetric GARCH on the SAME Gaussian likelihood when leverage
    # is present (lower negloglik). Compare like-for-like via the GJR nll fn
    # evaluated at the symmetric-GARCH optimum (gamma=0).
    g = garch11_fit(r)
    g_nll = _gjr_negloglik(r - r.mean(), g["omega"], g["alpha"], 0.0, g["beta"])
    assert gjr["negloglik"] < g_nll, (gjr["negloglik"], g_nll)


def _simulate_har(n: int, c: float, b_d: float, b_w: float, b_m: float,
                  noise_sd: float, seed: int) -> np.ndarray:
    """Simulate an RV series from a known HAR process (in RV level):
    RV_t = c + b_d*RV_{t-1} + b_w*mean(RV_{t-5..t-1}) + b_m*mean(RV_{t-22..t-1})
           + eps_t, with small additive noise and a positive floor so RV stays
    positive. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    burn = 200
    N = n + burn
    uncond = c / max(1e-9, (1.0 - b_d - b_w - b_m))
    rv = np.full(N, uncond)  # start at the unconditional mean
    for t in range(22, N):
        daily = rv[t - 1]
        weekly = rv[t - 5:t].mean()
        monthly = rv[t - 22:t].mean()
        eps = noise_sd * rng.standard_normal()
        rv[t] = c + b_d * daily + b_w * weekly + b_m * monthly + eps
        if rv[t] <= 1e-12:
            rv[t] = 1e-12  # keep variance positive
    return rv[burn:]


def _test_har_recovers_coefficients() -> None:
    true = dict(c=2e-6, b_d=0.35, b_w=0.35, b_m=0.20, noise_sd=2e-6)
    rv = _simulate_har(4000, seed=3, **true)
    res = har_rv(rv, horizon=1, use_log=False)
    c, bd, bw, bm = res["coef"]
    assert abs(bd - true["b_d"]) < 0.12, (bd, true["b_d"])
    assert abs(bw - true["b_w"]) < 0.15, (bw, true["b_w"])
    assert abs(bm - true["b_m"]) < 0.15, (bm, true["b_m"])
    assert res["r2"] > 0.5, res["r2"]


def _test_har_beats_static_baseline() -> None:
    true = dict(c=2e-6, b_d=0.4, b_w=0.3, b_m=0.2, noise_sd=2e-6)
    rv = _simulate_har(3000, seed=9, **true)
    res = har_rv(rv, horizon=1, use_log=False)
    fc = res["forecast"].to_numpy()
    target = rv.copy()  # one-step target at index t is RV_t itself
    valid = np.where(np.isfinite(fc))[0]
    # static baseline: trailing (expanding) mean RV through t-1 -- strictly
    # causal, the same information set the HAR forecast had.
    static = np.array([rv[:t].mean() for t in valid])
    mse_har = np.mean((target[valid] - fc[valid]) ** 2)
    mse_static = np.mean((target[valid] - static) ** 2)
    assert mse_har < mse_static, (mse_har, mse_static)


def _test_har_no_lookahead() -> None:
    """Lag unit test mirroring the vol_target leakage discipline: the forecast
    at index t must NOT change when RV_t (and later) are arbitrarily corrupted.
    If the RHS ever read a contemporaneous/future RV, perturbing it would move
    the forecast at t."""
    true = dict(c=2e-6, b_d=0.4, b_w=0.3, b_m=0.2, noise_sd=2e-6)
    rv = _simulate_har(800, seed=5, **true)
    res_full = har_rv(rv, horizon=1, use_log=False)
    fc_full = res_full["forecast"].to_numpy()
    coef = res_full["coef"]

    t0 = 400  # interior index with a valid forecast
    assert np.isfinite(fc_full[t0])
    # Recompute the forecast at t0 from a COPY whose values at indices >= t0 are
    # destroyed. A causal/trailing RHS uses only RV[t0-22 .. t0-1], so with the
    # same fitted coefficients the forecast at t0 must be IDENTICAL.
    rv_corrupt = rv.copy()
    rv_corrupt[t0:] = 1e6  # garbage future
    daily = rv_corrupt[t0 - 1]
    weekly = rv_corrupt[t0 - 5:t0].mean()
    monthly = rv_corrupt[t0 - 22:t0].mean()
    rhs = np.array([1.0, daily, weekly, monthly])
    fc_recomputed = float(rhs @ coef)
    assert np.isclose(fc_recomputed, fc_full[t0]), (
        "forecast at t0 depends on RV_t0 or later -> look-ahead leak")
    # first 22 entries lack a full lookback -> must be NaN; the rest finite.
    assert np.all(np.isnan(fc_full[:22]))
    assert np.all(np.isfinite(fc_full[22:]))


def _test_har_log_space() -> None:
    true = dict(c=1e-6, b_d=0.4, b_w=0.3, b_m=0.2, noise_sd=1e-6)
    rv = _simulate_har(2000, seed=15, **true)
    res = har_rv(rv, horizon=1, use_log=True)
    fc = res["forecast"].to_numpy()
    valid = np.isfinite(fc)
    assert np.all(fc[valid] > 0), "log-space forecast must exponentiate to RV>0"
    assert res["r2"] > 0.3, res["r2"]
    # multi-step (h=5) log forecast: still causal, still positive.
    res5 = har_rv(rv, horizon=5, use_log=True)
    fc5 = res5["forecast"].to_numpy()
    assert np.all(fc5[np.isfinite(fc5)] > 0)


def _test_realized_variance() -> None:
    rng = np.random.default_rng(42)
    n_days = 50
    # 78 5-min log returns per session (6.5h), sd 0.0008 per bar
    intraday = [rng.normal(0.0, 0.0008, 78) for _ in range(n_days)]
    rv = realized_variance(intraday)
    assert len(rv) == n_days
    assert np.all(rv.values > 0)
    # RV ~ n_bars * sigma_bar^2 (law of large numbers over the day)
    expected = 78 * 0.0008 ** 2
    assert abs(rv.mean() - expected) < 0.3 * expected, (rv.mean(), expected)
    # overnight term adds EXACTLY ov^2 and strictly increases mean RV
    overnight = rng.normal(0.0, 0.005, n_days)
    rv_on = realized_variance(intraday, overnight_returns=overnight)
    assert np.allclose((rv_on.values - rv.values), overnight ** 2)
    assert rv_on.mean() > rv.mean(), "overnight gap must add variance"
    # feeding RV into HAR runs end-to-end and yields positive forecasts
    long_intraday = [rng.normal(0.0, 0.0008, 78) for _ in range(300)]
    rv_long = realized_variance(long_intraday)
    res = har_rv(rv_long, horizon=1)
    fc = res["forecast"].to_numpy()
    assert np.all(fc[np.isfinite(fc)] > 0)


def _test_kalman_local_level() -> None:
    rng = np.random.default_rng(2)
    true_c = 5.0
    y = pd.Series(true_c + rng.normal(0.0, 1.0, 400))
    x = kalman_local_level(y, q=1e-4, r=1.0)
    assert len(x) == len(y)
    last = x.iloc[-1]
    assert abs(last - true_c) < 0.3, last
    assert abs(last - y.mean()) < 0.3, (last, y.mean())


def _test_kalman_dynamic_beta() -> None:
    rng = np.random.default_rng(3)
    n = 600
    x = rng.normal(0.0, 1.0, n)
    beta_true = np.concatenate([np.full(n // 2, 0.5), np.full(n - n // 2, 2.0)])
    y = beta_true * x + rng.normal(0.0, 0.05, n)
    b = kalman_dynamic_beta(y, x, q=1e-3, r=0.05 ** 2)
    assert abs(b.iloc[150] - 0.5) < 0.3, b.iloc[150]   # first regime
    assert abs(b.iloc[-1] - 2.0) < 0.3, b.iloc[-1]     # second regime
    assert b.iloc[-1] > b.iloc[150], "beta should rise across the break"


def _test_hmm() -> None:
    rng = np.random.default_rng(4)
    calm = rng.normal(0.0, 0.005, 400)     # low-variance regime
    storm = rng.normal(0.0, 0.03, 400)     # high-variance regime
    r = pd.Series(np.concatenate([calm, storm]))
    res = hmm_gaussian_2state(r, n_iter=60, seed=0)
    var0, var1 = res["variances"]
    assert var1 / var0 > 2.0, (var0, var1)   # distinct variances, sorted asc
    states = res["states"].to_numpy()
    truth = np.concatenate([np.zeros(400, dtype=int), np.ones(400, dtype=int)])
    acc = np.mean(states == truth)
    assert acc > 0.80, f"regime classification accuracy {acc:.3f} too low"
    # transition matrix rows are valid distributions
    A = res["transition"]
    assert np.allclose(A.sum(axis=1), 1.0)


def _test_cusum() -> None:
    rng = np.random.default_rng(5)
    n1, n2 = 200, 200
    pre = rng.normal(0.0, 1.0, n1)
    post = rng.normal(5.0, 1.0, n2)   # clear +5 mean shift at index 200
    s = pd.Series(np.concatenate([pre, post]))
    cps = cusum_changepoints(s, threshold=10.0, drift=0.5)  # threshold >> noise so the pre-period does not false-alarm
    assert len(cps) >= 1, "should detect the injected shift"
    first = cps[0]
    assert 200 <= first <= 245, f"detection at {first}, expected just after 200"


def _test_vol_target_scale() -> None:
    # scalar: doubling forecast vol halves exposure
    s1 = vol_target_scale(0.10, target_vol=0.10)
    s2 = vol_target_scale(0.20, target_vol=0.10)
    assert abs(s1 - 1.0) < 1e-12, s1
    assert abs(s2 - 0.5) < 1e-12, s2
    assert abs(s2 - s1 / 2.0) < 1e-12
    # cap is enforced when forecast vol is tiny / zero
    s_cap = vol_target_scale(0.001, target_vol=0.10, max_leverage=3.0)
    assert abs(s_cap - 3.0) < 1e-12, s_cap
    s_zero = vol_target_scale(0.0, target_vol=0.10, max_leverage=3.0)
    assert abs(s_zero - 3.0) < 1e-12, s_zero
    # series path preserves index and clipping
    fv = pd.Series([0.10, 0.20, 0.05], index=["a", "b", "c"])
    sc = vol_target_scale(fv, target_vol=0.10)
    assert isinstance(sc, pd.Series)
    assert list(sc.index) == ["a", "b", "c"]
    assert abs(sc["b"] - 0.5) < 1e-12


def _run_all_tests() -> None:
    _test_ewma_vol()
    _test_garch_filter()
    _test_garch_fit()
    _test_gjr_filter()
    _test_gjr_fit_recovers_leverage()
    _test_realized_variance()
    _test_har_recovers_coefficients()
    _test_har_beats_static_baseline()
    _test_har_no_lookahead()
    _test_har_log_space()
    _test_kalman_local_level()
    _test_kalman_dynamic_beta()
    _test_hmm()
    _test_cusum()
    _test_vol_target_scale()
    print("regime.py: all self-tests passed.")


if __name__ == "__main__":
    _run_all_tests()
