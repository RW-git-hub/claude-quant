"""regime.py - Time-series & regime-modeling toolkit for quant research.

Pure numpy / pandas / standard library. No scipy, no statsmodels: the GARCH
fit, Kalman filters, HMM (Baum-Welch + Viterbi) and CUSUM are implemented by
hand so the file is dependency-light and auditable. `statistics.NormalDist`
supplies the Gaussian pdf/cdf where needed.

Conventions
-----------
- `returns` are SIMPLE per-period returns (decimal, e.g. 0.01 == 1%). Volatility
  helpers operate on the return level directly; demean only where stated. For
  daily data the natural annualization of a vol is `vol * sqrt(252)` (not done
  here -- these functions return per-period vol so the caller controls ppy).
- Volatility = standard deviation (sqrt of variance), per period.
- All filters are CAUSAL: the value at index t uses information up to and
  including t (filtered, not smoothed). To use a regime/vol signal as a position
  you must still lag it vs the return it earns (pnl_t = pos.shift(1) * ret_t),
  exactly as elsewhere in this skill. These functions do NOT lag for you.

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
    mean usually hurts). lam=0.94 is the RiskMetrics daily default (~ a 33-day
    half-life); use ~0.97 for monthly.

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
    through t) and the resulting position lagged before earning returns.

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
    _test_kalman_local_level()
    _test_kalman_dynamic_beta()
    _test_hmm()
    _test_cusum()
    _test_vol_target_scale()
    print("regime.py: all self-tests passed.")


if __name__ == "__main__":
    _run_all_tests()
