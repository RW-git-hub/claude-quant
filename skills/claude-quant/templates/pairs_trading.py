"""pairs_trading.py - Cointegration / pairs-trading toolkit (numpy/pandas/stdlib only).

Statistical-arbitrage relative-value primitives: estimate a hedge ratio, build a
spread, test it for stationarity (Engle-Granger two-step with a self-implemented
Augmented Dickey-Fuller statistic), measure mean-reversion half-life, z-score the
spread, and generate stateful entry/exit/stop signals.

Conventions
-----------
- Returns/PnL: positions are lagged vs the returns they earn (pnl_t = pos.shift(1)*ret_t).
  This file produces a position series ON THE SPREAD; turning it into asset-leg
  positions (long y / short beta*x for pos=+1) and PnL is the caller's job, and the
  same shift(1) discipline applies there.
- Sign of the spread position: pos=+1 means LONG the spread (long y, short beta*x),
  taken when the spread is cheap (z <= -entry). pos=-1 is SHORT the spread.
- ADF: we regress delta s_t on [const, s_{t-1}, delta s_{t-1..t-lags}] and return the
  t-stat on the s_{t-1} coefficient. More negative => stronger rejection of a unit
  root => more stationary. Compare to MacKinnon critical values, NOT the standard
  normal:
    * Plain ADF with a constant (testing a raw series): ~ -2.86 at 5%.
    * Engle-Granger RESIDUAL test (the spread came from an estimated regression):
      use the more negative EG critical value ~ -3.34 at 5% for one cointegrating
      regressor, because the residual was constructed to look stationary (the
      pre-estimation of beta biases the naive ADF). Use EG values for engle_granger().
  These are asymptotic; for small samples and >1 regressor consult the full tables.

DETECT / FIX
------------
- DETECT: you ran ADF on the SPREAD using the -2.86 cutoff and declared cointegration.
  FIX: the spread used an ESTIMATED beta -> use the Engle-Granger residual critical
  value (~ -3.34), which is stricter, or you will over-accept spurious pairs.
- DETECT: half-life is negative or huge. FIX: phi >= 1 means no mean reversion
  (we return +inf); a negative half-life means phi < 0 (period-2 oscillation) and is
  not a tradable slow reversion -- treat as a reject.
- DETECT: in-sample beta used to trade out-of-sample. FIX: re-estimate beta on a
  training window only (purge+embargo for any CV), or use kalman_hedge_ratio for a
  time-varying estimate updated point-in-time.

No scipy / statsmodels: the ADF and OLS are done with numpy least squares;
statistics.NormalDist is allowed but unused here (kept available for callers).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "hedge_ratio",
    "spread",
    "adf_tstat",
    "engle_granger",
    "half_life",
    "zscore",
    "generate_signals",
    "kalman_hedge_ratio",
]


# --------------------------------------------------------------------------- #
# Low-level OLS helper                                                         #
# --------------------------------------------------------------------------- #
def _ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS via numpy least squares.

    Returns
    -------
    beta : (k,) coefficient vector.
    tstats : (k,) t-statistics on each coefficient using the classical
        homoskedastic covariance sigma^2 * (X'X)^{-1}.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    # Least-squares solution (handles rank deficiency via lstsq's SVD).
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    if dof <= 0:
        return beta, np.full(k, np.nan)
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return beta, np.full(k, np.nan)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0, beta / se, np.nan)
    return beta, tstats


def _as_array(s) -> np.ndarray:
    """Coerce a Series / array-like to a 1-D float ndarray (no NaN handling)."""
    if isinstance(s, pd.Series):
        return s.to_numpy(dtype=float)
    return np.asarray(s, dtype=float).ravel()


# --------------------------------------------------------------------------- #
# Hedge ratio & spread                                                         #
# --------------------------------------------------------------------------- #
def hedge_ratio(y, x) -> Tuple[float, float]:
    """OLS hedge ratio of y on x: y = intercept + beta * x + eps.

    Returns
    -------
    (beta, intercept) : the slope (hedge ratio) and intercept.

    Notes
    -----
    OLS is asymmetric: regressing y~x and x~y gives different betas. For a symmetric
    estimate consider total least squares (orthogonal regression); OLS is the
    Engle-Granger convention and is what adf_tstat's EG critical values assume.
    """
    yv = _as_array(y)
    xv = _as_array(x)
    if yv.shape[0] != xv.shape[0]:
        raise ValueError("y and x must have the same length")
    X = np.column_stack([np.ones_like(xv), xv])
    beta_vec, _ = _ols(X, yv)
    intercept, beta = float(beta_vec[0]), float(beta_vec[1])
    return beta, intercept


def spread(y, x, beta: float, intercept: float = 0.0):
    """Construct the spread s = y - beta*x - intercept.

    Preserves the pandas index if y is a Series (aligns x to y's index when both
    are Series).
    """
    if isinstance(y, pd.Series):
        x_aligned = x.reindex(y.index) if isinstance(x, pd.Series) else pd.Series(
            np.asarray(x, dtype=float), index=y.index
        )
        return y.astype(float) - beta * x_aligned.astype(float) - intercept
    yv = _as_array(y)
    xv = _as_array(x)
    return yv - beta * xv - intercept


# --------------------------------------------------------------------------- #
# Augmented Dickey-Fuller t-statistic                                          #
# --------------------------------------------------------------------------- #
def adf_tstat(series, lags: int = 0) -> float:
    """Augmented Dickey-Fuller t-statistic (constant, no trend).

    Regress delta s_t on [const, s_{t-1}, delta s_{t-1}, ..., delta s_{t-lags}]
    and return the t-statistic on the s_{t-1} coefficient (the gamma in
    delta s_t = mu + gamma * s_{t-1} + sum_i c_i * delta s_{t-i} + eps).

    A more negative statistic => stronger evidence AGAINST a unit root
    (i.e. the series is stationary / mean-reverting).

    Compare to MacKinnon critical values, NOT to a normal/t distribution:
      * raw-series ADF with constant: ~ -2.86 at 5%.
      * Engle-Granger residual (estimated beta): ~ -3.34 at 5% (stricter).

    Parameters
    ----------
    series : array-like or Series. NaNs are dropped before regression.
    lags : number of lagged first-differences to include (the "augmentation").
        Choose enough to whiten residual autocorrelation; too many wastes power.
    """
    s = _as_array(series)
    s = s[~np.isnan(s)]
    n = s.shape[0]
    if lags < 0:
        raise ValueError("lags must be >= 0")
    # Need: response delta s_t for t in [lags+1 .. n-1]; regressor s_{t-1}; and
    # `lags` lagged diffs. Minimum length so dof > 0.
    if n < lags + 3:
        return np.nan

    ds = np.diff(s)            # ds[i] = s[i+1] - s[i], length n-1
    s_lag = s[:-1]             # s_{t-1} aligned with ds, length n-1

    # Build the augmented design over the valid window [lags .. n-2] of ds.
    # Row t corresponds to ds[t]; regressors are s_lag[t] and ds[t-1..t-lags].
    start = lags
    y = ds[start:]                         # delta s_t
    cols = [np.ones_like(y), s_lag[start:]]  # const, s_{t-1}
    for k in range(1, lags + 1):
        cols.append(ds[start - k : len(ds) - k])
    X = np.column_stack(cols)

    _, tstats = _ols(X, y)
    return float(tstats[1])  # t-stat on the s_{t-1} (gamma) coefficient


# --------------------------------------------------------------------------- #
# Engle-Granger two-step                                                       #
# --------------------------------------------------------------------------- #
def engle_granger(y, x, lags: int = 0) -> dict:
    """Engle-Granger two-step cointegration test.

    Step 1: OLS hedge ratio y ~ x -> (beta, intercept).
    Step 2: ADF on the residual spread s = y - beta*x - intercept.

    Returns
    -------
    dict with keys:
      beta, intercept : hedge ratio.
      adf_tstat       : ADF t-stat on the residual (compare to EG critical
                        values, e.g. ~ -3.34 at 5%, NOT -2.86).
      resid           : the residual spread (same type/index as y when y is a Series).

    Reject the unit-root null (=> cointegrated) when adf_tstat is below the EG
    critical value. The order matters: cointegration is asymmetric in OLS, so the
    designated dependent variable (y) affects beta and the test.
    """
    beta, intercept = hedge_ratio(y, x)
    resid = spread(y, x, beta, intercept)
    t = adf_tstat(resid, lags=lags)
    return {
        "beta": beta,
        "intercept": intercept,
        "adf_tstat": t,
        "resid": resid,
    }


# --------------------------------------------------------------------------- #
# Half-life of mean reversion                                                  #
# --------------------------------------------------------------------------- #
def half_life(spread_series) -> float:
    """Half-life of mean reversion from an AR(1) fit on the spread.

    Fit delta s_t = a + (phi - 1) * s_{t-1} + eps  (equivalently
    s_t = a + phi * s_{t-1} + eps), then half-life = -ln(2) / ln(phi).

    Returns
    -------
    float : the half-life in observations (bars). Returns:
        +inf  if phi >= 1 (no mean reversion / explosive),
        nan   if phi <= 0 (no slow positive-autocorr reversion; either oscillatory
              phi<0 or degenerate) or if the fit is undefined.

    Interpretation: a half-life of H bars means a deviation decays to half its size
    in H bars. Very short H (~1) is mostly noise/microstructure; very long H means
    capital is tied up too long to be tradable.
    """
    s = _as_array(spread_series)
    s = s[~np.isnan(s)]
    if s.shape[0] < 3:
        return np.nan
    s_lag = s[:-1]
    s_now = s[1:]
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    beta_vec, _ = _ols(X, s_now)
    phi = float(beta_vec[1])
    if phi >= 1.0:
        return np.inf
    if phi <= 0.0:
        # phi<=0 is not a slowly-decaying mean reversion (oscillatory or degenerate).
        return np.nan
    hl = -np.log(2.0) / np.log(phi)
    return float(hl)


# --------------------------------------------------------------------------- #
# Z-score                                                                      #
# --------------------------------------------------------------------------- #
def zscore(spread_series, window: Optional[int] = None) -> pd.Series:
    """Z-score of the spread.

    Parameters
    ----------
    window : if None, use the full-sample mean/std (ddof=1). NOTE: this is
        IN-SAMPLE and look-ahead -- fine for research/diagnostics, NOT for a
        live signal. If an int, use a trailing rolling mean/std of that window
        (point-in-time, no look-ahead) so z_t depends only on data up to t.

    Returns
    -------
    pandas Series of z-scores (NaN where the rolling window is not yet full).
    """
    if isinstance(spread_series, pd.Series):
        s = spread_series.astype(float)
    else:
        s = pd.Series(_as_array(spread_series))

    if window is None:
        mu = s.mean()
        sd = s.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd

    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=1)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Stateful signal generation                                                  #
# --------------------------------------------------------------------------- #
def generate_signals(
    z,
    entry: float = 2.0,
    exit: float = 0.0,
    stop: float = 4.0,
) -> pd.Series:
    """Stateful pairs signals on the SPREAD, in {-1, 0, +1}.

    Rules (evaluated bar by bar with carried state):
      * Enter LONG spread (+1)  when flat and z <= -entry  (spread cheap).
      * Enter SHORT spread (-1) when flat and z >=  entry  (spread rich).
      * Exit to flat (0) when |z| <= exit (mean reversion captured) OR
        |z| >= stop (deviation too large -> bail / risk control).
      * Otherwise hold the current position.

    The returned series is the DESIRED position at the close of each bar (the
    decision uses information available at that bar). To compute PnL you must lag
    it against the spread return it earns: pnl_t = position.shift(1) * dspread_t,
    so today's decision earns tomorrow's move (no look-ahead).

    Parameters
    ----------
    z : array-like or Series of spread z-scores. NaN z (e.g. warmup of a rolling
        z-score) forces flat and is treated as "no information".
    entry, exit, stop : thresholds with exit < entry < stop (asserted).

    Returns
    -------
    pandas Series of int positions aligned to z.
    """
    if not (exit < entry < stop):
        raise ValueError("require exit < entry < stop")

    if isinstance(z, pd.Series):
        zv = z.to_numpy(dtype=float)
        index = z.index
    else:
        zv = _as_array(z)
        index = pd.RangeIndex(len(zv))

    pos = np.zeros(len(zv), dtype=int)
    state = 0
    for i, zi in enumerate(zv):
        if np.isnan(zi):
            state = 0  # no information -> stay/ go flat
            pos[i] = state
            continue

        az = abs(zi)
        if state == 0:
            if zi <= -entry:
                state = 1   # long spread (cheap)
            elif zi >= entry:
                state = -1  # short spread (rich)
            # else remain flat
        else:
            # In a position: exit on reversion to band or on stop breach.
            if az <= exit or az >= stop:
                state = 0
            # else hold
        pos[i] = state

    return pd.Series(pos, index=index, name="position")


# --------------------------------------------------------------------------- #
# Optional: time-varying hedge ratio via a scalar Kalman filter               #
# --------------------------------------------------------------------------- #
def kalman_hedge_ratio(
    y,
    x,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
) -> pd.DataFrame:
    """Time-varying hedge ratio (and intercept) via a random-walk Kalman filter.

    State theta_t = [intercept_t, beta_t] follows a random walk; observation is
    y_t = [1, x_t] @ theta_t + noise. This is point-in-time: theta_t is updated
    using data up to t only, so the filtered beta is safe to trade on.

    Parameters
    ----------
    delta : controls state transition variance via Vw = delta/(1-delta) * I.
        Larger delta -> faster-adapting (noisier) beta. Typical 1e-5..1e-3.
    obs_var : observation noise variance (measurement R).

    Returns
    -------
    DataFrame with columns ['intercept', 'beta'] indexed like y. The first row is
    the prior (zeros), so use beta from t>=1 onward in practice.

    This is a lightweight alternative to rolling-window OLS when the relationship
    drifts (common in crypto/FX). It does NOT replace the cointegration test --
    re-test stationarity of the resulting time-varying spread.
    """
    yv = _as_array(y)
    xv = _as_array(x)
    if yv.shape[0] != xv.shape[0]:
        raise ValueError("y and x must have the same length")
    n = yv.shape[0]

    wt = delta / (1.0 - delta)          # state transition variance scale
    Vw = wt * np.eye(2)
    R = obs_var

    theta = np.zeros(2)                 # [intercept, beta]
    P = np.zeros((2, 2))                # state covariance (start at 0)
    out = np.zeros((n, 2))

    for t in range(n):
        F = np.array([1.0, xv[t]])      # observation matrix row
        # Predict
        P_pred = P + Vw
        # Innovation
        yhat = F @ theta
        e = yv[t] - yhat
        S = F @ P_pred @ F + R          # innovation variance (scalar)
        K = (P_pred @ F) / S            # Kalman gain (2,)
        # Update
        theta = theta + K * e
        P = P_pred - np.outer(K, F) @ P_pred
        out[t] = theta

    index = y.index if isinstance(y, pd.Series) else pd.RangeIndex(n)
    return pd.DataFrame(out, index=index, columns=["intercept", "beta"])


# --------------------------------------------------------------------------- #
# Self-tests                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 1000

    # ---- Cointegrated pair: x is a random walk; y = 3 + 2x + stationary noise.
    x = np.cumsum(rng.standard_normal(N))
    # AR(1) spread (phi=0.8), not iid: gives a genuine, well-defined mean-reversion
    # half-life. (An iid spread has phi~0, for which the half-life is undefined.)
    noise = np.zeros(N)
    en = 0.5 * rng.standard_normal(N)
    for t in range(1, N):
        noise[t] = 0.8 * noise[t - 1] + en[t]
    y = 3.0 + 2.0 * x + noise
    xs = pd.Series(x)
    ys = pd.Series(y)

    # hedge_ratio recovers beta ~ 2.0, intercept ~ 3.0.
    beta, intercept = hedge_ratio(ys, xs)
    assert abs(beta - 2.0) < 0.1, f"beta off: {beta}"
    assert abs(intercept - 3.0) < 0.5, f"intercept off: {intercept}"

    # The residual spread should be strongly stationary.
    eg = engle_granger(ys, xs, lags=1)
    assert eg["adf_tstat"] < -4.0, f"spread not stationary enough: {eg['adf_tstat']}"
    assert abs(eg["beta"] - 2.0) < 0.1

    # Half-life is finite, positive, and reasonable (the noise is ~iid so reversion
    # is fast -> small but positive half-life).
    hl = half_life(eg["resid"])
    assert np.isfinite(hl) and hl > 0.0, f"bad half-life: {hl}"
    assert hl < 50.0, f"half-life implausibly long for iid noise: {hl}"

    # adf_tstat directly on a constructed AR(1) with known phi -> finite & negative.
    phi_true = 0.8
    ar = np.zeros(N)
    e = rng.standard_normal(N)
    for t in range(1, N):
        ar[t] = phi_true * ar[t - 1] + e[t]
    hl_ar = half_life(pd.Series(ar))
    expected_hl = -np.log(2.0) / np.log(phi_true)  # ~ 3.106
    assert abs(hl_ar - expected_hl) < 1.0, f"AR(1) half-life off: {hl_ar} vs {expected_hl}"
    assert adf_tstat(ar, lags=1) < -2.86, "stationary AR(1) should reject unit root"

    # ---- Two INDEPENDENT random walks: NOT cointegrated -> fails to reject.
    rw1 = np.cumsum(rng.standard_normal(N))
    rw2 = np.cumsum(rng.standard_normal(N))
    eg_spur = engle_granger(pd.Series(rw1), pd.Series(rw2), lags=1)
    assert eg_spur["adf_tstat"] > -2.86, (
        f"independent RWs wrongly look cointegrated: {eg_spur['adf_tstat']}"
    )

    # A pure random walk itself must fail the ADF test (unit root present).
    assert adf_tstat(rw1, lags=1) > -2.86, "random walk wrongly rejected unit root"

    # ---- half_life edge cases.
    # phi >= 1 (explosive / unit root) -> +inf by contract. A finite-sample random
    # walk has estimated phi slightly < 1 (finite half-life), so exercise the inf
    # branch with a clearly explosive series instead.
    _expl = np.zeros(2000)
    _ee = rng.standard_normal(2000)
    for _t in range(1, 2000):
        _expl[_t] = 1.02 * _expl[_t - 1] + _ee[_t]
    assert np.isinf(half_life(_expl)), 'explosive series should give +inf half-life'

    # ---- zscore: full-sample z has ~zero mean and unit std.
    z_full = zscore(eg["resid"])
    assert abs(float(z_full.mean())) < 1e-9
    assert abs(float(z_full.std(ddof=1)) - 1.0) < 1e-9
    # rolling z has NaNs in the warmup and is finite afterward.
    z_roll = zscore(eg["resid"], window=60)
    assert z_roll.iloc[:59].isna().all()
    assert np.isfinite(z_roll.iloc[60:]).all()

    # ---- generate_signals: deterministic state machine on a crafted z path.
    # Path: flat -> dip below -entry (go +1) -> revert into exit band (flat)
    #       -> spike above +entry (go -1) -> blow through stop (flat).
    z_path = pd.Series([0.0, -1.0, -2.5, -1.0, 0.0, 2.5, 1.0, 4.5, 0.0])
    sig = generate_signals(z_path, entry=2.0, exit=0.5, stop=4.0)
    expected = [0, 0, 1, 1, 0, -1, -1, 0, 0]
    assert list(sig) == expected, f"signals mismatch: {list(sig)} vs {expected}"

    # Spec checks: enters +1 when z below -entry; 0 when within exit band.
    assert sig.iloc[2] == 1            # z=-2.5 <= -entry -> long spread
    assert sig.iloc[4] == 0            # |z|=0.0 <= exit  -> flat
    assert sig.iloc[5] == -1           # z=2.5 >= entry   -> short spread
    assert sig.iloc[7] == 0            # |z|=4.5 >= stop  -> stop out

    # NaN z forces flat.
    sig_nan = generate_signals(pd.Series([np.nan, -3.0, np.nan, -3.0]),
                               entry=2.0, exit=0.5, stop=4.0)
    assert list(sig_nan) == [0, 1, 0, 1]

    # threshold ordering is enforced.
    try:
        generate_signals(z_path, entry=2.0, exit=2.0, stop=4.0)
        raise AssertionError("should have rejected exit >= entry")
    except ValueError:
        pass

    # ---- kalman_hedge_ratio converges toward the true beta=2.0 on the coint pair.
    kf = kalman_hedge_ratio(ys, xs, delta=1e-4, obs_var=0.25)
    final_beta = float(kf["beta"].iloc[-1])
    assert abs(final_beta - 2.0) < 0.2, f"kalman beta off: {final_beta}"

    print("pairs_trading.py: all self-tests passed.")
