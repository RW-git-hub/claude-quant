"""robustness.py - Robustness & overfitting lab for backtested edges.

Purpose
-------
A backtested edge can look real and be pure noise + selection. This module
provides resampling tests that ask "could this result have arisen by chance?"
and structural checks for knife-edge parameterization. It complements
Cycle-1 validation (templates/validation.py: purged/combinatorial CV) and
references/stats-risk.md (deflated/probabilistic Sharpe, PBO).

What each tool answers
----------------------
- monte_carlo_permutation_test : Is `metric_fn(values)` larger than what a
  reshuffling of the same values produces? (destroys ordering/structure)
- mean_significance_permutation : Is the mean return > 0 beyond sign noise?
  (sign-flip / randomization test; symmetric-around-0 null)
- stationary_bootstrap_indices : Politis-Romano (1994) geometric-block
  resampler. Preserves short-range serial dependence (autocorrelation, vol
  clustering) that a naive iid bootstrap destroys -> honest CIs for serially
  correlated returns.
- bootstrap_sharpe_ci : Percentile CI for the annualized Sharpe. Use the
  stationary bootstrap (mean_block set) for real return series; iid only for
  genuinely independent observations.
- reality_check_pvalue : White's (2000) Reality Check. When you tried N
  strategies and kept the best, the best in-sample mean is biased upward.
  This p-value tests max_k mean(strat_k) against a bootstrap of the *centered*
  statistics -> corrects for data snooping / multiple testing.
- parameter_plateau_score : Robust parameterizations sit on a broad plateau,
  not a lone spike. Reports the fraction of the parameter grid within `frac`
  of the peak. Higher = less overfit to a single lucky cell.

Conventions (house style)
-------------------------
- Simple returns; annualized Sharpe = mean/std(ddof=1)*sqrt(ppy), daily ppy=252
  (assumes iid; autocorrelation inflates naive Sharpe -- prefer the bootstrap CI
  here, which can use block resampling).
- Permutation p-values use the (1 + count) / (n_perm + 1) estimator so the
  p-value is never exactly 0 and the test is conservative/valid.
- All randomness via numpy.random.default_rng(seed) for determinism.

Pitfalls these tools DON'T fix (still your job)
----------------------------------------------
- Look-ahead / survivorship bias in the underlying returns: resampling a
  contaminated series just gives you a confident wrong answer.
- iid bootstrap on autocorrelated returns understates the CI -> false
  confidence. Use mean_block.
- Reality Check needs the FULL set of strategies you tried (incl. discarded
  ones); feeding only survivors reintroduces snooping.
- Permutation of `values` destroys time structure: appropriate for tests where
  the null is "ordering carries no information" (e.g. signal timing), not for
  every metric.

numpy / pandas / stdlib only (statistics.NormalDist permitted).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_1d_array(x) -> np.ndarray:
    """Coerce a Series / list / array to a finite 1-D float ndarray."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        a = np.asarray(x.values, dtype=float).ravel()
    else:
        a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("empty input array")
    if not np.all(np.isfinite(a)):
        raise ValueError("input contains non-finite values (NaN/inf); clean first")
    return a


def annualized_sharpe(returns, ppy: int = 252) -> float:
    """Annualized Sharpe of a (excess) return series.

    Sharpe = mean / std(ddof=1) * sqrt(ppy). Returns np.nan if std == 0.
    Note: assumes iid; serial correlation biases this -- use bootstrap_sharpe_ci
    with a block length for an honest interval.
    """
    a = _as_1d_array(returns)
    if a.size < 2:
        return float("nan")
    sd = a.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(a.mean() / sd * np.sqrt(ppy))


# --------------------------------------------------------------------------- #
# permutation / randomization tests
# --------------------------------------------------------------------------- #
def monte_carlo_permutation_test(
    values,
    metric_fn: Callable[[np.ndarray], float],
    n_perm: int = 1000,
    seed: int = 0,
    alternative: str = "greater",
) -> dict:
    """Monte-Carlo permutation test for an arbitrary scalar metric.

    Each permutation shuffles `values` (destroying their ordering), recomputes
    `metric_fn`, and builds the null distribution. The p-value uses the
    (1 + count) / (n_perm + 1) estimator (Phipson & Smyth 2010), which is valid
    and never returns exactly 0.

    Use when the null hypothesis is "the ordering of `values` carries no
    information" -- e.g. testing whether a signal's *timing* (not just its
    marginal distribution) produces the observed metric.

    Parameters
    ----------
    values : 1-D returns/values whose order is shuffled under the null.
    metric_fn : array -> float, e.g. np.mean or annualized_sharpe.
    n_perm : number of permutations.
    alternative : 'greater'  -> p = P(null >= observed)
                  'less'     -> p = P(null <= observed)
                  'two-sided'-> p based on |null| >= |observed| (around the
                                null mean) using both tails.

    Returns
    -------
    dict(observed, p_value, null) where null is the ndarray of permuted metrics.
    """
    a = _as_1d_array(values)
    rng = np.random.default_rng(seed)
    observed = float(metric_fn(a))

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = float(metric_fn(rng.permutation(a)))

    if alternative == "greater":
        count = int(np.sum(null >= observed))
    elif alternative == "less":
        count = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        center = null.mean()
        count = int(np.sum(np.abs(null - center) >= np.abs(observed - center)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    p_value = (1 + count) / (n_perm + 1)
    return {"observed": observed, "p_value": float(p_value), "null": null}


def mean_significance_permutation(
    returns,
    n_perm: int = 1000,
    seed: int = 0,
) -> float:
    """Sign-flip randomization test that mean(returns) > 0.

    Under H0 the return distribution is symmetric about 0, so flipping the sign
    of each observation is equiprobable. We draw random +/-1 masks, recompute
    the mean, and ask how often a random sign-assignment beats the observed
    mean. The observed assignment (all +1) is included via the +1 / +1 estimator.

    This is more appropriate than a t-test for fat-tailed / non-normal returns,
    and unlike monte_carlo_permutation_test it preserves each observation's
    magnitude (only the sign is randomized).

    Returns
    -------
    one-sided p-value for H1: mean > 0  ->  (1 + #{null_mean >= observed}) / (n+1)
    """
    a = _as_1d_array(returns)
    rng = np.random.default_rng(seed)
    observed = float(a.mean())

    n = a.size
    signs = rng.integers(0, 2, size=(n_perm, n)) * 2 - 1  # random +/-1
    null_means = (signs * a).mean(axis=1)

    count = int(np.sum(null_means >= observed))
    return float((1 + count) / (n_perm + 1))


# --------------------------------------------------------------------------- #
# stationary (block) bootstrap
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(
    n: int,
    mean_block: float,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Politis-Romano (1994) stationary-bootstrap index matrix.

    Builds `n_samples` resampled index sequences of length `n`. Each sequence is
    a concatenation of blocks whose lengths are i.i.d. Geometric(p) with
    p = 1 / mean_block (so expected block length = mean_block). Block start
    points are uniform on [0, n) and indexing wraps around (circular), which is
    what makes the resampled series stationary.

    Use these indices to resample ANY aligned arrays (returns, signals, pnl)
    consistently while preserving short-range serial dependence -- autocorrelation
    and volatility clustering that an iid bootstrap would destroy.

    Parameters
    ----------
    n : length of the original series (and of each resample).
    mean_block : expected block length (>= 1). Larger preserves longer-range
                 dependence; rule of thumb ~ n**(1/3) for moderate autocorr.
    n_samples : number of bootstrap replicates (rows of the output).

    Returns
    -------
    int ndarray of shape (n_samples, n) with values in [0, n).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    rng = np.random.default_rng(seed)
    p = 1.0 / float(mean_block)

    out = np.empty((n_samples, n), dtype=np.int64)
    for s in range(n_samples):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            cur = int(rng.integers(0, n))      # random block start
            idx[i] = cur
            i += 1
            while i < n:
                if rng.random() < p:           # start a new block
                    break
                cur = (cur + 1) % n            # continue current block (wrap)
                idx[i] = cur
                i += 1
        out[s] = idx
    return out


def bootstrap_sharpe_ci(
    returns,
    n_boot: int = 1000,
    mean_block: Optional[float] = None,
    alpha: float = 0.05,
    ppy: int = 252,
    seed: int = 0,
) -> tuple:
    """Percentile bootstrap CI for the annualized Sharpe ratio.

    If `mean_block` is given, uses the stationary bootstrap (recommended for
    real return series with serial dependence). If None, uses an iid bootstrap
    (only valid when observations are genuinely independent).

    Parameters
    ----------
    returns : 1-D (excess) returns.
    n_boot : number of bootstrap replicates.
    mean_block : expected block length for the stationary bootstrap, or None
                 for iid resampling.
    alpha : two-sided level; CI is the [alpha/2, 1-alpha/2] percentiles.
    ppy : periods per year for annualization.

    Returns
    -------
    (lo, hi) : floats, the percentile confidence interval for annualized Sharpe.
    """
    a = _as_1d_array(returns)
    n = a.size
    if n < 2:
        raise ValueError("need at least 2 observations")
    rng = np.random.default_rng(seed)

    if mean_block is None:
        # iid bootstrap: resample positions with replacement
        idx = rng.integers(0, n, size=(n_boot, n))
    else:
        idx = stationary_bootstrap_indices(n, mean_block, n_boot, seed=seed)

    samples = a[idx]                                  # (n_boot, n)
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(sds > 0, means / sds * np.sqrt(ppy), np.nan)
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return (float("nan"), float("nan"))

    lo = float(np.percentile(sharpes, 100 * alpha / 2))
    hi = float(np.percentile(sharpes, 100 * (1 - alpha / 2)))
    return (lo, hi)


# --------------------------------------------------------------------------- #
# White's Reality Check (data-snooping correction)
# --------------------------------------------------------------------------- #
def reality_check_pvalue(
    returns_matrix,
    n_boot: int = 1000,
    mean_block: float = 10,
    seed: int = 0,
) -> float:
    """White's (2000) Reality Check p-value for the best of N strategies.

    Given a (T x N) matrix of per-period performance for N candidate strategies
    (columns), the in-sample winner's mean is upward-biased by selection. The
    Reality Check tests:

        H0: max_k E[ f_k ] <= 0   (no strategy beats the benchmark of 0)

    Test statistic V = sqrt(T) * max_k mean(f_k). The null distribution is
    obtained by stationary-bootstrapping the *centered* statistics
    sqrt(T) * (mean(f_k^*) - mean(f_k)) and taking their max over k. Centering
    (demeaning each column by its own full-sample mean) is what enforces the
    least-favorable H0 and makes the correction honest.

    Feed the FULL set of strategies you tried, including the ones you discarded
    -- otherwise you reintroduce the snooping you are trying to correct. Columns
    should already be performance relative to the benchmark (use excess returns,
    or strategy-minus-benchmark, so that 0 is the no-edge value).

    Parameters
    ----------
    returns_matrix : (T, N) array-like of per-period relative performance.
    n_boot : bootstrap replicates.
    mean_block : expected block length for the stationary bootstrap.

    Returns
    -------
    p-value in (0, 1]; small => the best strategy's edge survives the
    multiple-testing correction.
    """
    M = np.asarray(
        returns_matrix.values if isinstance(returns_matrix, pd.DataFrame)
        else returns_matrix,
        dtype=float,
    )
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T x N)")
    if not np.all(np.isfinite(M)):
        raise ValueError("returns_matrix contains non-finite values")
    T, N = M.shape
    if T < 2 or N < 1:
        raise ValueError("need T>=2 and N>=1")

    col_means = M.mean(axis=0)                         # (N,)
    sqrtT = np.sqrt(T)
    V = float(sqrtT * np.max(col_means))               # observed statistic

    idx = stationary_bootstrap_indices(T, mean_block, n_boot, seed=seed)  # (B,T)
    # For each replicate: bootstrap mean of each column, centered by full mean.
    boot = M[idx]                                      # (B, T, N)
    boot_means = boot.mean(axis=1)                     # (B, N)
    centered = sqrtT * (boot_means - col_means[None, :])
    V_star = np.max(centered, axis=1)                  # (B,)

    count = int(np.sum(V_star >= V))
    return float((1 + count) / (n_boot + 1))


# --------------------------------------------------------------------------- #
# Hansen's Superior Predictive Ability (SPA) test
# --------------------------------------------------------------------------- #
def spa_pvalue(
    returns_matrix,
    n_boot: int = 1000,
    mean_block: float = 10,
    seed: int = 0,
    variant: str = "studentized",
) -> float:
    """Hansen's (2005) Superior Predictive Ability (SPA) p-value.

    Like White's Reality Check, SPA tests whether the BEST of N candidate
    strategies beats the benchmark (0) after correcting for data snooping:

        H0: max_k E[ f_k ] <= 0

    SPA improves on the Reality Check in two ways, both of which make it more
    powerful (lower p, i.e. more likely to detect a real edge) when the candidate
    set is padded with many obviously-dead strategies:

    1. STUDENTIZATION. Each column's mean is scaled by its own standard error
       (estimated by the stationary bootstrap), so the test statistic is
           T_SPA = max_k  sqrt(T) * mean(f_k) / se_k
       This puts strategies on a common scale and stops a single high-variance
       column from dominating the null distribution.
    2. RECENTERING ONLY INFERIOR STRATEGIES. The Reality Check recenters every
       column by its full-sample mean (the least-favorable configuration), which
       lets hopeless strategies (very negative mean) inflate the null and bury a
       true edge. SPA instead recenters a strategy by its mean ONLY when that
       mean is not "too negative" to plausibly be the best:
           g_k = mean(f_k) * 1{ mean(f_k) >= -A_k }
       with the Hansen threshold A_k = (1/4) * n^{-1/4} * se_k * sqrt(T)
       (so in studentized units the cutoff is -(1/4) * n^{-1/4}). Strategies
       below the threshold are recentered to 0 (treated as exactly at the
       benchmark) and therefore cannot pump up the bootstrap maximum. This is
       Hansen's consistent ("c") p-value.

    The bootstrap null statistic for replicate b is
        T*_b = max_k  sqrt(T) * (mean(f_k^{*b}) - g_k) / se_k
    and the p-value is (1 + #{ T*_b >= T_SPA }) / (n_boot + 1).

    Iron-Law note: this is an OFFLINE evaluation tool, not a backtest signal --
    it consumes a finished (T x N) panel of per-period performance and uses the
    FULL sample (including future rows) to compute SEs and recentering. It is
    leak-free for its intended use (post-hoc multiple-testing correction) but
    must NOT be embedded inside a walk-forward loop as if it were causal.

    Feed the FULL set of strategies you tried, including discarded ones; columns
    must be performance relative to the benchmark (excess or strategy-minus-
    benchmark) so that 0 is the no-edge value.

    Parameters
    ----------
    returns_matrix : (T, N) array-like of per-period relative performance.
    n_boot : stationary-bootstrap replicates.
    mean_block : expected block length for the stationary bootstrap.
    variant : "studentized" (default, Hansen's studentized statistic) or
              "raw" (skip studentization; recenter inferior strategies on the
              raw mean scale -- a midpoint between White's RC and full SPA).

    Returns
    -------
    p-value in (0, 1]; small => the best strategy's edge survives the SPA
    multiple-testing correction. By construction SPA's p-value is <= White's
    Reality Check p-value when many clearly-inferior strategies are present.

    Reference
    ---------
    Hansen, P. R. (2005), "A Test for Superior Predictive Ability", Journal of
    Business & Economic Statistics 23(4), 365-380.
    """
    if variant not in ("studentized", "raw"):
        raise ValueError("variant must be 'studentized' or 'raw'")
    M = np.asarray(
        returns_matrix.values if isinstance(returns_matrix, pd.DataFrame)
        else returns_matrix,
        dtype=float,
    )
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T x N)")
    if not np.all(np.isfinite(M)):
        raise ValueError("returns_matrix contains non-finite values")
    T, N = M.shape
    if T < 2 or N < 1:
        raise ValueError("need T>=2 and N>=1")

    col_means = M.mean(axis=0)                          # (N,)
    sqrtT = np.sqrt(T)

    idx = stationary_bootstrap_indices(T, mean_block, n_boot, seed=seed)  # (B,T)
    boot = M[idx]                                       # (B, T, N)
    boot_means = boot.mean(axis=1)                      # (B, N)

    # Bootstrap SE of sqrt(T)*mean for each column = sqrt(T)*std of the bootstrap
    # means (Politis-Romano stationary-bootstrap variance estimate). This is the
    # studentizing denominator se_k (in sqrt(T)*mean units).
    se = sqrtT * boot_means.std(axis=0, ddof=1)         # (N,)
    se = np.where(se > 0, se, np.inf)                   # dead-flat column -> drop

    if variant == "studentized":
        scale = se                                     # divide stats by se_k
    else:
        scale = np.ones(N)                             # raw scale

    # Observed studentized (or raw) statistic.
    t_obs = sqrtT * col_means / scale                  # (N,)
    T_obs = float(np.max(t_obs))

    # Hansen recentering: keep mean only if it is not too negative to be best.
    # In studentized units the threshold is -(1/4) * N_T^{-1/4} where N_T = T;
    # equivalently g_k = col_means * 1{ sqrtT*mean/se >= -(1/4) T^{-1/4} }.
    thresh = -0.25 * (T ** -0.25)                       # studentized cutoff
    studentized_means = sqrtT * col_means / se         # always studentize the gate
    keep = studentized_means >= thresh                 # True => recenter on mean
    g = np.where(keep, col_means, 0.0)                 # (N,) recentering vector

    # Bootstrap null: max over k of studentized (mean* - g_k).
    centered = sqrtT * (boot_means - g[None, :]) / scale[None, :]   # (B, N)
    T_star = np.max(centered, axis=1)                  # (B,)

    count = int(np.sum(T_star >= T_obs))
    return float((1 + count) / (n_boot + 1))


# --------------------------------------------------------------------------- #
# parameter-stability / plateau diagnostic
# --------------------------------------------------------------------------- #
def parameter_plateau_score(metric_grid, frac: float = 0.9) -> dict:
    """Measure how broad the 'good' region of a parameter sweep is.

    A robust strategy performs well across a contiguous neighbourhood of
    parameter settings (a plateau); an overfit one peaks on a single lucky cell
    surrounded by mediocrity (a spike). This computes the fraction of grid cells
    whose metric is within `frac` of the global peak.

    Parameters
    ----------
    metric_grid : array-like (1-D or N-D) of the performance metric over a
                  parameter grid (e.g. Sharpe across (lookback, threshold)).
                  Assumes 'higher is better'; negate first if minimizing.
    frac : threshold as a fraction of the peak (default 0.9 -> within 10%).
           Comparison handles negative peaks correctly (uses frac*peak as the
           cutoff regardless of sign).

    Returns
    -------
    dict(peak, plateau_ratio, argmax) where:
        peak          : global maximum metric.
        plateau_ratio : fraction of cells with metric >= frac*peak in [0,1];
                        higher = broader plateau = more robust.
        argmax        : index of the peak (tuple for N-D grids, int for 1-D).
    """
    g = np.asarray(metric_grid, dtype=float)
    if g.size == 0:
        raise ValueError("empty grid")
    if not np.all(np.isfinite(g)):
        raise ValueError("grid contains non-finite values")
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1]")

    peak = float(g.max())
    threshold = frac * peak
    plateau_ratio = float(np.mean(g >= threshold))

    flat_arg = int(np.argmax(g))
    argmax = flat_arg if g.ndim == 1 else tuple(int(i) for i in np.unravel_index(flat_arg, g.shape))

    return {"peak": peak, "plateau_ratio": plateau_ratio, "argmax": argmax}


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # (a) sign-flip mean test: noise insignificant, drift significant
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.01, size=750)
    p_noise = mean_significance_permutation(noise, n_perm=500, seed=1)
    assert p_noise > 0.05, f"zero-mean noise should be insignificant, got {p_noise}"

    drift = rng.normal(0.0015, 0.01, size=750)  # strong positive drift vs vol
    p_drift = mean_significance_permutation(drift, n_perm=500, seed=2)
    assert p_drift < 0.05, f"positive drift should be significant, got {p_drift}"

    # (b) order-shuffle permutation is BLIND to the mean: the mean is invariant to
    # reordering, so every permutation equals the observed and p collapses to ~1.0.
    # Teaching point: use the sign-flip test (a) for the mean; reserve
    # monte_carlo_permutation_test for PATH-DEPENDENT metrics (max drawdown, run
    # length) or signal-timing alignment, where order actually matters.
    res = monte_carlo_permutation_test(
        drift, metric_fn=np.mean, n_perm=500, seed=3, alternative="greater"
    )
    assert res["observed"] > 0
    assert res["null"].shape == (500,)
    assert np.isfinite(res["null"]).all()
    assert res["p_value"] > 0.5, f"order-shuffle is blind to the mean (expect p~1), got {res['p_value']}"

    # two-sided p-value is a valid probability
    res2 = monte_carlo_permutation_test(
        noise, metric_fn=np.mean, n_perm=300, seed=4, alternative="two-sided"
    )
    assert 0.0 < res2["p_value"] <= 1.0

    # (c) stationary bootstrap indices: shape + in-range
    n = 200
    idx = stationary_bootstrap_indices(n, mean_block=10, n_samples=50, seed=5)
    assert idx.shape == (50, n)
    assert idx.min() >= 0 and idx.max() < n
    assert idx.dtype.kind in ("i", "u")
    # mean block ~ 10 implies fewer "new block" jumps than iid; loose check that
    # consecutive-index continuation happens often (serial structure preserved):
    cont = np.mean((idx[:, 1:] - idx[:, :-1]) % n == 1)
    assert cont > 0.5, f"expected mostly contiguous steps with block=10, got {cont}"

    # (d) bootstrap Sharpe CI brackets the point estimate
    series = rng.normal(0.0008, 0.01, size=1000)  # positive Sharpe
    point = annualized_sharpe(series)
    assert point > 0
    lo_iid, hi_iid = bootstrap_sharpe_ci(series, n_boot=500, mean_block=None, seed=6)
    assert lo_iid < hi_iid
    slack = 0.05 * (hi_iid - lo_iid)
    assert lo_iid - slack <= point <= hi_iid + slack, (
        f"iid CI [{lo_iid}, {hi_iid}] should bracket point {point}"
    )
    lo_b, hi_b = bootstrap_sharpe_ci(series, n_boot=500, mean_block=20, seed=6)
    assert lo_b < hi_b
    slack_b = 0.05 * (hi_b - lo_b)
    assert lo_b - slack_b <= point <= hi_b + slack_b, (
        f"block CI [{lo_b}, {hi_b}] should bracket point {point}"
    )

    # (e) Reality Check: N=20 independent no-edge strategies -> not significant
    T, N = 500, 20
    rc_rng = np.random.default_rng(7)
    no_edge = rc_rng.normal(0.0, 0.01, size=(T, N))  # all true means = 0
    p_rc = reality_check_pvalue(no_edge, n_boot=500, mean_block=10, seed=8)
    assert p_rc > 0.05, (
        f"best of 20 no-edge strategies should NOT be significant after "
        f"snooping correction, got p={p_rc}"
    )
    # sanity: a genuine edge in one column should be detectable
    edge = rc_rng.normal(0.0, 0.01, size=(T, N))
    edge[:, 3] += 0.0025  # strong real edge in one strategy
    p_rc_edge = reality_check_pvalue(edge, n_boot=500, mean_block=10, seed=9)
    assert p_rc_edge < 0.05, f"a real strong edge should survive RC, got {p_rc_edge}"

    # (f) Hansen SPA test
    # 20 no-edge strategies -> SPA should NOT be significant.
    spa_rng = np.random.default_rng(11)
    spa_no_edge = spa_rng.normal(0.0, 0.01, size=(500, 20))
    p_spa_null = spa_pvalue(spa_no_edge, n_boot=500, mean_block=10, seed=12)
    assert p_spa_null > 0.05, (
        f"best of 20 no-edge strategies should NOT be SPA-significant, got {p_spa_null}"
    )
    # inject one genuine edge -> SPA should detect it.
    spa_edge = spa_rng.normal(0.0, 0.01, size=(500, 20))
    spa_edge[:, 7] += 0.0025
    p_spa_edge = spa_pvalue(spa_edge, n_boot=500, mean_block=10, seed=13)
    assert p_spa_edge < 0.05, f"a real edge should survive SPA, got {p_spa_edge}"

    # SPA <= White's RC when the candidate set is padded with many dead
    # strategies: one modest real edge + 60 strongly-negative dead strategies.
    # RC recenters every dead column on its (very negative) mean, inflating the
    # null max; SPA recenters dead strategies to 0 instead -> a smaller null and
    # a smaller (or equal) p-value.
    pad_rng = np.random.default_rng(14)
    Tt, Nedge, Ndead = 500, 1, 60
    real = pad_rng.normal(0.0, 0.01, size=(Tt, Nedge)) + 0.0014   # modest edge
    dead = pad_rng.normal(-0.004, 0.012, size=(Tt, Ndead))        # clearly dead
    padded = np.hstack([real, dead])
    p_rc_pad = reality_check_pvalue(padded, n_boot=800, mean_block=10, seed=15)
    p_spa_pad = spa_pvalue(padded, n_boot=800, mean_block=10, seed=15)
    assert p_spa_pad <= p_rc_pad + 1e-12, (
        f"SPA p ({p_spa_pad}) should be <= RC p ({p_rc_pad}) with many dead strategies"
    )

    # p-values are valid probabilities; "raw" variant runs and is bounded.
    assert 0.0 < p_spa_null <= 1.0 and 0.0 < p_spa_edge <= 1.0
    p_spa_raw = spa_pvalue(spa_edge, n_boot=300, mean_block=10, seed=16, variant="raw")
    assert 0.0 < p_spa_raw <= 1.0

    # (g) plateau score: broad peak > single spike
    xx, yy = np.meshgrid(np.linspace(-3, 3, 21), np.linspace(-3, 3, 21))
    broad = np.exp(-(xx**2 + yy**2) / 8.0)          # wide gaussian -> plateau
    spike = np.zeros((21, 21))
    spike[10, 10] = 1.0                             # lone spike
    sp_broad = parameter_plateau_score(broad, frac=0.9)
    sp_spike = parameter_plateau_score(spike, frac=0.9)
    assert sp_broad["plateau_ratio"] > sp_spike["plateau_ratio"], (
        f"broad peak ({sp_broad['plateau_ratio']}) should beat spike "
        f"({sp_spike['plateau_ratio']})"
    )
    assert sp_spike["argmax"] == (10, 10)
    assert 0.0 <= sp_broad["plateau_ratio"] <= 1.0
    # 1-D grid returns int argmax
    sp_1d = parameter_plateau_score(np.array([0.1, 0.5, 0.9, 0.4]), frac=0.9)
    assert sp_1d["argmax"] == 2 and isinstance(sp_1d["argmax"], int)
    # negative-peak handling: cutoff = frac*peak still well-defined
    sp_neg = parameter_plateau_score(np.array([-1.0, -2.0, -3.0]), frac=0.9)
    assert sp_neg["peak"] == -1.0 and sp_neg["argmax"] == 0

    print("robustness.py self-tests passed.")
