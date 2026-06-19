"""
calibration.py - Probability-calibration toolkit (numpy-only, dependency-free).

Why this exists
---------------
Many trading and betting workflows turn a model's predicted probability into a
*bet size* (Kelly, meta-label sizing). Ranking quality (AUC / IC) is NOT enough
there: if "70%" forecasts win 85% of the time, every Kelly stake is wrong and the
EV you think you are harvesting is corrupted even though the ordering of bets is
fine. Calibration measures and FIXES the map from predicted probability to
realized frequency. This module supplies the reliability/ECE/Brier-decomposition
diagnostics plus two recalibrators (Platt, isotonic) with no scikit-learn / scipy
dependency, so the skill can assume it is always available.

Used by:
  - references/prediction-sports-markets.md  §15 (calibrate model probs before Kelly)
  - references/ml-for-alpha.md                §9  (meta-label / probability bet sizing)

Iron Law -- OUT-OF-SAMPLE IS SACRED (this is the one that bites here)
--------------------------------------------------------------------
A recalibrator is a *fitted model*. Fit it on a HELD-OUT calibration fold that the
base model never trained on, and that is DISJOINT from the test fold you report on.
Fitting Platt/isotonic on the same data you evaluate makes any miscalibration
vanish trivially -- that is leakage, and it inflates the post-calibration scores
you would quote. For time-dependent data, the calibration fold must be carved with
purge+embargo (see templates/validation.py: PurgedKFold / CombinatorialPurgedKFold)
so an overlapping label window cannot bleed test outcomes into the fit.

Conventions
-----------
- p : predicted probabilities of the positive class, in [0, 1] (clipped internally
      where a transform needs the open interval).
- y : binary outcomes in {0, 1} (floats accepted; must be 0/1 valued).
- All scoring functions are proper for binary y. Lower Brier / log-loss is better.
- reliability_curve / ECE support two binning strategies:
    * 'uniform'  : equal-width bins on [0, 1] (classic reliability diagram).
    * 'quantile' : equal-COUNT bins (robust when predictions cluster; recommended
                   when probabilities are bunched, e.g. near 0.5).

Relationship to betting_markets.py
-----------------------------------
templates/betting_markets.py also ships brier_score / log_loss with the SAME
math but argument order (probs, outcomes). The copies here are intentional so this
module is standalone (numpy-only); both agree to floating point. Use whichever
module you have imported -- do not assume one wraps the other.

Dependencies: numpy + Python stdlib only. No scipy, no sklearn.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


# --------------------------------------------------------------------------- #
# Input validation helpers                                                    #
# --------------------------------------------------------------------------- #
def _check_py(p: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D float arrays and validate p in [0,1], y in {0,1}, same length."""
    pa = np.asarray(p, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if pa.shape != ya.shape:
        raise ValueError("p and y must have the same length")
    if pa.size == 0:
        raise ValueError("p and y must be non-empty")
    if np.any(~np.isfinite(pa)) or np.any(~np.isfinite(ya)):
        raise ValueError("p and y must be finite")
    if np.any(pa < 0.0) or np.any(pa > 1.0):
        raise ValueError("predicted probabilities p must lie in [0, 1]")
    if np.any((ya != 0.0) & (ya != 1.0)):
        raise ValueError("outcomes y must be binary {0, 1}")
    return pa, ya


# --------------------------------------------------------------------------- #
# Scoring rules                                                               #
# --------------------------------------------------------------------------- #
def brier_score(p: ArrayLike, y: ArrayLike) -> float:
    """Brier score = mean((p - y)^2). Proper scoring rule; lower is better.

    0 = perfect, 0.25 = always predicting 0.5, ~1 = confidently wrong.
    """
    pa, ya = _check_py(p, y)
    return float(np.mean((pa - ya) ** 2))


def log_loss(p: ArrayLike, y: ArrayLike, eps: float = 1e-15) -> float:
    """Binary cross-entropy = -mean(y*log p + (1-y)*log(1-p)), p clipped to [eps,1-eps].

    Proper scoring rule; punishes confident wrong calls much harder than Brier.
    Clipping keeps it finite when a forecast is exactly 0 or 1.
    """
    pa, ya = _check_py(p, y)
    pc = np.clip(pa, eps, 1.0 - eps)
    return float(-np.mean(ya * np.log(pc) + (1.0 - ya) * np.log(1.0 - pc)))


# --------------------------------------------------------------------------- #
# Reliability curve and calibration error                                     #
# --------------------------------------------------------------------------- #
def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Return monotonically non-decreasing bin edges of length n_bins+1 on [0,1]."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        # Equal-count edges from empirical quantiles of p; clamp ends to [0,1].
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, qs)
        edges[0], edges[-1] = 0.0, 1.0
        # Collapse duplicate edges (ties in p) so empty degenerate bins don't appear.
        return np.unique(edges)
    raise ValueError("strategy must be 'uniform' or 'quantile'")


def reliability_curve(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Bin predictions and return per-bin mean predicted prob vs observed frequency.

    Returns a dict with equal-length arrays (one entry per NON-EMPTY bin):
      - 'mean_pred'   : mean predicted probability in the bin (x of a reliability diagram)
      - 'obs_freq'    : observed positive frequency in the bin (y of the diagram)
      - 'count'       : number of samples in the bin
      - 'bin_lo'/'bin_hi' : the bin's edges
    Perfect calibration is obs_freq == mean_pred (the 45-degree line). obs_freq BELOW
    mean_pred => overconfident in that range (and, for market prices, the signature of
    favorite-longshot bias at the extremes).

    Binning is half-open [lo, hi) except the final bin is closed [lo, hi] so p == 1
    lands somewhere. Empty bins are dropped (so e.g. plotting won't draw phantom points).
    """
    pa, ya = _check_py(p, y)
    edges = _bin_edges(pa, n_bins, strategy)
    # np.digitize with right=False gives bin index in 1..len(edges)-1 for interior pts;
    # clip so the rightmost point (p == 1 == edges[-1]) joins the last real bin.
    idx = np.clip(np.digitize(pa, edges[1:-1], right=False), 0, len(edges) - 2)

    mean_pred, obs_freq, count, lo, hi = [], [], [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred.append(float(pa[mask].mean()))
        obs_freq.append(float(ya[mask].mean()))
        count.append(n)
        lo.append(float(edges[b]))
        hi.append(float(edges[b + 1]))
    return {
        "mean_pred": np.asarray(mean_pred, dtype=float),
        "obs_freq": np.asarray(obs_freq, dtype=float),
        "count": np.asarray(count, dtype=int),
        "bin_lo": np.asarray(lo, dtype=float),
        "bin_hi": np.asarray(hi, dtype=float),
    }


def expected_calibration_error(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """ECE = sum_b (n_b / N) * |obs_freq_b - mean_pred_b|.

    The count-weighted average gap between confidence and accuracy across bins.
    0 = perfectly calibrated (within binning resolution). Sensitive to n_bins and
    to strategy: report the bin settings alongside the number.
    """
    curve = reliability_curve(p, y, n_bins=n_bins, strategy=strategy)
    n = curve["count"]
    if n.sum() == 0:
        return 0.0
    gap = np.abs(curve["obs_freq"] - curve["mean_pred"])
    return float(np.sum(n * gap) / n.sum())


def max_calibration_error(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """MCE = max_b |obs_freq_b - mean_pred_b|: the worst-calibrated bin's gap."""
    curve = reliability_curve(p, y, n_bins=n_bins, strategy=strategy)
    if curve["count"].size == 0:
        return 0.0
    return float(np.max(np.abs(curve["obs_freq"] - curve["mean_pred"])))


# --------------------------------------------------------------------------- #
# Brier decomposition (Murphy 1973)                                           #
# --------------------------------------------------------------------------- #
def brier_decomposition(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Murphy (1973) three-term decomposition of the Brier score over K bins.

        Brier ~= reliability - resolution + uncertainty

    where, with N total samples, K non-empty bins, n_k samples in bin k whose mean
    forecast is f_k and observed frequency is o_k, and base rate o_bar = mean(y):

        reliability = (1/N) * sum_k n_k * (f_k - o_k)^2     (lower is better; 0 = calibrated)
        resolution  = (1/N) * sum_k n_k * (o_k - o_bar)^2   (higher is better; bins differ)
        uncertainty = o_bar * (1 - o_bar)                   (irreducible base-rate variance)

    NOTE: the identity is EXACT only when each bin's forecast is replaced by its bin
    mean f_k (i.e. the Brier of the *binned* forecasts). With wide bins it approximates
    the raw Brier; with one-sample-per-bin (or matching mean_pred per bin) it reconstructs
    it. We therefore also return:
      - 'brier'        : the raw mean((p-y)^2)
      - 'brier_binned' : reliability - resolution + uncertainty (the decomposition's Brier)
    These coincide exactly when every bin is a single distinct forecast value; otherwise
    'brier_binned' is the Brier of the binned forecasts. Use a quantile/fine binning to
    make them agree (the self-tests verify the exact single-value-bin reconstruction).

    Returns dict: reliability, resolution, uncertainty, brier, brier_binned.
    """
    pa, ya = _check_py(p, y)
    n_total = pa.size
    o_bar = float(ya.mean())
    uncertainty = o_bar * (1.0 - o_bar)

    curve = reliability_curve(pa, ya, n_bins=n_bins, strategy=strategy)
    n_k = curve["count"].astype(float)
    f_k = curve["mean_pred"]
    o_k = curve["obs_freq"]

    reliability = float(np.sum(n_k * (f_k - o_k) ** 2) / n_total)
    resolution = float(np.sum(n_k * (o_k - o_bar) ** 2) / n_total)
    brier_binned = reliability - resolution + uncertainty
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": float(np.mean((pa - ya) ** 2)),
        "brier_binned": float(brier_binned),
    }


# --------------------------------------------------------------------------- #
# Logit helpers                                                               #
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pc = np.clip(p, eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid for arrays.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# --------------------------------------------------------------------------- #
# Platt scaling (logistic recalibration on the logit of p)                    #
# --------------------------------------------------------------------------- #
def platt_scale(
    p: ArrayLike,
    y: ArrayLike,
    max_iter: int = 100,
    tol: float = 1e-10,
    l2: float = 1e-6,
) -> Callable[[ArrayLike], np.ndarray]:
    """Fit Platt scaling: calibrated_prob = sigmoid(a * logit(p) + b).

    Fits (a, b) by Newton / IRLS maximum-likelihood logistic regression of y on the
    single feature x = logit(p) (hand-rolled; no sklearn). A tiny L2 ridge (`l2`) on
    (a, b) keeps the Hessian invertible under perfect separation. Returns a CLOSURE
    `transform(p_new) -> calibrated probabilities`, so you fit once on the held-out
    calibration fold and apply to test/live.

    Platt is parametric (2 params): robust on small calibration sets, but it can only
    apply a monotone logistic squashing of the logit -- it cannot fix non-monotone
    miscalibration (use isotonic for that).

    Iron Law: fit on a calibration fold DISJOINT from both the base model's training
    data and the evaluation fold (see module docstring).
    """
    pa, ya = _check_py(p, y)
    x = _logit(pa)
    # Design matrix [x, 1] for params theta = [a, b].
    X = np.column_stack([x, np.ones_like(x)])
    theta = np.zeros(2, dtype=float)
    ridge = l2 * np.eye(2)
    ridge[1, 1] = l2  # regularize both a and b lightly

    for _ in range(max_iter):
        z = X @ theta
        mu = _sigmoid(z)
        # Gradient of negative log-likelihood + L2: X^T (mu - y) + l2*theta
        grad = X.T @ (mu - ya) + l2 * theta
        w = mu * (1.0 - mu)
        # Hessian: X^T W X + l2 I  (W diagonal); add ridge for stability.
        H = X.T @ (X * w[:, None]) + ridge
        step = np.linalg.solve(H, grad)
        theta_new = theta - step
        if np.max(np.abs(step)) < tol:
            theta = theta_new
            break
        theta = theta_new

    a, b = float(theta[0]), float(theta[1])

    def transform(p_new: ArrayLike) -> np.ndarray:
        pn = np.asarray(p_new, dtype=float).ravel()
        if np.any(pn < 0.0) or np.any(pn > 1.0):
            raise ValueError("p_new must lie in [0, 1]")
        return _sigmoid(a * _logit(pn) + b)

    transform.coef_ = (a, b)  # type: ignore[attr-defined]  # expose for inspection
    return transform


# --------------------------------------------------------------------------- #
# Isotonic regression via Pool-Adjacent-Violators (PAVA)                      #
# --------------------------------------------------------------------------- #
def isotonic_fit(
    p: ArrayLike,
    y: ArrayLike,
) -> Callable[[ArrayLike], np.ndarray]:
    """Fit a non-decreasing calibration map p -> calibrated prob via PAVA.

    Pool-Adjacent-Violators solves the weighted isotonic least-squares problem
    (minimize sum w_i (g(x_i) - y_i)^2 subject to g non-decreasing in x). We sort by
    predicted p, run PAVA on the outcomes to get a monotone step function of fitted
    values, then expose a `transform` that maps any new p by piecewise-constant /
    linear interpolation over the fitted breakpoints (clamped at the ends).

    Isotonic is non-parametric and CAN fix non-monotone-in-magnitude miscalibration
    while enforcing monotonicity; it is more flexible than Platt but needs more data
    and can overfit small calibration sets (steps fit to noise). Prefer Platt when the
    calibration fold is small.

    Iron Law: fit on a held-out calibration fold disjoint from training and evaluation.

    Returns a CLOSURE transform(p_new) -> calibrated probs (non-decreasing in p_new).
    """
    pa, ya = _check_py(p, y)
    order = np.argsort(pa, kind="mergesort")  # stable
    xs = pa[order]
    ys = ya[order]

    # PAVA with weights. Maintain blocks of (sum_y, count); merge while a block's mean
    # is less than the previous block's mean (violates non-decreasing).
    block_sum = []
    block_cnt = []
    for val in ys:
        block_sum.append(float(val))
        block_cnt.append(1.0)
        # Merge backwards while previous mean > current mean.
        while (
            len(block_sum) >= 2
            and block_sum[-2] / block_cnt[-2] > block_sum[-1] / block_cnt[-1]
        ):
            s = block_sum.pop() + block_sum[-1]
            c = block_cnt.pop() + block_cnt[-1]
            block_sum[-1] = s
            block_cnt[-1] = c

    # Expand block means back to per-sample fitted values (already in sorted-x order).
    fitted = np.empty_like(ys)
    pos = 0
    for s, c in zip(block_sum, block_cnt):
        m = s / c
        k = int(round(c))
        fitted[pos : pos + k] = m
        pos += k

    # Build the interpolation breakpoints. Collapse duplicate x by keeping, for each
    # distinct x, the fitted value (fitted is constant within a tie because sort is
    # stable and PAVA pools equal-mean neighbors monotonically). Use the LAST fitted
    # value at each distinct x so the step map is right-continuous and non-decreasing.
    bp_x = []
    bp_y = []
    i = 0
    n = len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        bp_x.append(float(xs[i]))
        bp_y.append(float(fitted[j]))  # fitted is non-decreasing, take block value
        i = j + 1
    bp_x = np.asarray(bp_x, dtype=float)
    bp_y = np.asarray(bp_y, dtype=float)
    # Enforce numerical monotonicity (np.interp needs sorted x; y already monotone by PAVA).
    bp_y = np.maximum.accumulate(bp_y)

    def transform(p_new: ArrayLike) -> np.ndarray:
        pn = np.asarray(p_new, dtype=float).ravel()
        if np.any(pn < 0.0) or np.any(pn > 1.0):
            raise ValueError("p_new must lie in [0, 1]")
        if bp_x.size == 1:
            # Degenerate: a single distinct training x -> constant map.
            return np.full_like(pn, bp_y[0])
        # Linear interpolation between breakpoints; clamp to end values outside range.
        return np.interp(pn, bp_x, bp_y, left=bp_y[0], right=bp_y[-1])

    transform.breakpoints_ = (bp_x, bp_y)  # type: ignore[attr-defined]
    return transform


# --------------------------------------------------------------------------- #
# Self-tests                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    TOL = 1e-9

    # ----------------------------------------------------------------------- #
    # Scoring rules: analytic anchors                                         #
    # ----------------------------------------------------------------------- #
    assert abs(brier_score([1.0, 0.0], [1, 0]) - 0.0) < TOL          # perfect
    assert abs(brier_score([0.0, 1.0], [1, 0]) - 1.0) < TOL          # worst
    assert abs(brier_score([0.5, 0.5], [1, 0]) - 0.25) < TOL         # uninformative
    assert np.isfinite(log_loss([1.0], [0]))                          # clip keeps finite
    assert abs(log_loss([0.5, 0.5], [1, 0]) - (-np.log(0.5))) < 1e-12 # = ln 2

    # Input validation rejects bad inputs.
    for bad in (lambda: brier_score([1.2], [1]),      # p out of range
                lambda: brier_score([0.5], [2]),      # y not binary
                lambda: brier_score([0.5, 0.5], [1])):  # length mismatch
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # ----------------------------------------------------------------------- #
    # Reliability curve: counts conserved, perfectly-calibrated synthetic data #
    # ----------------------------------------------------------------------- #
    N = 40_000
    # Perfectly calibrated: draw p uniformly, then y ~ Bernoulli(p). By construction
    # observed freq in each bin -> mean predicted prob, so ECE -> 0 in the large sample.
    p_perfect = rng.uniform(0.0, 1.0, size=N)
    y_perfect = (rng.uniform(size=N) < p_perfect).astype(float)

    curve = reliability_curve(p_perfect, y_perfect, n_bins=10, strategy="uniform")
    assert curve["count"].sum() == N                       # every sample binned once
    assert np.all(curve["mean_pred"] >= curve["bin_lo"] - 1e-12)
    assert np.all(curve["mean_pred"] <= curve["bin_hi"] + 1e-12)
    ece_perfect = expected_calibration_error(p_perfect, y_perfect, n_bins=10)
    assert ece_perfect < 0.02, ece_perfect                 # near-zero for calibrated data
    mce_perfect = max_calibration_error(p_perfect, y_perfect, n_bins=10)
    assert mce_perfect < 0.05, mce_perfect

    # Quantile strategy also conserves counts and is near-calibrated here.
    cq = reliability_curve(p_perfect, y_perfect, n_bins=10, strategy="quantile")
    assert cq["count"].sum() == N
    assert expected_calibration_error(p_perfect, y_perfect, strategy="quantile") < 0.02

    # p == 1 must land in the final bin (closed on the right).
    cb = reliability_curve([0.0, 1.0], [0, 1], n_bins=4, strategy="uniform")
    assert cb["count"].sum() == 2

    # ----------------------------------------------------------------------- #
    # Brier decomposition: exact reconstruction with single-value bins         #
    # ----------------------------------------------------------------------- #
    # When each bin holds a single distinct forecast value, mean_pred == that value,
    # so reliability-resolution+uncertainty reconstructs the RAW Brier exactly.
    p_disc = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=5000)
    y_disc = (rng.uniform(size=5000) < p_disc).astype(float)
    # 5 uniform width-0.2 bins put each distinct value {0.1,..,0.9} in its OWN bin, so
    # mean_pred_b == the forecast value and the decomposition reconstructs Brier exactly.
    dec = brier_decomposition(p_disc, y_disc, n_bins=5, strategy="uniform")
    assert abs(dec["brier_binned"] - dec["brier"]) < 1e-9, (dec["brier_binned"], dec["brier"])
    # Term signs/ranges.
    assert dec["reliability"] >= -1e-12
    assert dec["resolution"] >= -1e-12
    assert 0.0 <= dec["uncertainty"] <= 0.25 + 1e-12
    # Uncertainty equals base-rate variance exactly.
    o_bar = float(np.mean(y_disc))
    assert abs(dec["uncertainty"] - o_bar * (1.0 - o_bar)) < TOL

    # A perfectly calibrated, perfectly resolving forecast: y deterministic from p in {0,1}.
    p_det = np.array([0.0, 0.0, 1.0, 1.0])
    y_det = np.array([0.0, 0.0, 1.0, 1.0])
    dd = brier_decomposition(p_det, y_det, n_bins=2, strategy="uniform")
    assert abs(dd["brier"]) < TOL
    assert abs(dd["reliability"]) < TOL                    # calibrated
    assert abs(dd["uncertainty"] - 0.25) < TOL             # base rate 0.5
    assert abs(dd["resolution"] - 0.25) < TOL              # fully resolved => res == unc

    # ----------------------------------------------------------------------- #
    # Overconfident generator: Platt AND isotonic reduce ECE and log-loss      #
    # ----------------------------------------------------------------------- #
    # True latent prob t; the model REPORTS an overconfident prob by pushing the logit
    # away from 0 (multiply logit by 1.8). Outcomes are drawn from the TRUE prob, so the
    # reported probs are systematically too extreme -> high ECE, fixable by recalibration.
    M = 20_000
    t = rng.uniform(0.05, 0.95, size=M)
    y_all = (rng.uniform(size=M) < t).astype(float)
    logit_t = np.log(t / (1.0 - t))
    p_over_all = 1.0 / (1.0 + np.exp(-(1.8 * logit_t)))     # overconfident reported prob

    # Split into DISJOINT calibration and test folds (Iron Law: never fit & eval on same).
    half = M // 2
    p_cal, y_cal = p_over_all[:half], y_all[:half]
    p_te, y_te = p_over_all[half:], y_all[half:]

    ece_raw = expected_calibration_error(p_te, y_te, n_bins=15)
    ll_raw = log_loss(p_te, y_te)
    brier_raw = brier_score(p_te, y_te)

    # Fit BOTH recalibrators on the calibration fold ONLY, evaluate on the test fold.
    platt = platt_scale(p_cal, y_cal)
    iso = isotonic_fit(p_cal, y_cal)

    p_platt = platt(p_te)
    p_iso = iso(p_te)

    ece_platt = expected_calibration_error(p_platt, y_te, n_bins=15)
    ece_iso = expected_calibration_error(p_iso, y_te, n_bins=15)
    ll_platt = log_loss(p_platt, y_te)
    ll_iso = log_loss(p_iso, y_te)

    # Recalibration must MEANINGFULLY cut the calibration error and the proper score,
    # out-of-sample (this is the whole point of the module).
    assert ece_platt < ece_raw * 0.6, (ece_raw, ece_platt)
    assert ece_iso < ece_raw * 0.6, (ece_raw, ece_iso)
    assert ll_platt < ll_raw, (ll_raw, ll_platt)
    assert ll_iso < ll_raw, (ll_raw, ll_iso)
    assert brier_score(p_platt, y_te) < brier_raw
    assert brier_score(p_iso, y_te) < brier_raw

    # Platt recovers ~the inverse scaling: a should be < 1 (it shrinks the inflated logit).
    a_hat, b_hat = platt.coef_
    assert a_hat < 1.0, a_hat
    # b near 0 because the distortion is symmetric (no base-rate shift).
    assert abs(b_hat) < 0.2, b_hat

    # ----------------------------------------------------------------------- #
    # Isotonic output is monotone non-decreasing and stays in [0, 1]           #
    # ----------------------------------------------------------------------- #
    grid = np.linspace(0.0, 1.0, 200)
    iso_grid = iso(grid)
    assert np.all(np.diff(iso_grid) >= -1e-12), "isotonic map must be non-decreasing"
    assert np.all(iso_grid >= -1e-12) and np.all(iso_grid <= 1.0 + 1e-12)
    # Platt map is also monotone increasing when a > 0 (a may be <1 but positive here).
    platt_grid = platt(grid)
    assert np.all(np.diff(platt_grid) >= -1e-12)

    # Isotonic exactly reproduces a clean monotone relationship (no violators to pool).
    x_mono = np.linspace(0.05, 0.95, 200)
    y_mono = (x_mono > 0.5).astype(float)  # step at 0.5; isotonic should track it
    iso2 = isotonic_fit(x_mono, y_mono)
    assert iso2(0.2) <= iso2(0.5) <= iso2(0.8)
    assert iso2(0.95) >= iso2(0.05)

    # Calibrating already-calibrated data should NOT materially worsen ECE.
    platt_pc = platt_scale(p_perfect[:half], y_perfect[:half])
    ece_after = expected_calibration_error(platt_pc(p_perfect[half:]), y_perfect[half:], n_bins=10)
    assert ece_after < 0.03, ece_after

    # transform input validation
    for f in (platt, iso):
        try:
            f([1.5])
            raise AssertionError("expected ValueError on out-of-range p_new")
        except ValueError:
            pass

    print("calibration.py: all self-tests passed.")
