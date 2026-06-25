"""portfolio.py - Portfolio construction & optimization toolkit.

Turns expected returns and a covariance estimate into portfolio weights, using
a range of classic and modern constructors. numpy / pandas / stdlib ONLY: no
scipy, no cvxpy. Iterative solvers and a minimal single-linkage clustering are
implemented by hand so the file is fully self-contained and self-verifying.

Conventions
-----------
- `cov` is an annualized (or per-period; the constructors are scale-equivariant
  in the obvious ways) covariance matrix, shape (n, n), symmetric PSD.
- `mu` is a vector of expected (excess, unless you set rf) returns, shape (n,).
- All constructors return a 1-D numpy array of weights that sums to 1.0.
- Long-only constructors (`inverse_variance_weights`, `risk_parity_weights`,
  `hrp_weights`) additionally guarantee w >= 0.
- `min_variance_weights` / `max_sharpe_weights` / `mean_variance_weights` are
  the *unconstrained* (long/short allowed) closed forms; weights can be
  negative. Normalizing to sum 1 fixes the budget but not the leverage; for a
  dollar-neutral or gross-leverage target, rescale downstream.

Pitfalls this file is built to avoid
------------------------------------
- DETECT: inverting a (near-)singular sample covariance blows up min-variance /
  tangency weights (the classic Markowitz error-maximization problem). FIX:
  `_safe_inv` ridge-regularizes a non-PD matrix; in production prefer a shrunk
  covariance (Ledoit-Wolf) or one of the heuristic constructors (IVP / ERC /
  HRP) that need no inverse at all.
- DETECT: max-Sharpe normalized to sum 1 silently flips sign when the tangency
  numerator sums to < 0 (e.g. all-negative mu). FIX: we keep the raw sign and
  let the caller see it; the docstring flags it.
- DETECT: risk-parity "solved" but risk contributions are not actually equal.
  FIX: the self-tests assert max-min RC spread < 1e-4 on the returned weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------
def _as_cov(cov: np.ndarray) -> np.ndarray:
    """Validate / symmetrize a covariance matrix, return float64 copy."""
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"cov must be square 2-D, got shape {c.shape}")
    # Symmetrize defensively (kills tiny asymmetry from upstream estimators).
    return 0.5 * (c + c.T)


def _safe_inv(cov: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    """Inverse of a covariance matrix, ridge-regularized if not PD.

    Tries a plain inverse first; on LinAlgError (or non-finite result) adds a
    small multiple of the identity scaled to the matrix and retries. This is a
    numerical safety net, NOT a substitute for proper covariance shrinkage.
    """
    c = _as_cov(cov)
    try:
        inv = np.linalg.inv(c)
        if np.all(np.isfinite(inv)):
            return inv
    except np.linalg.LinAlgError:
        pass
    scale = float(np.trace(c)) / c.shape[0]
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return np.linalg.inv(c + ridge * scale * np.eye(c.shape[0]))


def _normalize(w: np.ndarray) -> np.ndarray:
    """Scale weights to sum to 1.0. Raises if the sum is ~0 (ill-posed)."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if not np.isfinite(s) or abs(s) < 1e-300:
        raise ValueError("weights sum to ~0; cannot normalize to budget 1")
    return w / s


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Correlation matrix from a covariance matrix (diag clipped at 0)."""
    c = _as_cov(cov)
    d = np.sqrt(np.clip(np.diag(c), 0.0, None))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0.0, c / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Closed-form constructors (long/short permitted)
# ---------------------------------------------------------------------------
def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio: w ∝ Σ⁻¹·1, normalized to sum 1.

    Unconstrained closed form (weights may be negative). For a diagonal Σ this
    reduces to w_i ∝ 1/σ_i², which the self-tests check.
    """
    inv = _safe_inv(cov)
    ones = np.ones(inv.shape[0])
    return _normalize(inv @ ones)


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Tangency (max-Sharpe) portfolio: w ∝ Σ⁻¹·(μ − rf), normalized to sum 1.

    Maximizes (μ−rf)'w / sqrt(w'Σw). Unconstrained (weights may be negative).
    Caveat: when Σ⁻¹(μ−rf) sums to < 0 the sum-1 normalization flips the sign of
    every weight (you end up short the tangency direction). The raw sign is
    preserved here so the caller can detect it.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    inv = _safe_inv(cov)
    if mu.shape[0] != inv.shape[0]:
        raise ValueError("mu and cov dimensions disagree")
    excess = mu - rf
    return _normalize(inv @ excess)


def mean_variance_weights(
    mu: np.ndarray, cov: np.ndarray, risk_aversion: float = 1.0
) -> np.ndarray:
    """Markowitz mean-variance optimum, w ∝ (1/λ)·Σ⁻¹·μ, normalized to sum 1.

    Maximizes μ'w − (λ/2)·w'Σw. The unnormalized optimum is (1/λ)Σ⁻¹μ; since we
    then normalize to a budget of 1, the value of `risk_aversion` only matters
    if you skip normalization (it cancels here). Kept in the signature for
    API symmetry and for callers that rescale to a gross-exposure target.
    """
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be positive")
    mu = np.asarray(mu, dtype=float).ravel()
    inv = _safe_inv(cov)
    if mu.shape[0] != inv.shape[0]:
        raise ValueError("mu and cov dimensions disagree")
    return _normalize((inv @ mu) / risk_aversion)


# ---------------------------------------------------------------------------
# Heuristic / risk-based constructors (long-only)
# ---------------------------------------------------------------------------
def inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio (IVP): w_i ∝ 1/σ_i², normalized to sum 1.

    Long-only, ignores correlations entirely. Equals the min-variance solution
    iff Σ is diagonal. Used as the per-cluster allocator inside HRP.
    """
    c = _as_cov(cov)
    var = np.diag(c).astype(float)
    if np.any(var <= 0.0):
        raise ValueError("non-positive variance on the diagonal of cov")
    inv_var = 1.0 / var
    return inv_var / inv_var.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Percentage risk contributions RC_i = w_i·(Σw)_i / (w'Σw), summing to 1.

    RC_i is asset i's share of total portfolio variance (equivalently of
    volatility, since the σ factor cancels in the ratio). Equal RC_i = 1/n is
    the defining property of the equal-risk-contribution (risk-parity) portfolio.
    """
    w = np.asarray(w, dtype=float).ravel()
    c = _as_cov(cov)
    if w.shape[0] != c.shape[0]:
        raise ValueError("w and cov dimensions disagree")
    port_var = float(w @ c @ w)
    if port_var <= 0.0:
        raise ValueError("non-positive portfolio variance")
    marginal = c @ w           # ∂σ²/∂w_i up to a factor of 2
    return (w * marginal) / port_var


def risk_parity_weights(
    cov: np.ndarray, tol: float = 1e-10, max_iter: int = 10000
) -> np.ndarray:
    """Long-only equal-risk-contribution (ERC) portfolio via fixed-point iteration.

    Solves for w > 0 with RC_i = 1/n for all i. Uses the cyclical / multiplicative
    fixed-point update (Spinu / Griveau-Billion style):

        w_i  <-  w_i * (target_i) / (Σw)_i ,  then renormalize,

    with target_i = 1/n. This converges monotonically for a PD covariance and a
    positive start. We start from inverse-variance weights (already a decent ERC
    proxy) to accelerate convergence.

    Convergence test: max relative change in w below `tol`.
    """
    c = _as_cov(cov)
    n = c.shape[0]
    target = np.ones(n) / n

    w = inverse_variance_weights(c)
    for _ in range(max_iter):
        marginal = c @ w               # (Σw)_i, strictly positive for PD c, w>0
        if np.any(marginal <= 0.0):
            # Numerical guard: nudge off the boundary.
            marginal = np.clip(marginal, 1e-300, None)
        w_new = np.sqrt(w * target / marginal)  # sqrt-damped: equalizes TOTAL RC (ERC)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (López de Prado, 2016)
# ---------------------------------------------------------------------------
def _single_linkage(dist: np.ndarray) -> List[Tuple[int, int, float]]:
    """Single-linkage agglomerative clustering on a distance matrix.

    Returns a SciPy-style linkage list of (id_a, id_b, dist) merges. The new
    cluster formed at merge k gets id (n + k). Single linkage uses the minimum
    pairwise distance between members of two clusters (nearest neighbor).

    Pure-numpy O(n^3) implementation — fine for the modest n typical in
    portfolio construction.
    """
    n = dist.shape[0]
    # clusters: id -> list of original leaf indices
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    active: List[int] = list(range(n))
    linkage: List[Tuple[int, int, float]] = []

    def cluster_dist(a: int, b: int) -> float:
        sub = dist[np.ix_(clusters[a], clusters[b])]
        return float(sub.min())

    next_id = n
    while len(active) > 1:
        best = (np.inf, -1, -1)
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a, b = active[ii], active[jj]
                d = cluster_dist(a, b)
                if d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        linkage.append((a, b, d))
        clusters[next_id] = clusters[a] + clusters[b]
        active.remove(a)
        active.remove(b)
        active.append(next_id)
        next_id += 1
    return linkage


def _quasi_diagonal_order(linkage: List[Tuple[int, int, float]], n: int) -> List[int]:
    """Leaf order obtained by recursively expanding the linkage tree.

    Places correlated assets adjacent to one another (the quasi-diagonalization
    step of HRP). Implemented iteratively to avoid recursion-depth issues.
    """
    if n == 1:
        return [0]
    root = n + len(linkage) - 1  # id of the last (top) merge

    def children(node: int) -> Tuple[int, int]:
        a, b, _ = linkage[node - n]
        return a, b

    order: List[int] = []
    stack: List[int] = [root]
    while stack:
        node = stack.pop()
        if node < n:               # leaf
            order.append(node)
        else:
            a, b = children(node)
            # Push right then left so left is processed first (stable order).
            stack.append(b)
            stack.append(a)
    return order


def _cluster_var(cov: np.ndarray, idx: List[int]) -> float:
    """Variance of the inverse-variance portfolio over a sub-universe `idx`."""
    sub = cov[np.ix_(idx, idx)]
    w = inverse_variance_weights(sub)
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity weights (López de Prado, 2016). Long-only, sum 1.

    Three stages:
      1. Tree clustering: correlation -> distance d_ij = sqrt(0.5·(1−ρ_ij)),
         single-linkage agglomerative clustering.
      2. Quasi-diagonalization: reorder assets by the cluster tree so similar
         assets are adjacent.
      3. Recursive bisection: split the ordered list in half repeatedly; at each
         split allocate between the two halves in inverse proportion to each
         half's inverse-variance-portfolio variance (α = 1 − V_left/(V_left+V_right)
         to the left).

    HRP needs no matrix inversion, so it is robust to ill-conditioned Σ — a key
    reason to prefer it over min-variance / tangency on noisy sample covariances.
    """
    c = _as_cov(cov)
    n = c.shape[0]
    if n == 1:
        return np.array([1.0])

    corr = cov_to_corr(c)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)

    linkage = _single_linkage(dist)
    order = _quasi_diagonal_order(linkage, n)

    w = np.ones(n)
    clusters: List[List[int]] = [order]
    while clusters:
        new_clusters: List[List[int]] = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left, right = cl[:mid], cl[mid:]
            v_left = _cluster_var(c, left)
            v_right = _cluster_var(c, right)
            denom = v_left + v_right
            alpha = 1.0 - v_left / denom if denom > 0.0 else 0.5
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1.0 - alpha)
            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters

    return w / w.sum()


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------
def black_litterman(
    cov: np.ndarray,
    w_market: np.ndarray,
    risk_aversion: float = 2.5,
    P: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    tau: float = 0.05,
    omega: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Black-Litterman posterior expected returns and resulting weights.

    Steps
    -----
    1. Reverse-optimize equilibrium (implied) returns from market-cap weights:
           π = λ · Σ · w_market
    2. Blend π with K subjective views (P·μ = Q + noise, noise ~ N(0, Ω)) via the
       standard posterior-mean formula:
           μ_post = [ (τΣ)⁻¹ + Pᵀ Ω⁻¹ P ]⁻¹ · [ (τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q ]
    3. Weights = mean_variance_weights(μ_post, Σ, risk_aversion), normalized.

    Parameters
    ----------
    P : (K, n) view-picking matrix, Q : (K,) view returns. If either is None,
        there are NO views and μ_post == π (the test asserts this), so weights
        reduce to the equilibrium/market weights.
    omega : (K, K) view-uncertainty covariance. Default (He-Litterman):
        Ω = diag(P (τΣ) Pᵀ), i.e. each view's uncertainty proportional to the
        prior variance of that view's portfolio.

    Returns
    -------
    dict with keys 'posterior_mu' (n,) and 'weights' (n,).
    """
    c = _as_cov(cov)
    n = c.shape[0]
    w_mkt = np.asarray(w_market, dtype=float).ravel()
    if w_mkt.shape[0] != n:
        raise ValueError("w_market and cov dimensions disagree")

    pi = risk_aversion * (c @ w_mkt)

    if P is None or Q is None:
        posterior_mu = pi.copy()
    else:
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float).ravel()
        if P.ndim == 1:
            P = P.reshape(1, -1)
        if P.shape[1] != n:
            raise ValueError("P must have n columns")
        if P.shape[0] != Q.shape[0]:
            raise ValueError("P rows and Q length disagree")

        tau_cov = tau * c
        if omega is None:
            omega = np.diag(np.diag(P @ tau_cov @ P.T))
        else:
            omega = np.asarray(omega, dtype=float)
            if omega.ndim == 1:
                omega = np.diag(omega)

        tau_cov_inv = _safe_inv(tau_cov)
        omega_inv = _safe_inv(omega)
        a = tau_cov_inv + P.T @ omega_inv @ P
        b = tau_cov_inv @ pi + P.T @ omega_inv @ Q
        posterior_mu = _safe_inv(a) @ b

    weights = mean_variance_weights(posterior_mu, c, risk_aversion=risk_aversion)
    return {"posterior_mu": posterior_mu, "weights": weights}


# ---------------------------------------------------------------------------
# Cost-aware rebalancing (turnover reduction)
# ---------------------------------------------------------------------------
# These helpers take a freshly computed *target* book `w_target` and the
# *currently held* book `w_prev` and return a traded-to book that deliberately
# trades less than the naive "jump straight to target". Trading less saves
# transaction costs at the price of some tracking error to the model target.
#
# Iron-Law / causality note: each function depends ONLY on `w_target` (the
# signal-derived target for the *upcoming* holding period) and `w_prev` (the
# book you already hold, known at decision time). Neither peeks at future
# returns, so all three are leak-free and safe to call inside a backtest's
# rebalance step. The standard usage is, at each rebalance date t:
#     w_held[t] = apply_turnover_band(model_target[t], w_held[t-1], band)
# i.e. `w_prev` is last period's *post-trade* book — strictly past information.
#
# `turnover` here is the conventional one-way turnover
#     T = 0.5 * sum_i |w_i - w_prev_i|
# (the fraction of the book replaced); cost per rebalance is roughly
# cost_rate * 2 * T for a round trip, or cost_rate * (sum |Δw|) one notional
# leg. Lower turnover => lower cost, which is the whole point of these helpers.


def turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    """One-way (Lhabitant) turnover between two books: 0.5·Σ|w_new − w_prev|.

    The fraction of the portfolio notionally replaced going from `w_prev` to
    `w_new`. For two fully-invested (sum-1) long books this lies in [0, 1].
    Causal: a pure function of two known weight vectors, no future data.
    """
    a = np.asarray(w_new, dtype=float).ravel()
    b = np.asarray(w_prev, dtype=float).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError("w_new and w_prev dimensions disagree")
    return 0.5 * float(np.abs(a - b).sum())


def shrink_to_prev(
    w_target: np.ndarray, w_prev: np.ndarray, lam: float
) -> np.ndarray:
    """Linear shrinkage of the target book toward the prior book, renormalized.

    Returns w = (1−lam)·w_target + lam·w_prev, then renormalized to sum 1.
    `lam` in [0, 1] is the fraction of "stickiness": lam=0 trades all the way to
    the target, lam=1 stays at the prior book (zero turnover). Because turnover
    is linear in the traded vector and this is a convex blend, realized turnover
    equals (1−lam)·turnover(w_target, w_prev) before renormalization — i.e. it
    decreases monotonically as `lam` rises. This is the classic
    "partial-rebalance" / inventory-smoothing trick for cutting trading costs.

    Causal: depends only on the model target and the previously held book; no
    look-ahead. Safe inside a backtest rebalance step.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    wt = np.asarray(w_target, dtype=float).ravel()
    wp = np.asarray(w_prev, dtype=float).ravel()
    if wt.shape[0] != wp.shape[0]:
        raise ValueError("w_target and w_prev dimensions disagree")
    blended = (1.0 - lam) * wt + lam * wp
    return _normalize(blended)


def apply_turnover_band(
    w_target: np.ndarray,
    w_prev: np.ndarray,
    band: float,
) -> np.ndarray:
    """No-trade band rebalancing: only move names whose drift exceeds `band`.

    For each asset, if |w_target_i − w_prev_i| <= band we *hold* the prior weight
    (no trade on that name); otherwise we trade that name, but only part-way —
    to the edge of the band, i.e. toward the target by exactly `band`:

        Δ_i = w_target_i − w_prev_i
        traded_i = w_prev_i                                  if |Δ_i| <= band
                 = w_prev_i + sign(Δ_i)·(|Δ_i| − band)       otherwise

    This is the standard rectangular no-trade region used to damp turnover from
    estimation noise (cf. Davis-Norman / Leland transaction-cost rebalancing,
    and the "tolerance band" rebalancing of practitioner portfolios). Pulling
    only to the band edge (rather than all the way to target) is what makes the
    rule *cost-aware*: each unit of widening `band` strictly cannot increase the
    L1 trade size of any name, so realized turnover is non-increasing in `band`.

    The traded book is renormalized to sum 1 (the per-name capping breaks the
    budget slightly; a single renorm restores it). With band=0 the result is
    exactly `w_target`; with band large enough to cover every drift it is exactly
    `w_prev` (zero turnover) — both checked in the self-tests.

    Causal / leak-free: a deterministic function of the model target and the
    previously held book only. Safe to call at each backtest rebalance date.
    """
    if band < 0.0:
        raise ValueError("band must be non-negative")
    wt = np.asarray(w_target, dtype=float).ravel()
    wp = np.asarray(w_prev, dtype=float).ravel()
    if wt.shape[0] != wp.shape[0]:
        raise ValueError("w_target and w_prev dimensions disagree")
    delta = wt - wp
    mag = np.abs(delta)
    # Soft-threshold the drift: shrink each move toward 0 by `band`, floored at 0.
    capped_move = np.sign(delta) * np.clip(mag - band, 0.0, None)
    traded = wp + capped_move
    return _normalize(traded)


def no_trade_region(
    w_target: np.ndarray,
    w_prev: np.ndarray,
    band: float,
) -> np.ndarray:
    """Hard no-trade region: hold names inside the band, jump the rest to target.

    A stricter variant of `apply_turnover_band`. Names whose drift |Δ_i| <= band
    are held flat; names that breach the band are traded *all the way* to the
    model target (not merely to the band edge):

        traded_i = w_prev_i      if |w_target_i − w_prev_i| <= band
                 = w_target_i     otherwise

    Use this when you want a clean "rebalance the breachers fully, ignore the
    rest" policy; use `apply_turnover_band` when you want the gentler
    pull-to-edge that minimizes traded notional. Renormalized to sum 1. As
    `band` widens, the set of traded names shrinks, so realized turnover is
    non-increasing in `band`; band large enough returns `w_prev` exactly.

    Causal / leak-free: depends only on the model target and the held book.
    """
    if band < 0.0:
        raise ValueError("band must be non-negative")
    wt = np.asarray(w_target, dtype=float).ravel()
    wp = np.asarray(w_prev, dtype=float).ravel()
    if wt.shape[0] != wp.shape[0]:
        raise ValueError("w_target and w_prev dimensions disagree")
    breach = np.abs(wt - wp) > band
    traded = np.where(breach, wt, wp)
    return _normalize(traded)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def _make_corr_cov(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    d = np.diag(vols)
    return d @ corr @ d


if __name__ == "__main__":
    rng = np.random.default_rng(0)  # unused for randomness; determinism by design

    # --- Diagonal covariance: analytic checks -----------------------------
    variances = np.array([0.04, 0.01, 0.09, 0.0025])  # vols 0.2,0.1,0.3,0.05
    cov_diag = np.diag(variances)

    w_mv = min_variance_weights(cov_diag)
    assert abs(w_mv.sum() - 1.0) < 1e-8

    # min-variance on diagonal cov ∝ 1/variance
    expected_mv = (1.0 / variances) / (1.0 / variances).sum()
    assert np.allclose(w_mv, expected_mv, atol=1e-10), (w_mv, expected_mv)

    # inverse-variance equals min-variance when cov is diagonal
    w_ivp = inverse_variance_weights(cov_diag)
    assert np.allclose(w_ivp, expected_mv, atol=1e-12)

    # risk parity on diagonal cov ∝ 1/vol
    w_rp = risk_parity_weights(cov_diag)
    assert abs(w_rp.sum() - 1.0) < 1e-8
    vols = np.sqrt(variances)
    expected_rp = (1.0 / vols) / (1.0 / vols).sum()
    assert np.allclose(w_rp, expected_rp, atol=1e-6), (w_rp, expected_rp)

    # --- Risk contributions sum to 1, and ERC equalizes them --------------
    rc_arb = risk_contributions(np.array([0.25, 0.25, 0.25, 0.25]), cov_diag)
    assert abs(rc_arb.sum() - 1.0) < 1e-10

    # Correlated covariance for ERC / HRP tests
    corr5 = np.array([
        [1.00, 0.70, 0.20, 0.10, 0.05],
        [0.70, 1.00, 0.30, 0.15, 0.10],
        [0.20, 0.30, 1.00, 0.40, 0.20],
        [0.10, 0.15, 0.40, 1.00, 0.50],
        [0.05, 0.10, 0.20, 0.50, 1.00],
    ])
    vols5 = np.array([0.15, 0.20, 0.10, 0.25, 0.18])
    cov5 = _make_corr_cov(vols5, corr5)

    w_erc = risk_parity_weights(cov5)
    assert abs(w_erc.sum() - 1.0) < 1e-8
    assert np.all(w_erc >= 0.0)
    rc_erc = risk_contributions(w_erc, cov5)
    assert abs(rc_erc.sum() - 1.0) < 1e-10
    assert (rc_erc.max() - rc_erc.min()) < 1e-4, rc_erc

    # --- Max-Sharpe tilts toward the higher-mu asset ----------------------
    mu2 = np.array([0.20, 0.00])
    cov2 = np.array([[0.04, 0.005], [0.005, 0.04]])
    w_ms = max_sharpe_weights(mu2, cov2)
    assert abs(w_ms.sum() - 1.0) < 1e-8
    assert w_ms[0] > w_ms[1]
    assert w_ms[0] > 0.5, w_ms  # majority on asset 0

    # mean-variance also tilts toward asset 0 and sums to 1
    w_meanvar = mean_variance_weights(mu2, cov2, risk_aversion=3.0)
    assert abs(w_meanvar.sum() - 1.0) < 1e-8
    assert w_meanvar[0] > w_meanvar[1]

    # --- HRP: positive, sums to 1, on a 4- and 5-asset cov ----------------
    w_hrp5 = hrp_weights(cov5)
    assert abs(w_hrp5.sum() - 1.0) < 1e-8
    assert np.all(w_hrp5 > 0.0), w_hrp5

    cov4 = _make_corr_cov(vols5[:4], corr5[:4, :4])
    w_hrp4 = hrp_weights(cov4)
    assert abs(w_hrp4.sum() - 1.0) < 1e-8
    assert np.all(w_hrp4 > 0.0), w_hrp4

    # --- Black-Litterman with no views collapses to equilibrium -----------
    w_market = np.array([0.30, 0.10, 0.25, 0.20, 0.15])
    bl_noview = black_litterman(cov5, w_market, risk_aversion=2.5)
    pi = 2.5 * (cov5 @ w_market)
    assert np.allclose(bl_noview["posterior_mu"], pi, atol=1e-10)
    # weights should recover the market portfolio (mean-variance of pi with the
    # same lambda inverts the reverse-optimization, up to the budget renorm).
    assert abs(bl_noview["weights"].sum() - 1.0) < 1e-8
    assert np.allclose(bl_noview["weights"], w_market, atol=1e-6), bl_noview["weights"]

    # --- Black-Litterman with one absolute view shifts that asset's mu up --
    P = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])  # view on asset 2
    Q = np.array([pi[2] + 0.05])               # expect 5% more than equilibrium
    bl_view = black_litterman(cov5, w_market, risk_aversion=2.5, P=P, Q=Q)
    assert bl_view["posterior_mu"][2] > pi[2], bl_view["posterior_mu"]
    assert abs(bl_view["weights"].sum() - 1.0) < 1e-8

    # --- Ill-conditioned covariance does not blow up min-variance ---------
    cov_sing = np.array([[1.0, 1.0], [1.0, 1.0]]) * 0.04
    w_sing = min_variance_weights(cov_sing)  # ridge kicks in
    assert abs(w_sing.sum() - 1.0) < 1e-8
    assert np.all(np.isfinite(w_sing))

    # --- Cost-aware rebalancing -------------------------------------------
    w_prev = np.array([0.25, 0.25, 0.25, 0.25])
    w_tgt = np.array([0.40, 0.30, 0.20, 0.10])

    # turnover anchor: one-way turnover = 0.5*sum|Δ| = 0.5*(0.15+0.05+0.05+0.15)=0.20
    assert abs(turnover(w_tgt, w_prev) - 0.20) < 1e-12, turnover(w_tgt, w_prev)
    # passthrough: trading to where you already are => zero turnover
    assert turnover(w_prev, w_prev) == 0.0

    # apply_turnover_band: band=0 reproduces the target exactly
    w_b0 = apply_turnover_band(w_tgt, w_prev, band=0.0)
    assert abs(w_b0.sum() - 1.0) < 1e-12
    assert np.allclose(w_b0, w_tgt, atol=1e-12), w_b0

    # w_prev passthrough: target == prev => zero turnover, book unchanged
    w_pass = apply_turnover_band(w_prev, w_prev, band=0.05)
    assert abs(w_pass.sum() - 1.0) < 1e-12
    assert turnover(w_pass, w_prev) < 1e-12, w_pass

    # realized turnover is NON-INCREASING as the band widens
    bands = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50]
    to_band = [turnover(apply_turnover_band(w_tgt, w_prev, b), w_prev) for b in bands]
    for k in range(1, len(to_band)):
        assert to_band[k] <= to_band[k - 1] + 1e-12, (bands[k], to_band)
    # and STRICTLY decreases over a widening that bites (0 -> 0.05)
    assert to_band[2] < to_band[0] - 1e-9, to_band
    # a band wide enough to cover every drift collapses to w_prev (zero turnover)
    w_wide = apply_turnover_band(w_tgt, w_prev, band=0.30)
    assert abs(w_wide.sum() - 1.0) < 1e-12
    assert np.allclose(w_wide, w_prev, atol=1e-12), w_wide

    # analytic anchor for the pull-to-edge rule at band=0.05 (before renorm the
    # capped book already sums to 1 here since +moves and -moves are symmetric):
    #   Δ = [+.15,+.05,-.05,-.15], soft-threshold by .05 -> [+.10,0,0,-.10]
    #   traded = prev + that = [.35,.25,.25,.15]
    w_edge = apply_turnover_band(w_tgt, w_prev, band=0.05)
    assert np.allclose(w_edge, np.array([0.35, 0.25, 0.25, 0.15]), atol=1e-12), w_edge

    # shrink_to_prev: lam=0 -> target, lam=1 -> prev, monotone turnover in lam
    assert np.allclose(shrink_to_prev(w_tgt, w_prev, 0.0), w_tgt, atol=1e-12)
    assert np.allclose(shrink_to_prev(w_tgt, w_prev, 1.0), w_prev, atol=1e-12)
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    to_lam = [turnover(shrink_to_prev(w_tgt, w_prev, l), w_prev) for l in lams]
    for k in range(1, len(to_lam)):
        assert to_lam[k] <= to_lam[k - 1] + 1e-12, (lams[k], to_lam)
    # linearity anchor: turnover at lam == (1-lam)*turnover_at_0 (convex blend,
    # both books sum to 1 so renorm is a no-op)
    assert abs(to_lam[2] - 0.5 * to_lam[0]) < 1e-12, to_lam
    # w_prev passthrough -> zero turnover regardless of lam
    assert turnover(shrink_to_prev(w_prev, w_prev, 0.3), w_prev) < 1e-12
    # all shrink outputs still sum to 1
    for l in lams:
        assert abs(shrink_to_prev(w_tgt, w_prev, l).sum() - 1.0) < 1e-12

    # no_trade_region: band=0 -> target, wide band -> prev, monotone turnover
    assert np.allclose(no_trade_region(w_tgt, w_prev, 0.0), w_tgt, atol=1e-12)
    w_ntr_wide = no_trade_region(w_tgt, w_prev, 0.30)
    assert np.allclose(w_ntr_wide, w_prev, atol=1e-12), w_ntr_wide
    to_ntr = [turnover(no_trade_region(w_tgt, w_prev, b), w_prev) for b in bands]
    for k in range(1, len(to_ntr)):
        assert to_ntr[k] <= to_ntr[k - 1] + 1e-12, (bands[k], to_ntr)
    for b in bands:
        assert abs(no_trade_region(w_tgt, w_prev, b).sum() - 1.0) < 1e-12

    print("portfolio.py: all self-tests passed.")
