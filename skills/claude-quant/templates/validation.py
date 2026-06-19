"""
validation.py - leak-free cross-validation for time-series / financial ML, plus a
constant-correlation covariance shrinkage estimator.

WHY plain KFold/shuffle leaks on financial data:
  (a) labels built from a forward window of length `label_horizon` overlap
      neighbouring samples, so a train sample next to the test fold shares
      information with it; and
  (b) serial correlation bleeds train into test across the fold boundary.

PurgedKFold removes ("purges") train samples whose label window overlaps the test
fold, and "embargoes" a buffer of bars immediately after each test block.
CombinatorialPurgedKFold (Lopez de Prado's CPCV) tests on every combination of
`n_test_groups` of the `n_splits` contiguous groups, producing many distinct
backtest paths whose distribution feeds PBO/CSCV (see references/stats-risk.md).

constant_correlation_shrinkage implements Ledoit-Wolf (2003), "Honey, I Shrunk the
Sample Covariance Matrix" - the CONSTANT-CORRELATION target appropriate for asset
returns. (sklearn.covariance.LedoitWolf uses the scaled-IDENTITY target of LW 2004,
which is usually worse for return covariances.)

numpy-only, self-testing. Run: python validation.py
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np


def _n_samples(X) -> int:
    return int(X.shape[0]) if hasattr(X, "shape") else len(X)


def _purged_train_mask(n: int, test_idx, embargo: int, label_horizon: int) -> np.ndarray:
    """Boolean train mask of length n. Drops the test indices, then for each
    CONTIGUOUS test block [a, b] drops the purge+embargo region [a-label_horizon,
    b+embargo]. `test_idx` may be non-contiguous (CPCV) - each run is handled."""
    keep = np.ones(n, dtype=bool)
    s = np.sort(np.asarray(test_idx, dtype=int))
    if s.size == 0:
        return keep
    keep[s] = False
    breaks = np.where(np.diff(s) > 1)[0]
    starts = np.r_[s[0], s[breaks + 1]]
    ends = np.r_[s[breaks], s[-1]]
    for a, b in zip(starts, ends):
        # Purge label_horizon on BOTH sides: a train label can reach forward into
        # the test span (left), and a test label can reach forward into train
        # features (right). Embargo adds an extra right-side buffer for serial corr.
        lo = max(a - label_horizon, 0)
        hi = min(b + label_horizon + embargo + 1, n)
        keep[lo:hi] = False
    return keep


class PurgedKFold:
    """K-fold CV with purging + embargo for time-ordered data. sklearn-compatible
    API (.split / .get_n_splits) without importing sklearn. Folds are contiguous
    in time order; do NOT shuffle the data first.

    label_horizon: forward window (in bars) used to build the label y. Set it equal
    to the holding period / forward-return horizon of your target.
    embargo_pct: fraction of the sample embargoed after each test fold.
    """

    def __init__(self, n_splits: int = 5, *, embargo_pct: float = 0.01, label_horizon: int = 1):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.label_horizon = label_horizon

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _n_samples(X)
        embargo = int(round(n * self.embargo_pct))
        for fold in np.array_split(np.arange(n), self.n_splits):
            test_idx = np.asarray(fold, dtype=int)
            mask = _purged_train_mask(n, test_idx, embargo, self.label_horizon)
            yield np.flatnonzero(mask), test_idx


class CombinatorialPurgedKFold:
    """Combinatorial Purged Cross-Validation (Lopez de Prado). Partitions the data
    into `n_splits` contiguous groups and tests on every combination of
    `n_test_groups` of them, purging+embargoing around EACH test group.

    Number of splits = C(n_splits, n_test_groups). Number of distinct backtest
    PATHS the splits reconstruct = C(n_splits-1, n_test_groups-1) per group.
    """

    def __init__(self, n_splits: int = 6, n_test_groups: int = 2, *,
                 embargo_pct: float = 0.01, label_horizon: int = 1):
        if not (1 <= n_test_groups < n_splits):
            raise ValueError("require 1 <= n_test_groups < n_splits")
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.label_horizon = label_horizon

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return comb(self.n_splits, self.n_test_groups)

    def n_paths(self) -> int:
        return comb(self.n_splits - 1, self.n_test_groups - 1)

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _n_samples(X)
        embargo = int(round(n * self.embargo_pct))
        blocks = np.array_split(np.arange(n), self.n_splits)
        for combo in combinations(range(self.n_splits), self.n_test_groups):
            test_idx = np.concatenate([blocks[g] for g in combo]).astype(int)
            test_idx.sort()
            mask = _purged_train_mask(n, test_idx, embargo, self.label_horizon)
            yield np.flatnonzero(mask), test_idx


def constant_correlation_shrinkage(returns) -> dict:
    """Ledoit-Wolf (2003) shrinkage of the sample covariance toward the
    constant-correlation target.

    returns: (T, N) array/DataFrame of period returns (rows=observations).
    Returns dict with keys: covariance (NxN shrunk estimate), shrinkage (delta in
    [0,1]), target (F), sample (S, 1/T normalised), avg_correlation (rbar).
    """
    X = np.asarray(returns, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T                       # LW use 1/T (MLE), not ddof=1
    var = np.diag(S).copy()
    if N == 1:
        return {"covariance": S, "shrinkage": 0.0, "target": S.copy(),
                "sample": S, "avg_correlation": float("nan")}
    std = np.sqrt(var)
    outer_std = np.outer(std, std)
    R = S / outer_std
    iu = np.triu_indices(N, k=1)
    rbar = float(R[iu].mean())
    F = rbar * outer_std                       # constant-correlation target
    np.fill_diagonal(F, var)

    # pi_hat: sum of asymptotic variances of sample cov entries
    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / T - S ** 2        # E[x_i^2 x_j^2] - s_ij^2
    pi_hat = float(pi_mat.sum())

    # rho_hat: asymptotic covariance between sample cov and the target
    term1 = ((Xc ** 3).T @ Xc) / T             # E[x_i^3 x_j]
    term2 = var[:, None] * S                    # s_ii * s_ij
    theta = term1 - term2
    np.fill_diagonal(theta, 0.0)
    weight = np.outer(1.0 / std, std)          # sqrt(s_jj/s_ii)
    rho_hat = float(np.diag(pi_mat).sum() + rbar * np.sum(weight * theta))

    # gamma_hat: misspecification of the target
    gamma_hat = float(np.sum((S - F) ** 2))

    if gamma_hat <= 0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa / T))

    sigma = delta * F + (1.0 - delta) * S
    return {"covariance": sigma, "shrinkage": float(delta), "target": F,
            "sample": S, "avg_correlation": rbar}


# --------------------------------------------------------------------------- #
# Self-tests - run: python validation.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- PurgedKFold ----
    n = 100
    X = np.arange(n)
    pk = PurgedKFold(n_splits=5, embargo_pct=0.02, label_horizon=3)
    assert pk.get_n_splits() == 5
    seen_test = []
    for tr, te in pk.split(X):
        # (1) train and test never intersect
        assert np.intersect1d(tr, te).size == 0
        # (2) no train index within label_horizon of any test index
        if tr.size and te.size:
            d = np.abs(tr[:, None] - te[None, :]).min()
            assert d > 3, "purge failed: train too close to test"
        # (3) embargo removed bars right after the test block
        embargo = int(round(n * 0.02))
        b = te.max()
        post = np.arange(b + 1, min(b + 1 + embargo, n))
        assert np.intersect1d(tr, post).size == 0, "embargo failed"
        seen_test.append(te)
    # test folds tile the whole series
    assert np.array_equal(np.sort(np.concatenate(seen_test)), np.arange(n))

    # (4) embargo_pct=0, label_horizon=0 -> plain KFold complements
    pk0 = PurgedKFold(n_splits=5, embargo_pct=0.0, label_horizon=0)
    for tr, te in pk0.split(X):
        assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(n))

    # ---- CombinatorialPurgedKFold ----
    cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_groups=2,
                                    embargo_pct=0.01, label_horizon=2)
    assert cpcv.get_n_splits() == comb(6, 2) == 15
    assert cpcv.n_paths() == comb(5, 1) == 5
    n_splits_seen = 0
    for tr, te in cpcv.split(X):
        n_splits_seen += 1
        assert np.intersect1d(tr, te).size == 0
        if tr.size and te.size:
            assert np.abs(tr[:, None] - te[None, :]).min() > 2, "CPCV purge failed"
    assert n_splits_seen == 15

    # ---- constant_correlation_shrinkage ----
    # build returns from a known constant-correlation covariance
    N, T = 8, 4000
    rho = 0.4
    true_corr = np.full((N, N), rho)
    np.fill_diagonal(true_corr, 1.0)
    L = np.linalg.cholesky(true_corr)
    R = rng.standard_normal((T, N)) @ L.T * 0.01
    out = constant_correlation_shrinkage(R)
    cov, delta = out["covariance"], out["shrinkage"]
    assert cov.shape == (N, N)
    assert np.allclose(cov, cov.T), "covariance not symmetric"
    assert np.linalg.eigvalsh(cov).min() > -1e-12, "covariance not PSD"
    assert 0.0 <= delta <= 1.0
    assert abs(out["avg_correlation"] - rho) < 0.05, "avg corr off"
    # scalar case
    one = constant_correlation_shrinkage(rng.standard_normal((100, 1)))
    assert one["covariance"].shape == (1, 1)

    print("validation.py: all self-tests passed "
          f"(CPCV splits={cpcv.get_n_splits()}, shrinkage delta={delta:.3f})")
