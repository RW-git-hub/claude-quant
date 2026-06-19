"""
validation.py - leak-free cross-validation for time-series / financial ML, plus a
constant-correlation covariance shrinkage estimator and a walk-forward parameter-
sweep harness that wires the leak-free splitters into honest, deflated OOS stats.

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

walk_forward_evaluate / summarize_search are the canonical PARAMETER-SWEEP entry
point (research-backtest.md sections 1 and 8, Playbook 4). They march an
expanding/rolling purged+embargoed window over the data, call a user strategy_fn
for every grid point, assemble the (config x time) OOS performance matrix, and
deflate the winner honestly: Deflated Sharpe with an EFFECTIVE trial count (not the
naive grid size, since grid points are correlated) and a PBO/CSCV overfitting
probability. This is the glue that operationalizes Iron Law 4 (OOS sacred) and
Iron Law 5 (deflated, honest stats) end-to-end. The harness owns trial counting and
the purge/embargo, which is exactly where under-counting and OOS peeking otherwise
creep in.

constant_correlation_shrinkage implements Ledoit-Wolf (2003), "Honey, I Shrunk the
Sample Covariance Matrix" - the CONSTANT-CORRELATION target appropriate for asset
returns. (sklearn.covariance.LedoitWolf uses the scaled-IDENTITY target of LW 2004,
which is usually worse for return covariances.)

numpy/pandas-only, self-testing. Run: python validation.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
from itertools import combinations
from math import comb, e as _E
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

_NORM = NormalDist()
_EULER = 0.5772156649015329  # Euler-Mascheroni constant
_EPS = 1e-12


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


# --------------------------------------------------------------------------- #
# Walk-forward parameter-sweep harness
#
# The missing glue between the splitters above, the Deflated Sharpe (metrics.py),
# and the PBO/CSCV verdict (overfitting.py). It runs the sweep FOR you so the
# trial count is honest and the purge/embargo cannot be forgotten.
# --------------------------------------------------------------------------- #
def walk_forward_windows(n: int, train: int, test: int, *, anchored: bool = False,
                         label_horizon: int = 1, embargo_pct: float = 0.0
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (train_idx, test_idx) for a marching walk-forward.

    The window slides forward in non-overlapping `test`-sized steps. With
    anchored=False the train window is a ROLLING window of the most recent `train`
    bars before each test block; with anchored=True it is an EXPANDING window
    [0, test_start) (all history). Either way every train index lies strictly
    BEFORE its test block (Iron Law 1: no look-ahead), and the purge+embargo of
    `label_horizon` + round(n*embargo_pct) bars is applied around the test block so
    no train label window overlaps the test span.

    Returns a list of (train_idx, test_idx) ndarrays (chronological). The test
    blocks are contiguous and tile [train, train + k*test) without overlap, so
    concatenating each config's per-block OOS returns yields a single
    gap-free OOS path.
    """
    if train < 2 or test < 1:
        raise ValueError("require train >= 2 and test >= 1")
    if train + test > n:
        raise ValueError(f"train+test ({train + test}) exceeds n ({n}); no window fits")
    embargo = int(round(n * embargo_pct))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    test_start = train
    while test_start + test <= n:
        test_idx = np.arange(test_start, test_start + test)
        tr_lo = 0 if anchored else test_start - train
        tr_full = np.arange(tr_lo, test_start)
        # Purge/embargo the FULL series, then keep only the candidate train rows
        # that survive. Because train rows all precede the test block, this trims
        # the last `label_horizon` train bars whose label window reaches into the
        # test span (the embargo is a right-side buffer and so trims nothing here,
        # but is applied for symmetry with the splitters / future test blocks).
        mask = _purged_train_mask(n, test_idx, embargo, label_horizon)
        train_idx = tr_full[mask[tr_full]]
        out.append((train_idx, test_idx))
        test_start += test
    if not out:
        raise ValueError("no walk-forward windows produced; check train/test/n")
    return out


class WalkForwardResult:
    """Container returned by walk_forward_evaluate.

    Attributes
    ----------
    perf_matrix : (T_oos x N) DataFrame of per-period OOS returns, one column per
        config (column label = stringified params), rows = concatenated OOS path
        in time order. This is exactly the matrix overfitting.pbo_cscv and the
        Deflated Sharpe consume.
    config_names : list[str] of the N column labels (grid order preserved).
    params : list of the original param dicts (aligned with config_names).
    n_trials : int == len(param_grid). The HONEST trial count for multiple-testing
        deflation - every grid point you ran, survivors and failures alike.
    windows : list of (train_idx, test_idx) used (for purge/embargo auditing).
    """

    __slots__ = ("perf_matrix", "config_names", "params", "n_trials", "windows")

    def __init__(self, perf_matrix, config_names, params, n_trials, windows):
        self.perf_matrix = perf_matrix
        self.config_names = config_names
        self.params = params
        self.n_trials = n_trials
        self.windows = windows


def walk_forward_evaluate(strategy_fn: Callable[[np.ndarray, np.ndarray, Mapping], "pd.Series | np.ndarray"],
                          param_grid: Sequence[Mapping],
                          data,
                          *,
                          train: int,
                          test: int,
                          anchored: bool = False,
                          label_horizon: int = 1,
                          embargo_pct: float = 0.0) -> WalkForwardResult:
    """Run a parameter sweep through a purged+embargoed walk-forward and assemble
    the (config x time) OOS performance matrix plus an honest trial count.

    Parameters
    ----------
    strategy_fn : callable(train_idx, test_idx, params) -> oos return series.
        Called ONCE per (window, params). It receives integer positional indices
        into `data` (NOT labels) for the purged train rows and the test rows, plus
        the params dict for this grid point. It must return the per-period
        strategy returns realized on the TEST rows (length == len(test_idx)),
        already lagged/costed by you (the harness does not lag for you - Iron Law 1
        is your strategy's responsibility, but the harness guarantees train_idx
        never overlaps test_idx). Returns may be a Series or array.
    param_grid : sequence of param dicts. len(param_grid) IS n_trials.
    data : the panel passed straight through to strategy_fn (only its length is
        used here, via shape[0]/len). Keeping it opaque lets the same harness drive
        a returns vector, a feature matrix, or a price panel.
    train, test : window sizes in bars (see walk_forward_windows).
    anchored : expanding (True) vs rolling (False) train window.
    label_horizon, embargo_pct : purge/embargo controls (see _purged_train_mask).

    Returns
    -------
    WalkForwardResult. Feed .perf_matrix to summarize_search (or directly to
    overfitting.pbo_cscv) to get the deflated verdict.

    Notes
    -----
    - Deterministic: no RNG here. Any randomness must live in strategy_fn (seed it).
    - Each config's OOS columns are the SAME length and aligned (same test blocks),
      so the matrix needs no further alignment before CSCV.
    """
    if len(param_grid) < 1:
        raise ValueError("param_grid is empty")
    n = _n_samples(data)
    windows = walk_forward_windows(n, train, test, anchored=anchored,
                                   label_horizon=label_horizon, embargo_pct=embargo_pct)

    # Sanity: total OOS length is the same for every config (sum of test-block sizes).
    oos_len = sum(len(te) for _, te in windows)

    columns: dict[str, np.ndarray] = {}
    names: list[str] = []
    params_list: list[Mapping] = []
    for params in param_grid:
        segs = []
        for train_idx, test_idx in windows:
            # Hard guard (cheap, runs every call): the harness's core promise.
            # A test index appearing in its own train fold is a leak; refuse to run.
            if np.intersect1d(train_idx, test_idx).size != 0:
                raise AssertionError("purge/embargo violated: train overlaps test")
            r = strategy_fn(train_idx, test_idx, params)
            r = np.asarray(pd.Series(r, dtype="float64").to_numpy(), dtype=float)
            if r.shape[0] != test_idx.shape[0]:
                raise ValueError(
                    f"strategy_fn returned {r.shape[0]} obs for a {test_idx.shape[0]}-bar "
                    f"test block (params={dict(params)}); must equal len(test_idx)"
                )
            segs.append(r)
        col = np.concatenate(segs)
        assert col.shape[0] == oos_len, "internal: OOS column length mismatch"
        name = str(dict(params))
        # Disambiguate duplicate param dicts so columns stay 1:1 with trials.
        if name in columns:
            name = f"{name}#{len(names)}"
        columns[name] = col
        names.append(name)
        params_list.append(dict(params))

    perf = pd.DataFrame(columns)[names]  # preserve grid order
    return WalkForwardResult(
        perf_matrix=perf,
        config_names=names,
        params=params_list,
        n_trials=len(param_grid),
        windows=windows,
    )


# --------------------------------------------------------------------------- #
# Effective number of trials (correlation-aware deflation)
# --------------------------------------------------------------------------- #
def effective_n_trials(perf_matrix) -> float:
    """Effective number of INDEPENDENT trials in a (T x N) performance matrix.

    A grid search rarely tests N independent strategies: neighbouring parameters
    produce highly correlated return streams, so the true multiple-testing budget
    is smaller than N. We estimate it from the participation ratio of the eigenvalue
    spectrum of the config correlation matrix C:

        n_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2

    where lambda_i are the eigenvalues of C. This equals N when columns are
    uncorrelated (C = I -> all lambda = 1) and collapses toward 1 as columns become
    perfectly correlated (one dominant eigenvalue). It is the spectral entropy /
    participation-ratio measure of effective dimensionality.

    Using n_eff (rather than the raw grid size) as the trial count in the Deflated
    Sharpe is the correct, non-conservative-but-non-anticonservative choice: the
    expected max Sharpe under the null grows with the number of INDEPENDENT bets.

    Guards: a config with zero variance (degenerate column) is dropped before the
    correlation is formed (its correlation is undefined). Returns 1.0 for a single
    usable config; clamps the result to [1, N_usable].
    """
    M = np.asarray(perf_matrix.values if isinstance(perf_matrix, pd.DataFrame)
                   else perf_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T x N)")
    sd = M.std(axis=0, ddof=1)
    keep = sd > _EPS
    M = M[:, keep]
    N = M.shape[1]
    if N <= 1:
        return 1.0
    C = np.corrcoef(M, rowvar=False)
    # numerical hygiene: symmetrize, clip tiny negative eigenvalues to 0.
    C = 0.5 * (C + C.T)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-10]
    if ev.size == 0:
        return 1.0
    n_eff = float((ev.sum() ** 2) / np.sum(ev ** 2))
    return float(min(max(n_eff, 1.0), N))


# --------------------------------------------------------------------------- #
# Self-contained fallbacks for the deflated stats (so this file stands alone).
# summarize_search PREFERS the canonical sibling implementations (metrics.py,
# overfitting.py) when importable, keeping a single source of truth in production,
# and falls back to these vetted copies otherwise. The fallbacks mirror the sibling
# formulas exactly and are covered by this file's own self-tests.
# --------------------------------------------------------------------------- #
def _pp_sharpe(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    sd = np.std(a, ddof=1)
    return float(a.mean() / sd) if sd > _EPS else float("nan")


def _skew_kurt(a: np.ndarray) -> tuple[float, float]:
    s = np.std(a, ddof=0)
    if s == 0:
        return 0.0, 3.0
    z = (a - a.mean()) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def _psr_fallback(a: np.ndarray, benchmark_sr: float) -> float:
    a = np.asarray(a, dtype=float)
    n = a.size
    if n < 3:
        return float("nan")
    sr = _pp_sharpe(a)
    if math.isnan(sr):
        return float("nan")
    g1, g2 = _skew_kurt(a)
    denom = math.sqrt(max(1e-12, 1 - g1 * sr + (g2 - 1) / 4.0 * sr ** 2))
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / denom
    return _NORM.cdf(z)


def _expected_max_sharpe_fallback(trial_sharpe_std: float, n_trials: float) -> float:
    if n_trials <= 1:
        return 0.0
    z1 = _NORM.inv_cdf(1 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1 - 1.0 / (n_trials * _E))
    return trial_sharpe_std * ((1 - _EULER) * z1 + _EULER * z2)


def _dsr_fallback(returns: np.ndarray, n_trials: float, trial_sharpe_std: float) -> float:
    sr0 = _expected_max_sharpe_fallback(trial_sharpe_std, n_trials)
    return _psr_fallback(returns, benchmark_sr=sr0)


def _pbo_fallback(perf: np.ndarray, n_blocks: int) -> dict:
    M = np.asarray(perf, dtype=float)
    T, N = M.shape
    if n_blocks % 2 != 0 or n_blocks < 2 or n_blocks > T:
        raise ValueError("invalid n_blocks for CSCV")
    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2
    logits = []
    for train_blocks in combinations(range(n_blocks), half):
        ts = set(train_blocks)
        tr = np.concatenate([blocks[b] for b in train_blocks])
        te = np.concatenate([blocks[b] for b in range(n_blocks) if b not in ts])
        is_sr = M[tr].mean(0) / (M[tr].std(0, ddof=1) + _EPS)
        oos_sr = M[te].mean(0) / (M[te].std(0, ddof=1) + _EPS)
        n_star = int(np.argmax(is_sr))
        n_le = int(np.sum(oos_sr <= oos_sr[n_star]))
        omega = n_le / (N + 1.0)
        logits.append(np.log(omega / (1.0 - omega)))
    logits = np.asarray(logits, dtype=float)
    return {"pbo": float(np.mean(logits <= 0.0)), "logits": logits,
            "n_splits": comb(n_blocks, half), "median_logit": float(np.median(logits))}


def _load_sibling(mod_name: str):
    """Import a sibling template module (metrics / overfitting) from THIS file's
    directory without requiring a package or a particular sys.path. Returns the
    module, or None if it is not importable (then the fallback is used)."""
    try:
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        path = Path(__file__).resolve().parent / f"{mod_name}.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def summarize_search(result: "WalkForwardResult | pd.DataFrame | np.ndarray",
                     *, n_trials: int | None = None,
                     periods_per_year: int = 252,
                     pbo_blocks: int = 14) -> dict:
    """One-call honest verdict for a parameter sweep (Iron Laws 4 and 5).

    Takes the OOS performance matrix (a WalkForwardResult, or a raw (T x N)
    DataFrame/array of per-config per-period OOS returns) and returns a single dict
    summarizing the selection decision:

        best_config             : column label / index of the highest-OOS-Sharpe config
        oos_sharpe              : ANNUALIZED OOS Sharpe of that config
        dsr                     : Deflated Sharpe Ratio of the winner, deflated by the
                                  EFFECTIVE trial count (effective_n_trials) and the
                                  cross-trial Sharpe std. DSR < 0.95 => the winner is
                                  plausibly a multiple-testing artefact.
        pbo                     : Probability of Backtest Overfitting (CSCV). > 0.5 =>
                                  selecting the in-sample best is anti-predictive OOS.
        n_trials                : the HONEST raw grid size (every config you ran).
        n_eff                   : effective independent trials (n_trials adjusted for
                                  cross-config correlation; <= n_trials).
        performance_degradation : dict from overfitting.performance_degradation
                                  (IS->OOS slope, prob_oos_loss, ...). None if the
                                  overfitting.py sibling is not importable (the PBO
                                  fallback still fires) or the matrix is too small for
                                  the requested pbo_blocks.

    The Sharpe used for ranking and for n_eff is the PER-PERIOD Sharpe; oos_sharpe
    is annualized for reporting. DSR/PBO use per-period internally (annualization is
    a common positive scale and changes no rank or probability).

    Implementation prefers the canonical metrics.deflated_sharpe_ratio and
    overfitting.pbo_cscv / performance_degradation when importable (single source of
    truth); otherwise uses the vetted in-file fallbacks (identical formulas).
    """
    if isinstance(result, WalkForwardResult):
        perf = result.perf_matrix
        raw_n_trials = result.n_trials
    else:
        perf = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        raw_n_trials = n_trials if n_trials is not None else perf.shape[1]

    if not isinstance(perf, pd.DataFrame):
        perf = pd.DataFrame(perf)
    M = perf.to_numpy(dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("need a (T x N>=2) OOS performance matrix")
    if not np.all(np.isfinite(M)):
        raise ValueError("performance matrix has non-finite values; clean/align first")

    # Rank configs by per-period OOS Sharpe; pick the winner (first on ties).
    pp = np.array([_pp_sharpe(M[:, j]) for j in range(M.shape[1])])
    pp_safe = np.where(np.isnan(pp), -np.inf, pp)
    best_j = int(np.argmax(pp_safe))
    best_label = perf.columns[best_j]
    best_ret = M[:, best_j]

    # Cross-trial Sharpe dispersion (per-period units) and effective trial count.
    trial_sharpe_std = float(np.nanstd(pp, ddof=1)) if np.sum(~np.isnan(pp)) > 1 else 0.0
    n_eff = effective_n_trials(M)

    sqrt_ppy = math.sqrt(periods_per_year)
    oos_sharpe_ann = (pp[best_j] * sqrt_ppy) if not math.isnan(pp[best_j]) else float("nan")

    # --- Deflated Sharpe (prefer metrics.py) ---
    metrics = _load_sibling("metrics")
    if metrics is not None and hasattr(metrics, "deflated_sharpe_ratio"):
        dsr = float(metrics.deflated_sharpe_ratio(
            best_ret, n_trials=max(int(round(n_eff)), 1),
            trial_sharpe_std=trial_sharpe_std, periods_per_year=periods_per_year))
    else:
        dsr = float(_dsr_fallback(best_ret, n_trials=max(n_eff, 1.0),
                                  trial_sharpe_std=trial_sharpe_std))

    # --- PBO + degradation (prefer overfitting.py) ---
    over = _load_sibling("overfitting")
    pbo = float("nan")
    degradation = None
    # CSCV needs each half to hold >= 2 rows for a ddof=1 Sharpe.
    usable_blocks = min(pbo_blocks, (M.shape[0] // 2) * 2 // 2 * 2)
    if usable_blocks % 2:
        usable_blocks -= 1
    if usable_blocks >= 2 and usable_blocks <= M.shape[0]:
        if over is not None and hasattr(over, "pbo_cscv"):
            pbo = float(over.pbo_cscv(M, n_blocks=usable_blocks)["pbo"])
            if hasattr(over, "performance_degradation"):
                degradation = over.performance_degradation(M, n_blocks=usable_blocks)
        else:
            pbo = float(_pbo_fallback(M, usable_blocks)["pbo"])

    return {
        "best_config": best_label,
        "oos_sharpe": oos_sharpe_ann,
        "dsr": dsr,
        "pbo": pbo,
        "n_trials": int(raw_n_trials),
        "n_eff": float(n_eff),
        "performance_degradation": degradation,
    }


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

    # ---- walk_forward_windows: leak-free, tiling, anchored vs rolling --------
    wins = walk_forward_windows(1000, train=200, test=100,
                                label_horizon=5, embargo_pct=0.01)
    cover = []
    for tr, te in wins:
        # core promise: test never appears in its own train fold
        assert np.intersect1d(tr, te).size == 0, "train overlaps test"
        # every train index strictly precedes the test block (no look-ahead)
        assert tr.max() < te.min(), "train index not before test block"
        # purge: last label_horizon train bars before the test block are dropped
        assert te.min() - tr.max() > 5, "purge gap too small"
        cover.append(te)
    # test blocks tile [200, 1000) without overlap or gap
    cov = np.concatenate(cover)
    assert np.array_equal(cov, np.arange(200, 1000)), "test blocks must tile contiguously"
    # anchored window expands: later folds have >= as many train rows as earlier
    wa = walk_forward_windows(1000, train=200, test=100, anchored=True, label_horizon=5)
    sizes = [len(tr) for tr, _ in wa]
    assert all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1)), "anchored not expanding"
    assert wa[0][0].min() == 0 and wa[-1][0].min() == 0, "anchored train must start at 0"

    # ---- walk_forward_evaluate + summarize_search: GENUINE EDGE -------------
    # Forward return follows the sign of predictor column 5 (a modest, persistent
    # edge); the other 19 configs key off pure-noise predictors. The chosen best
    # should be the truly-good config, its OOS Sharpe should beat the noise median,
    # and PBO should be LOW (selecting the IS-best generalizes OOS).
    T, Npar = 1600, 20
    preds = rng.standard_normal((T, Npar))
    noise = rng.normal(0.0, 0.01, T)
    fwd = 0.0014 * np.sign(preds[:, 5]) + noise        # genuine driver = column 5

    def strat_edge(train_idx, test_idx, params):
        j = params["j"]
        # position is a pure function of the predictor on the SAME test bars; fwd is
        # already the next-bar return, so this earns OOS returns with no look-ahead.
        return np.sign(preds[test_idx, j]) * fwd[test_idx]

    grid = [{"j": j} for j in range(Npar)]
    res_edge = walk_forward_evaluate(strat_edge, grid, preds,
                                     train=400, test=100, label_horizon=1, embargo_pct=0.0)
    assert res_edge.n_trials == Npar == 20, "n_trials must equal the raw grid size"
    assert res_edge.perf_matrix.shape[1] == Npar
    # per-config OOS path length == sum of test-block sizes
    assert res_edge.perf_matrix.shape[0] == sum(len(te) for _, te in res_edge.windows)

    summ_edge = summarize_search(res_edge)
    # the winner is the genuinely-good config
    assert summ_edge["best_config"] == "{'j': 5}", f"got {summ_edge['best_config']}"
    # its OOS Sharpe beats the median across the grid (annualized, monotone in pp)
    pp_all = np.array([_pp_sharpe(res_edge.perf_matrix.to_numpy()[:, j]) for j in range(Npar)])
    assert summ_edge["oos_sharpe"] / math.sqrt(252) > np.median(pp_all), \
        "winner should beat the noise median"
    # selection generalizes: PBO is low and the deflated Sharpe still clears 0.95
    # even AFTER deflating for the effective trial count - the mark of a real edge.
    assert summ_edge["pbo"] < 0.1, f"genuine edge PBO should be low, got {summ_edge['pbo']:.3f}"
    assert summ_edge["dsr"] > 0.95, f"genuine edge DSR should clear 0.95, got {summ_edge['dsr']:.3f}"
    # honest vs effective trial count
    assert summ_edge["n_trials"] == 20
    assert 1.0 <= summ_edge["n_eff"] <= 20.0
    # degradation diagnostics are an optional enrichment supplied by the
    # overfitting.py sibling; when it is importable (the production layout) they
    # must be present and sane for a real edge. In a truly standalone run (no
    # sibling) the PBO fallback still fires but degradation is None - so gate the
    # assertion on availability rather than coupling the self-test to the sibling.
    if summ_edge["performance_degradation"] is not None:
        assert summ_edge["performance_degradation"]["prob_oos_loss"] < 0.5

    # ---- walk_forward_evaluate + summarize_search: ALL-NOISE GRID -----------
    # Forward return is pure noise; every config keys off an independent predictor.
    # No config has a real edge, so the Deflated Sharpe of the (lucky) in-sample
    # winner must be UNCONVINCING (DSR < 0.95): the headline number is fully
    # explained by having searched ~20 trials.
    fwd_noise = rng.normal(0.0, 0.01, T)

    def strat_noise(train_idx, test_idx, params):
        j = params["j"]
        return np.sign(preds[test_idx, j]) * fwd_noise[test_idx]

    res_noise = walk_forward_evaluate(strat_noise, grid, preds,
                                      train=400, test=100, label_horizon=1)
    summ_noise = summarize_search(res_noise)
    assert summ_noise["dsr"] < 0.95, f"all-noise DSR must be < 0.95, got {summ_noise['dsr']:.3f}"
    # the edge grid should look strictly better than the noise grid on the DSR axis
    assert summ_edge["dsr"] > summ_noise["dsr"], "edge DSR must exceed noise DSR"

    # PBO of a RAW i.i.d.-noise grid is a SOFT null: each config has a sliver of
    # persistent full-sample mean (finite-sample fluke), so the in-sample winner is
    # also a top OOS performer often enough that PBO sits below 0.5 - and exactly
    # where is realization-dependent. The HARD overfitting null - the one PBO is
    # designed to flag - is selection on differences with nothing to bank OOS.
    # Model it by demeaning each config to a true mean of exactly zero (same
    # construction as overfitting.py's own self-test): now any in-sample lead is a
    # pure fluke, and selecting the IS-best must be anti-predictive OOS.
    overfit_null = res_noise.perf_matrix - res_noise.perf_matrix.mean(axis=0)
    summ_overfit = summarize_search(overfit_null, n_trials=res_noise.n_trials)
    assert summ_overfit["pbo"] > 0.5, \
        f"overfit-null PBO must be > 0.5, got {summ_overfit['pbo']:.3f}"
    assert summ_overfit["dsr"] < 0.95, \
        f"overfit-null DSR must be < 0.95, got {summ_overfit['dsr']:.3f}"
    # the genuine-edge grid must look strictly better than the overfit null on PBO
    assert summ_edge["pbo"] < summ_overfit["pbo"], "edge PBO must be below overfit-null PBO"

    # ---- summarize_search also accepts a raw DataFrame ----------------------
    summ_df = summarize_search(res_edge.perf_matrix)
    assert summ_df["best_config"] == summ_edge["best_config"]
    assert summ_df["n_trials"] == res_edge.perf_matrix.shape[1]

    # ---- effective_n_trials: uncorrelated ~ N, perfectly correlated ~ 1 -----
    indep = rng.standard_normal((2000, 10))
    assert abs(effective_n_trials(indep) - 10) < 2.0, "independent columns -> n_eff ~ N"
    base = rng.standard_normal((2000, 1))
    clones = base + 1e-6 * rng.standard_normal((2000, 10))   # 10 near-identical cols
    assert effective_n_trials(clones) < 2.0, "near-identical columns -> n_eff ~ 1"

    # ---- harness rejects a strategy_fn that returns the wrong length --------
    def strat_bad(train_idx, test_idx, params):
        return np.zeros(test_idx.size + 1)
    try:
        walk_forward_evaluate(strat_bad, [{"j": 0}, {"j": 1}], preds, train=400, test=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on wrong-length strategy output")

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
          f"(CPCV splits={cpcv.get_n_splits()}, shrinkage delta={delta:.3f}; "
          f"edge: best={summ_edge['best_config']} DSR={summ_edge['dsr']:.2f} "
          f"PBO={summ_edge['pbo']:.2f} n_eff={summ_edge['n_eff']:.1f}; "
          f"noise DSR={summ_noise['dsr']:.2f}; overfit-null PBO={summ_overfit['pbo']:.2f})")
