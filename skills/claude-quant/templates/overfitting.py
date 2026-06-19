"""overfitting.py - Probability of Backtest Overfitting via CSCV.

Purpose
-------
You searched N configurations (parameters, features, universes) and kept the
best by in-sample Sharpe. Is the *selection process* itself overfit -- i.e. is
the in-sample winner no better than a coin flip out-of-sample? This module ships
the answer the rest of the skill keeps promising: Bailey, Borwein, Lopez de Prado
& Zhu (2017), "The Probability of Backtest Overfitting" (Journal of Computational
Finance), implemented as Combinatorially Symmetric Cross-Validation (CSCV).

The key idea (why CSCV from ONE dataset approximates PBO from many samples):
split the (T x N) per-period performance matrix into S even contiguous blocks,
enumerate all C(S, S/2) ways to assign half the blocks to a training set and the
other half to a test set (symmetric: every train split has a complementary test
split). In each split, pick the config with the best IN-SAMPLE Sharpe, then look
at where that same config's OUT-OF-SAMPLE Sharpe ranks among all configs. If the
IS-best config lands in the bottom half OOS, the selection was overfit on that
split. PBO = the fraction of splits where the IS winner ranks below the OOS
median. PBO > 0.5 means your selection rule is anti-predictive: picking the
in-sample best is worse than random.

What each tool answers
----------------------
- build_perf_matrix      : turn {config_name: return_series} into the aligned
                           (T x N) matrix the rest of the module consumes.
- pbo_cscv               : the CSCV verdict -- PBO plus the full distribution of
                           per-split logits lambda = ln(rank/(1-rank)). A PBO near
                           0 with logits piled up positive = robust selection; PBO
                           > 0.5 with logits piled up negative = overfit.
- performance_degradation: Lopez de Prado's "is OOS predicted by IS?" diagnostics:
                           the OLS slope of OOS Sharpe on IS Sharpe across splits
                           (slope <= 0 => optimizing IS does not help OOS), and the
                           probability the selected config actually loses money OOS
                           (P[OOS Sharpe < 0 | selected]).

How this fits the Iron Laws
---------------------------
- Iron Law 4 (OOS sacred): CSCV is a principled, multi-path OOS evaluation of the
  *selection* decision, not a single lucky holdout.
- Iron Law 5 (deflated, honest stats): PBO is the overfitting probability you
  report alongside the Deflated Sharpe Ratio (templates/metrics.py). DSR deflates
  the headline number for N trials; PBO asks whether the *ranking* generalizes.
  Report both -- they fail in different ways.

Relationship to the rest of the skill
-------------------------------------
- Feed `pbo_cscv` the matrix of per-period returns for every config you tried
  (the FULL search, not just survivors -- like reality_check_pvalue in
  robustness.py, feeding only survivors reintroduces the snooping).
- CombinatorialPurgedKFold (templates/validation.py) produces the multi-path OOS
  returns for an ML model; collect each path's per-period returns into columns of
  the matrix and pass it here. CSCV is the verdict; CPCV is one way to generate
  the inputs.
- expected_max_sharpe / deflated_sharpe_ratio (templates/metrics.py) are the
  analytic complements.

Conventions (house style)
-------------------------
- Per-period Sharpe used internally is mean/std(ddof=1) -- NOT annualized, because
  CSCV only ever compares/ranks configs within the same split, where the constant
  sqrt(ppy) factor cancels. (Annualization would not change any rank or PBO.)
- All inputs coerced to finite float arrays; no NaNs allowed (clean/align first).
- Deterministic: enumeration is exhaustive (itertools.combinations), no RNG. The
  only nondeterminism a caller can introduce is tie-breaking in argmax, handled
  by numpy's first-index rule.

numpy / pandas / stdlib only.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Mapping

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _per_period_sharpe(block: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-period Sharpe of each column of a (rows x N) block.

    Sharpe_j = mean_t(block[:, j]) / (std_t(block[:, j], ddof=1) + eps).
    Not annualized: CSCV only ranks configs within a split, where the sqrt(ppy)
    factor is common to all columns and cancels. The eps guards a zero-variance
    (e.g. all-flat) config from producing inf/nan and corrupting the argmax/rank.
    """
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    return mu / (sd + eps)


def _coerce_matrix(perf) -> np.ndarray:
    """Coerce a (T x N) performance matrix to a finite 2-D float ndarray."""
    M = np.asarray(
        perf.values if isinstance(perf, pd.DataFrame) else perf, dtype=float
    )
    if M.ndim != 2:
        raise ValueError("perf must be 2-D (T observations x N configs)")
    if M.shape[1] < 2:
        raise ValueError(
            "need >= 2 configs to rank a winner; PBO is undefined for one config"
        )
    if not np.all(np.isfinite(M)):
        raise ValueError("perf contains non-finite values (NaN/inf); clean first")
    return M


# --------------------------------------------------------------------------- #
# build the per-config performance matrix
# --------------------------------------------------------------------------- #
def build_perf_matrix(strategy_returns: Mapping[str, "pd.Series | np.ndarray"]):
    """Assemble the (T x N) per-period return matrix CSCV consumes.

    Parameters
    ----------
    strategy_returns : mapping {config_name: per-period return series}. For pandas
        Series the columns are OUTER-joined on the index and the result is
        restricted to dates present for EVERY config (inner overlap) so each row
        is a genuine cross-config comparison -- a config that did not trade on a
        date must not be silently compared against one that did. For plain arrays
        all series must already be equal length and aligned.

    Returns
    -------
    (perf, names) where perf is a (T x N) float ndarray (columns ordered as
    `names`) and names is the list of config keys in insertion order.

    Raises if fewer than 2 configs, or if the aligned overlap has < 2 rows.
    """
    if len(strategy_returns) < 2:
        raise ValueError("need >= 2 configs to assess selection overfitting")
    names = list(strategy_returns.keys())
    values = list(strategy_returns.values())

    if all(isinstance(v, pd.Series) for v in values):
        df = pd.concat(values, axis=1, join="outer")
        df.columns = names
        df = df.dropna(how="any")           # keep only dates present for all configs
        perf = df.to_numpy(dtype=float)
    else:
        arrs = [np.asarray(v, dtype=float).ravel() for v in values]
        lengths = {a.size for a in arrs}
        if len(lengths) != 1:
            raise ValueError(
                f"array inputs must be equal length and aligned; got lengths {lengths}"
            )
        perf = np.column_stack(arrs)

    if perf.shape[0] < 2:
        raise ValueError("aligned overlap has < 2 observations; check alignment")
    if not np.all(np.isfinite(perf)):
        raise ValueError("aligned matrix contains non-finite values; clean first")
    return perf, names


# --------------------------------------------------------------------------- #
# PBO via CSCV
# --------------------------------------------------------------------------- #
def pbo_cscv(perf, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    Bailey, Borwein, Lopez de Prado & Zhu (2017). One run on a single (T x N)
    performance matrix approximates the PBO you would estimate from many
    independent samples.

    Algorithm
    ---------
    1. Split the T rows into S = n_blocks even contiguous blocks (preserving time
       order; CSCV resamples whole blocks, never shuffles within).
    2. Enumerate all C(S, S/2) ways to pick S/2 blocks as the training set; the
       complementary S/2 blocks are the test set (symmetric construction).
    3. In each split:
         - compute every config's IN-SAMPLE per-period Sharpe on the train rows;
         - n* = argmax IS Sharpe (the config you would have selected);
         - compute every config's OUT-OF-SAMPLE Sharpe on the test rows;
         - omega = relative rank of n* among OOS Sharpes in (0, 1):
               omega = (1 + #{configs with OOS Sharpe <= OOS Sharpe[n*]}) / (N + 1)
           so omega = N/(N+1) means n* is the OOS best, omega ~ 1/(N+1) the worst,
           and omega = 0.5 means it landed exactly at the OOS median.
         - logit lambda = ln(omega / (1 - omega)). lambda <= 0  <=>  omega <= 0.5
           <=>  the IS winner ranked at or below the OOS median (an overfit split).
    4. PBO = mean(lambda <= 0) = fraction of splits where selecting the IS-best
       config did not beat a coin flip out-of-sample.

    Parameters
    ----------
    perf : (T x N) array-like (DataFrame ok) of per-period performance/returns,
           one column per config. Must be finite and have >= 2 columns.
    n_blocks : S, number of contiguous blocks. MUST be even (symmetric split).
        Larger S => more splits (C(S,S/2) grows fast: 16->12,870; 18->48,620;
        20->184,756). 14-16 is a sane default; keep <= 20 or enumeration is slow.

    Returns
    -------
    dict with:
        pbo          : float in [0, 1]. > 0.5 => selection is anti-predictive.
        logits       : (n_splits,) ndarray of per-split lambda values.
        n_splits     : int == C(n_blocks, n_blocks // 2).
        n_blocks     : int (echoed).
        median_logit : float median of the logit distribution (robust summary;
                       << 0 corroborates a high PBO, >> 0 a robust selection).

    Notes
    -----
    - Ranking uses per-period (un-annualized) Sharpe; annualizing multiplies every
      column by the same sqrt(ppy) and changes no rank, hence no PBO.
    - The omega definition uses the standard (1 + count)/(N + 1) plotting-position
      so omega is strictly inside (0, 1) and the logit is always finite (no
      clipping fudge needed), and ties are handled by the `<=` count.
    """
    M = _coerce_matrix(perf)
    T, N = M.shape
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even for a symmetric split")
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    if n_blocks > T:
        raise ValueError(
            f"n_blocks ({n_blocks}) exceeds T ({T}); each block needs >= 1 row "
            f"(and >= 2 rows per half for a ddof=1 Sharpe)"
        )

    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2

    logits = []
    for train_blocks in combinations(range(n_blocks), half):
        train_set = set(train_blocks)
        tr = np.concatenate([blocks[b] for b in train_blocks])
        te = np.concatenate([blocks[b] for b in range(n_blocks) if b not in train_set])

        is_sr = _per_period_sharpe(M[tr])          # (N,) in-sample Sharpe
        oos_sr = _per_period_sharpe(M[te])         # (N,) out-of-sample Sharpe

        n_star = int(np.argmax(is_sr))             # config you would have picked
        # relative rank of n* among OOS Sharpes, in (0, 1) via plotting position:
        n_le = int(np.sum(oos_sr <= oos_sr[n_star]))   # includes n* itself (>=1)
        omega = n_le / (N + 1.0)
        logits.append(np.log(omega / (1.0 - omega)))

    logits = np.asarray(logits, dtype=float)
    n_splits = comb(n_blocks, half)
    assert logits.size == n_splits, "enumeration count mismatch"

    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "logits": logits,
        "n_splits": int(n_splits),
        "n_blocks": int(n_blocks),
        "median_logit": float(np.median(logits)),
    }


# --------------------------------------------------------------------------- #
# performance-degradation diagnostics
# --------------------------------------------------------------------------- #
def performance_degradation(perf, n_blocks: int = 16) -> dict:
    """OOS-vs-IS degradation diagnostics across all CSCV splits (Lopez de Prado).

    Complements pbo_cscv with two questions:
      1. Does optimizing in-sample even help out-of-sample? Regress the SELECTED
         config's OOS Sharpe on its IS Sharpe across splits (OLS). A slope <= 0
         means picking a higher IS Sharpe does NOT buy a higher OOS Sharpe -- the
         hallmark of overfitting. (LdP regress all configs' OOS on IS; here we use
         the per-split selected pair, which is the decision you actually make.)
      2. Probability of loss given selection: P[OOS Sharpe < 0 | config selected
         in-sample], i.e. how often the config you would have traded actually lost
         money out-of-sample.

    NOTE on the slope: across CSCV's symmetric splits, IS and OOS partition the
    SAME fixed full sample, so a split that hands the selected config a lucky-high
    IS Sharpe tends to leave a lower OOS Sharpe in the complement (and vice
    versa). This induces a mechanical NEGATIVE component in the selected-pair
    slope even when a genuine edge exists -- so read `slope` as a directional flag
    (clearly positive is reassuring), not a calibrated effect size, and lean on
    `prob_oos_loss` / `mean_oos_selected` and `pbo_cscv` for the verdict.

    Parameters
    ----------
    perf : (T x N) array-like of per-period performance, one column per config.
    n_blocks : as in pbo_cscv (even).

    Returns
    -------
    dict with:
        slope            : OLS slope of OOS Sharpe on IS Sharpe (selected config,
                           per split). See the NOTE above on interpreting it.
        intercept        : OLS intercept (the no-IS-info baseline OOS Sharpe).
        prob_oos_loss    : P[selected config's OOS Sharpe < 0]. High => the
                           winner frequently loses money out of sample.
        mean_is_selected : mean IS Sharpe of the selected configs.
        mean_oos_selected: mean OOS Sharpe of the selected configs (compare to
                           mean_is_selected to see the raw degradation).
        n_splits         : int.

    The arrays are deterministic functions of `perf` and `n_blocks` (no RNG).
    """
    M = _coerce_matrix(perf)
    T, N = M.shape
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even for a symmetric split")
    if n_blocks > T:
        raise ValueError(f"n_blocks ({n_blocks}) exceeds T ({T})")

    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2

    is_sel = []
    oos_sel = []
    for train_blocks in combinations(range(n_blocks), half):
        train_set = set(train_blocks)
        tr = np.concatenate([blocks[b] for b in train_blocks])
        te = np.concatenate([blocks[b] for b in range(n_blocks) if b not in train_set])
        is_sr = _per_period_sharpe(M[tr])
        oos_sr = _per_period_sharpe(M[te])
        n_star = int(np.argmax(is_sr))
        is_sel.append(is_sr[n_star])
        oos_sel.append(oos_sr[n_star])

    is_sel = np.asarray(is_sel, dtype=float)
    oos_sel = np.asarray(oos_sel, dtype=float)

    # OLS of OOS on IS: slope, intercept via the closed form (guard zero variance
    # in the IS regressor, which happens only if every split picked an identical
    # IS Sharpe -- then there is no IS variation to explain OOS and slope is nan).
    var_is = is_sel.var()
    if var_is > 0:
        cov = np.cov(is_sel, oos_sel, ddof=0)[0, 1]
        slope = cov / var_is
        intercept = oos_sel.mean() - slope * is_sel.mean()
    else:
        slope = float("nan")
        intercept = float(oos_sel.mean())

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "prob_oos_loss": float(np.mean(oos_sel < 0.0)),
        "mean_is_selected": float(is_sel.mean()),
        "mean_oos_selected": float(oos_sel.mean()),
        "n_splits": int(is_sel.size),
    }


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N = 1200, 40

    # ---- (1) OVERFIT NULL: configs with NO persistent edge ---------------- #
    # The honest null for backtest overfitting is "no config has a real edge."
    # Full-sample-demeaning each column forces its true mean to exactly 0, so any
    # in-sample lead is a pure in-sample fluke with nothing to bank OOS. CSCV must
    # then report a HIGH PBO: selecting the IS-best is anti-predictive OOS.
    # (NOTE: raw i.i.d. noise is a SOFTER, AMBIGUOUS null -- each column's nonzero
    #  full-sample mean is a hair of persistent "edge" drawn from N(0, sigma^2/T),
    #  so its PBO is NOT reliably above or below 0.5: across seeds it hovers near
    #  0.5 with wide scatter (~0.3-0.7). That is exactly why we use the DEMEANED
    #  null here -- forcing every column's true mean to 0 makes the overfitting
    #  null deterministic and unambiguous, so PBO->1 is a clean, repeatable test.)
    noise = rng.normal(0.0, 0.01, size=(T, N))
    overfit = noise - noise.mean(axis=0, keepdims=True)   # true mean == 0 per config
    res_overfit = pbo_cscv(overfit, n_blocks=14)
    assert 0.0 <= res_overfit["pbo"] <= 1.0
    assert res_overfit["pbo"] > 0.5, (
        f"no-persistent-edge selection should be overfit (PBO>0.5), "
        f"got {res_overfit['pbo']:.3f}"
    )
    assert res_overfit["median_logit"] <= 0.0, (
        f"overfit-null median logit should be <= 0, got {res_overfit['median_logit']:.3f}"
    )

    # ---- (2) ONE GENUINE EDGE: a single config has a constant real drift --- #
    # CSCV should report a LOW PBO: the IS-best is reliably the truly-good config,
    # which also wins OOS, so it lands in the TOP half nearly every split.
    edge = rng.normal(0.0, 0.01, size=(T, N))
    edge[:, 7] += 0.0020                     # genuine, persistent edge in one column
    res_edge = pbo_cscv(edge, n_blocks=14)
    assert res_edge["pbo"] < 0.1, (
        f"a genuine constant edge should NOT be overfit (PBO low), got {res_edge['pbo']:.3f}"
    )
    assert res_edge["pbo"] < res_overfit["pbo"], "edge PBO must be below overfit-null PBO"
    assert res_edge["median_logit"] > 0.0, "genuine-edge median logit should be > 0"

    # ---- (3) partition count == C(S, S/2), and determinism ---------------- #
    assert res_edge["n_splits"] == comb(14, 7) == 3432
    res16 = pbo_cscv(overfit, n_blocks=16)
    assert res16["n_splits"] == comb(16, 8) == 12870
    # determinism: identical inputs -> identical PBO and identical logit vector.
    res_edge_again = pbo_cscv(edge, n_blocks=14)
    assert res_edge_again["pbo"] == res_edge["pbo"]
    assert np.array_equal(res_edge_again["logits"], res_edge["logits"])

    # logits are always finite (plotting-position omega is strictly in (0,1)).
    assert np.all(np.isfinite(res_overfit["logits"]))
    assert np.all(np.isfinite(res_edge["logits"]))

    # ---- (4) performance_degradation: overfit null vs genuine edge -------- #
    deg_overfit = performance_degradation(overfit, n_blocks=14)
    deg_edge = performance_degradation(edge, n_blocks=14)
    # Genuine edge: selected config makes money OOS almost always; overfit null:
    # IS-best loses money OOS much of the time (no real OOS edge to bank).
    assert deg_edge["prob_oos_loss"] < 0.1, (
        f"genuine edge should rarely lose OOS, got {deg_edge['prob_oos_loss']:.3f}"
    )
    assert deg_overfit["prob_oos_loss"] > 0.5, (
        f"overfit null should lose OOS often, got {deg_overfit['prob_oos_loss']:.3f}"
    )
    # IS->OOS degradation: a genuine edge keeps a clearly positive OOS Sharpe,
    # while the overfit null's selected config degrades to a negative OOS Sharpe
    # on average (the IS lead was pure in-sample noise).
    assert deg_edge["mean_oos_selected"] > 0.05
    assert deg_overfit["mean_oos_selected"] < 0.0, (
        f"overfit-null IS-best should degrade to a negative OOS Sharpe on average, "
        f"got {deg_overfit['mean_oos_selected']:.3f}"
    )
    assert deg_overfit["mean_oos_selected"] < deg_edge["mean_oos_selected"]
    for d in (deg_overfit, deg_edge):
        assert d["n_splits"] == comb(14, 7)

    # ---- (5) build_perf_matrix: pandas alignment + array path ------------- #
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    sa = pd.Series(np.arange(10, dtype=float), index=idx)
    sb = pd.Series(np.arange(8, dtype=float) * 2, index=idx[2:])
    # sb is missing the first 2 dates -> inner overlap should be 8 rows.
    perf_mat, names = build_perf_matrix({"a": sa, "b": sb})
    assert names == ["a", "b"]
    assert perf_mat.shape == (8, 2), f"expected (8,2) inner overlap, got {perf_mat.shape}"
    # array path: equal-length arrays stack into columns in insertion order.
    pm2, nm2 = build_perf_matrix({"x": np.ones(5), "y": np.zeros(5), "z": np.arange(5.0)})
    assert pm2.shape == (5, 3) and nm2 == ["x", "y", "z"]

    # ---- (6) input guards ------------------------------------------------- #
    for bad_call in (
        lambda: pbo_cscv(noise, n_blocks=15),                 # odd S
        lambda: pbo_cscv(noise[:, :1]),                       # single config
        lambda: pbo_cscv(np.full((100, 3), np.nan)),          # non-finite
        lambda: pbo_cscv(noise, n_blocks=2000),               # S > T
        lambda: build_perf_matrix({"only": np.ones(10)}),     # < 2 configs
        lambda: build_perf_matrix({"a": np.ones(5), "b": np.ones(6)}),  # ragged
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected a ValueError guard to reject invalid input")

    print(
        f"overfitting.py self-tests passed "
        f"(overfit-null PBO={res_overfit['pbo']:.2f}, edge PBO={res_edge['pbo']:.2f}, "
        f"splits={res_edge['n_splits']})."
    )
