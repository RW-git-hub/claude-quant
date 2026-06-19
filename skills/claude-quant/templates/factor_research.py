"""
factor_research.py — cross-sectional FACTOR-EVALUATION toolkit.

TEACHING TEMPLATE, NOT PRODUCTION. Correct, leak-free building blocks for
ranking a panel of assets by a factor and measuring whether that factor
predicts the cross-section of forward returns.

DATA MODEL
==========
Everything is a DataFrame indexed by DATE (rows) x ASSET (columns):
  - factor_df.loc[t, a]    = factor value for asset a, known AS OF date t.
  - fwd_ret_df.loc[t, a]   = return for asset a realized AFTER date t (e.g. the
                             t -> t+1 simple return). KNOWN ONLY POST-t.
The two frames must be POINT-IN-TIME ALIGNED on the same (date x asset) grid so
that row t pairs a signal known at t with the return it is meant to predict.
This alignment is the entire ballgame: if fwd_ret.loc[t] were the return INTO t
(i.e. t-1 -> t) you would be measuring look-ahead, not predictive power.

WHY EVERYTHING IS PER-DATE (CROSS-SECTIONAL)
============================================
Cross-sectional factor research asks "within each date, do high-factor names
beat low-factor names?". Every transform and every regression is therefore
computed independently within each row. Pooling across dates (e.g. a single OLS
over the stacked panel, or a global z-score) leaks the future into the past and
lets high-volatility regimes dominate. We never do it.

CONVENTIONS (consistent with the rest of the skill)
===================================================
  - SIMPLE returns; they compound multiplicatively.
  - Information Coefficient (IC) = the cross-sectional correlation, at each date,
    between the factor at t and FORWARD returns. Spearman (rank) by default
    because factor->return is monotone-but-not-linear and rank IC is robust to
    outliers. method='pearson' available.
  - IC information ratio: IC_IR = mean(IC) / std(IC, ddof=1). Annualize by
    multiplying by sqrt(periods_per_year) if you want an annualized figure.
  - t-stat of mean IC = mean(IC)/std(IC, ddof=1) * sqrt(n_dates); same number as
    IC_IR * sqrt(n) — the Sharpe-of-IC view of significance.
  - Sharpe of the long/short quantile spread uses std with ddof=1 and the stated
    periods_per_year (252 daily). It is a GROSS, cost-free number; real spreads
    pay turnover — see backtest_skeleton.py / metrics.py.
  - All regressions add an intercept and are solved per-date via least squares on
    the rows with complete data only.

PITFALLS THIS TEMPLATE IS BUILT TO AVOID (detect/fix framing)
=============================================================
  - LOOK-AHEAD via misaligned returns: detect by confirming fwd_ret.loc[t] uses
    only price data dated > t; this file assumes the caller has done that.
  - GLOBAL standardization leaking cross-date info: fixed by per-row transforms.
  - POOLED neutralization regression leaking the future: fixed by fitting OLS
    independently within each date in neutralize() and fama_macbeth().
  - SURVIVORSHIP / changing universe: NaNs are dropped per-date, never filled,
    so an asset absent on date t simply does not participate that date.

Dependencies: numpy + pandas only.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cross-sectional transforms (each operates independently within every row)
# ---------------------------------------------------------------------------
def cross_sectional_zscore(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Per-date z-score: (x - mean_t) / std_t across assets on each date.

    Uses sample std (ddof=1) over the non-NaN names in the row. Rows with <2
    valid names (std undefined) yield all-NaN. NaNs are preserved, never filled.
    """
    mean = factor_df.mean(axis=1)
    std = factor_df.std(axis=1, ddof=1)
    z = factor_df.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
    return z


def cross_sectional_rank(factor_df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Per-date cross-sectional rank.

    pct=True -> ranks in (0, 1] (average method, ties shared). pct=False ->
    ordinal ranks 1..k where k = number of valid names that date. NaNs stay NaN.
    """
    return factor_df.rank(axis=1, pct=pct, method="average")


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row to its [lower, upper] cross-sectional quantiles.

    Tames outliers BEFORE z-scoring/regression so a single bad print does not
    dominate the cross-section. Quantiles are computed per-date on valid names;
    NaNs are ignored and preserved.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("require 0 <= lower < upper <= 1")
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


# ---------------------------------------------------------------------------
# Neutralization — strip exposure to nuisance factors, PER DATE
# ---------------------------------------------------------------------------
def _ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual y - X beta from least squares, with an intercept already in X.

    Returns NaN for any row that was excluded (handled by the caller via mask).
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def neutralize(factor_df: pd.DataFrame, *exposure_dfs: pd.DataFrame) -> pd.DataFrame:
    """Residualize the factor against exposures via PER-DATE cross-sectional OLS.

    For each date t independently, regress factor_t on [1, exposure1_t, ...] over
    the assets with complete data, and return the residual. The residual is, by
    construction, cross-sectionally orthogonal to every exposure that date — i.e.
    the part of the factor NOT explained by sector/beta/size/etc.

    Fitting each date on its own is what keeps this leak-free: betas estimated on
    date t never see dates t+1.., so no future information enters the residual.

    Assets missing the factor or ANY exposure on a date are dropped from that
    date's fit and returned as NaN (you cannot residualize what you cannot
    regress). Output shares factor_df's index/columns.
    """
    if not exposure_dfs:
        raise ValueError("neutralize requires at least one exposure DataFrame")

    exposures = [e.reindex_like(factor_df) for e in exposure_dfs]
    out = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)

    for t in factor_df.index:
        y = factor_df.loc[t]
        cols = [y] + [e.loc[t] for e in exposures]
        block = pd.concat(cols, axis=1)
        valid = block.dropna()
        if len(valid) < block.shape[1] + 1:  # need > n_params points to identify
            continue
        yv = valid.iloc[:, 0].to_numpy(dtype=float)
        Xv = valid.iloc[:, 1:].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(valid)), Xv])
        resid = _ols_residual(yv, X)
        out.loc[t, valid.index] = resid

    return out


# ---------------------------------------------------------------------------
# Information Coefficient — predictive correlation of factor vs FORWARD returns
# ---------------------------------------------------------------------------
def information_coefficient(
    factor_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Per-date IC: cross-sectional corr(factor_t, fwd_ret_t).

    method='spearman' (default) -> rank IC; 'pearson' -> linear IC. For each date
    the correlation is taken over assets present in BOTH frames with non-NaN
    values. Dates with <2 paired names yield NaN. The result is a Series indexed
    by date — the time series of daily cross-sectional skill.

    Sign convention: positive IC means high factor today -> high return next; a
    factor whose IC is reliably NEGATIVE is still a signal (flip it).
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")

    fr = fwd_ret_df.reindex_like(factor_df)
    ic = pd.Series(np.nan, index=factor_df.index, dtype=float)

    for t in factor_df.index:
        pair = pd.concat([factor_df.loc[t], fr.loc[t]], axis=1).dropna()
        if len(pair) < 2:
            continue
        a = pair.iloc[:, 0]
        b = pair.iloc[:, 1]
        if method == "spearman":
            a = a.rank()
            b = b.rank()
        # constant input -> correlation undefined; leave as NaN
        if a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
            continue
        ic.loc[t] = np.corrcoef(a.to_numpy(float), b.to_numpy(float))[0, 1]

    return ic


def ic_summary(ic: pd.Series) -> Dict[str, float]:
    """Summarize an IC time series.

    Returns:
      mean_ic   : average daily IC (the headline number).
      ic_std    : std of daily IC (ddof=1); IC volatility / decay risk.
      ic_ir     : IC information ratio = mean_ic / ic_std.
      t_stat    : mean_ic / ic_std * sqrt(n) — significance of mean IC under iid.
      hit_rate  : fraction of dates with IC > 0 (directional consistency).

    A factor that "works" typically shows mean_ic well away from 0 with |t_stat|
    comfortably > 2-3; a high mean IC with low IR (erratic, regime-driven) is a
    fragile factor.
    """
    x = ic.dropna()
    n = len(x)
    if n == 0:
        return dict(mean_ic=np.nan, ic_std=np.nan, ic_ir=np.nan, t_stat=np.nan, hit_rate=np.nan)
    mean_ic = float(x.mean())
    ic_std = float(x.std(ddof=1)) if n > 1 else np.nan
    ic_ir = mean_ic / ic_std if ic_std not in (0.0, np.nan) and np.isfinite(ic_std) else np.nan
    t_stat = ic_ir * np.sqrt(n) if np.isfinite(ic_ir) else np.nan
    hit_rate = float((x > 0).mean())
    return dict(mean_ic=mean_ic, ic_std=ic_std, ic_ir=ic_ir, t_stat=t_stat, hit_rate=hit_rate)


# ---------------------------------------------------------------------------
# Quantile (decile/quintile) portfolios — does the factor sort returns?
# ---------------------------------------------------------------------------
def quantile_returns(factor_df: pd.DataFrame, fwd_ret_df: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """Per-date equal-weight mean forward return within each factor quantile.

    For each date, valid names are sorted into q buckets by factor value
    (1 = lowest factor, q = highest) and the EQUAL-WEIGHT mean of their forward
    returns is recorded. Columns are the bucket labels 1..q plus 'spread' =
    bucket q minus bucket 1 (the long-top / short-bottom dollar-neutral leg).

    A monotone, positive-spread quantile profile is the visual confirmation of a
    positive IC; the spread row is the gross, cost-free long/short return series.
    """
    if q < 2:
        raise ValueError("q must be >= 2")

    fr = fwd_ret_df.reindex_like(factor_df)
    labels = list(range(1, q + 1))
    rows: Dict[pd.Timestamp, Dict[int, float]] = {}

    for t in factor_df.index:
        pair = pd.concat([factor_df.loc[t], fr.loc[t]], axis=1).dropna()
        if len(pair) < q:  # need at least one name per bucket
            continue
        f = pair.iloc[:, 0]
        r = pair.iloc[:, 1]
        # qcut on ranks so ties / duplicate factor values split into equal-count
        # buckets rather than collapsing edges.
        ranks = f.rank(method="first")
        try:
            buckets = pd.qcut(ranks, q, labels=labels)
        except ValueError:
            continue
        rows[t] = r.groupby(buckets, observed=False).mean().to_dict()

    qr = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=labels)
    qr.index.name = factor_df.index.name
    qr["spread"] = qr[q] - qr[1]
    return qr


def quantile_spread_summary(qr: pd.DataFrame, periods_per_year: int = 252) -> Dict[str, object]:
    """Summarize the quantile table produced by quantile_returns.

    Returns:
      mean_spread   : average top-minus-bottom forward return (gross).
      spread_sharpe : annualized Sharpe of the spread series,
                      mean/std(ddof=1)*sqrt(ppy). Gross of costs.
      monotonic     : True iff the buckets' time-mean returns are strictly
                      increasing from bucket 1 to bucket q (the ideal factor
                      shape). A positive spread with monotonic=False warns that
                      the edge lives only in the extremes — fragile.
    """
    spread = qr["spread"].dropna()
    bucket_cols = [c for c in qr.columns if c != "spread"]
    bucket_means = qr[bucket_cols].mean(axis=0)

    mean_spread = float(spread.mean()) if len(spread) else np.nan
    sd = spread.std(ddof=1) if len(spread) > 1 else np.nan
    spread_sharpe = (
        float(spread.mean() / sd * np.sqrt(periods_per_year))
        if np.isfinite(sd) and sd != 0
        else np.nan
    )
    monotonic = bool(np.all(np.diff(bucket_means.to_numpy(dtype=float)) > 0))
    return dict(mean_spread=mean_spread, spread_sharpe=spread_sharpe, monotonic=monotonic)


# ---------------------------------------------------------------------------
# Fama-MacBeth — cross-sectional risk premia with proper time-series SEs
# ---------------------------------------------------------------------------
def fama_macbeth(fwd_ret_df: pd.DataFrame, *exposure_dfs: pd.DataFrame) -> Dict[str, pd.Series]:
    """Fama-MacBeth (1973) two-pass cross-sectional regression.

    Pass 1: for each date t, regress forward returns on [1, exposure1_t, ...]
            across assets -> a vector of coefficients (lambda_t) for that date.
    Pass 2: average each coefficient over time; its t-stat uses the TIME-SERIES
            std of the per-date estimates (ddof=1), which automatically accounts
            for cross-sectional correlation in residuals — the whole point of the
            FM procedure versus a single pooled OLS.

    Per-date regressions are independent (no leakage). Each coefficient series is
    named 'intercept', 'exposure_0', 'exposure_1', ... in input order.

    Returns dict(premia=Series, t_stats=Series), both indexed by coefficient name.
    A genuine priced factor shows a premium of stable sign with |t| > ~2-3.
    """
    if not exposure_dfs:
        raise ValueError("fama_macbeth requires at least one exposure DataFrame")

    exposures = [e.reindex_like(fwd_ret_df) for e in exposure_dfs]
    names = ["intercept"] + [f"exposure_{i}" for i in range(len(exposures))]
    per_date: list[np.ndarray] = []

    for t in fwd_ret_df.index:
        y = fwd_ret_df.loc[t]
        block = pd.concat([y] + [e.loc[t] for e in exposures], axis=1).dropna()
        if len(block) < len(names) + 1:  # need > n_params points
            continue
        yv = block.iloc[:, 0].to_numpy(dtype=float)
        Xv = block.iloc[:, 1:].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(block)), Xv])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        per_date.append(beta)

    coefs = pd.DataFrame(per_date, columns=names)
    n = len(coefs)
    premia = coefs.mean(axis=0)
    sd = coefs.std(axis=0, ddof=1) if n > 1 else pd.Series(np.nan, index=names)
    t_stats = premia / sd * np.sqrt(n)
    t_stats = t_stats.where(np.isfinite(t_stats), np.nan)
    return dict(premia=premia, t_stats=t_stats)


# ===========================================================================
# Self-tests: synthetic panel with a KNOWN factor. Deterministic and fast.
# ===========================================================================
def _make_panel(T: int = 300, N: int = 50, seed: int = 7):
    """Panel where forward returns are a clean function of the factor's per-row
    z-score plus iid noise, so every metric has a known correct sign."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    assets = [f"A{i:02d}" for i in range(N)]

    F = pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=assets)
    Fz = cross_sectional_zscore(F)
    noise = pd.DataFrame(0.02 * rng.standard_normal((T, N)), index=dates, columns=assets)
    fwd = 0.05 * Fz + noise
    return F, fwd


def _run_self_tests() -> None:
    F, fwd = _make_panel()

    # --- transforms: per-row standardization is exact -------------------------
    z = cross_sectional_zscore(F)
    assert np.allclose(z.mean(axis=1).to_numpy(), 0.0, atol=1e-12), "zscore rows must be mean 0"
    assert np.allclose(z.std(axis=1, ddof=1).to_numpy(), 1.0, atol=1e-12), "zscore rows std 1"

    r = cross_sectional_rank(F, pct=True)
    assert r.to_numpy().min() > 0 and r.to_numpy().max() <= 1.0, "pct ranks in (0,1]"

    # winsorize clips into the row quantile band
    w = winsorize(F, 0.05, 0.95)
    assert (w.max(axis=1) <= F.quantile(0.95, axis=1) + 1e-9).all(), "winsor upper clip"
    assert (w.min(axis=1) >= F.quantile(0.05, axis=1) - 1e-9).all(), "winsor lower clip"

    # --- IC: should be strongly positive and highly significant ---------------
    ic = information_coefficient(F, fwd, method="spearman")
    s = ic_summary(ic)
    assert s["mean_ic"] > 0.1, f"mean IC too low: {s['mean_ic']}"
    assert s["t_stat"] > 3, f"IC t-stat too low: {s['t_stat']}"
    assert 0.0 <= s["hit_rate"] <= 1.0 and s["hit_rate"] > 0.6, "hit rate should be high"
    # IC_IR and t_stat are the same statistic up to sqrt(n)
    assert np.isclose(s["t_stat"], s["ic_ir"] * np.sqrt(ic.dropna().shape[0])), "t_stat=IR*sqrt(n)"

    # --- quantile portfolios: monotone, positive spread -----------------------
    qr = quantile_returns(F, fwd, q=5)
    qs = quantile_spread_summary(qr)
    top_mean = qr[5].mean()
    bot_mean = qr[1].mean()
    assert top_mean > bot_mean, "top quantile must beat bottom"
    assert qs["mean_spread"] > 0, "mean spread must be positive"
    assert qs["monotonic"] is True, "bucket means must be monotone increasing"
    assert qs["spread_sharpe"] > 0, "spread Sharpe positive"

    # --- Fama-MacBeth: premium on the factor positive and significant ---------
    fm = fama_macbeth(fwd, F)
    assert fm["premia"]["exposure_0"] > 0, "FM premium on factor must be positive"
    assert abs(fm["t_stats"]["exposure_0"]) > 3, f"FM |t| too low: {fm['t_stats']['exposure_0']}"

    # --- neutralize: residual is cross-sectionally orthogonal to exposure -----
    # factor = 2*exposure + independent signal; after neutralizing on exposure
    # the residual should carry essentially no exposure correlation.
    rng = np.random.default_rng(123)
    T, N = F.shape
    exposure = pd.DataFrame(rng.standard_normal((T, N)), index=F.index, columns=F.columns)
    signal = pd.DataFrame(rng.standard_normal((T, N)), index=F.index, columns=F.columns)
    factor = 2.0 * exposure + signal

    # before: strong per-date correlation with the exposure
    def _mean_abs_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
        cs = []
        for t in a.index:
            pair = pd.concat([a.loc[t], b.loc[t]], axis=1).dropna()
            if len(pair) >= 2:
                cs.append(abs(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]))
        return float(np.mean(cs))

    assert _mean_abs_corr(factor, exposure) > 0.5, "raw factor should load on exposure"
    resid = neutralize(factor, exposure)
    assert _mean_abs_corr(resid, exposure) < 0.05, "neutralized residual must be orthogonal"
    # and the residual should still resemble the independent signal
    assert _mean_abs_corr(resid, signal) > 0.5, "residual should retain the true signal"

    print("factor_research.py self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
