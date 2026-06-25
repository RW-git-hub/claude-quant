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


def _newey_west_lrv(x: np.ndarray, nw_lags: int) -> float:
    """Newey-West (1987) HAC long-run variance of the MEAN of x.

    Returns an estimate of Var(mean(x)) that is robust to autocorrelation up to
    `nw_lags` lags using the Bartlett kernel w_k = 1 - k/(nw_lags+1):

        S = gamma_0 + 2 * sum_{k=1..L} w_k * gamma_k      (long-run variance of x)
        Var(mean) = S / n

    where gamma_k = (1/n) * sum_t (x_t - xbar)(x_{t-k} - xbar) is the biased
    (divide-by-n) sample autocovariance. With nw_lags=0 this collapses to the iid
    estimate gamma_0 / n = (population) variance / n. The Bartlett weights
    guarantee S >= 0 (positive semidefinite), so the returned variance is never
    negative. Reference: Newey & West, Econometrica 55 (1987), 703-708.

    PURELY DESCRIPTIVE on a realized IC series — uses no future data per element,
    but it is a full-sample statistic and so is NOT a causal backtest signal.
    """
    n = x.size
    if n < 2:
        return float("nan")
    xc = x - x.mean()
    gamma0 = float(xc @ xc) / n
    s = gamma0
    L = min(int(nw_lags), n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (nw_lags + 1.0)
        gamma_k = float(xc[k:] @ xc[:-k]) / n
        s += 2.0 * w * gamma_k
    s = max(s, 0.0)  # Bartlett kernel is PSD; guard tiny negative FP drift
    return s / n


def ic_summary_hac(ic: pd.Series, nw_lags: int = 5) -> Dict[str, float]:
    """ic_summary plus a Newey-West (HAC) t-stat of the mean IC.

    Overlapping forward-return windows make consecutive ICs autocorrelated, which
    inflates the naive iid t-stat (it assumes independent ICs). The HAC t-stat
    divides the mean IC by a Newey-West standard error that accounts for serial
    correlation up to `nw_lags`, and is the honest significance number for an
    overlapping panel.

    Returns every key from ic_summary plus:
      t_stat_hac : mean_ic / sqrt(NW long-run var of the mean), nw_lags Bartlett.
      nw_lags    : the lag truncation actually requested.

    With positively autocorrelated ICs (the usual overlapping case) t_stat_hac is
    typically SMALLER in magnitude than the iid t_stat — that shrinkage is the
    point. Hand-rolled in numpy; see _newey_west_lrv for the math/reference.
    """
    base = ic_summary(ic)
    x = ic.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 2:
        base.update(t_stat_hac=np.nan, nw_lags=int(nw_lags))
        return base
    var_mean = _newey_west_lrv(x, nw_lags)
    se = np.sqrt(var_mean) if np.isfinite(var_mean) and var_mean > 0 else np.nan
    t_stat_hac = float(x.mean() / se) if np.isfinite(se) and se != 0 else np.nan
    base.update(t_stat_hac=t_stat_hac, nw_lags=int(nw_lags))
    return base


# ---------------------------------------------------------------------------
# IC decay — how fast does the signal's edge fade over the holding horizon?
# ---------------------------------------------------------------------------
def ic_decay(
    factor_df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizons,
    method: str = "spearman",
) -> pd.Series:
    """Mean IC of the factor at t vs the h-step-ahead PER-ASSET forward return.

    `fwd_returns.loc[t, a]` is the one-step forward return of asset a known after
    date t (same convention as everywhere in this file: it is the t -> t+1
    return). For horizon h we correlate factor_t with the return realized h steps
    later, i.e. fwd_returns shifted UP by (h - 1) rows so that row t holds the
    t+h-1 -> t+h return. h=1 reproduces the contemporaneous one-step IC.

    LEAK-SAFE ALIGNMENT: the panel is first sorted by date (ascending) so "shift
    up by k" is unambiguously "later in time"; we never look backward. shift(-k)
    pulls a FUTURE row's return back to align with the present factor (correct:
    the factor is known now, the return is realized later) and the last k rows
    become NaN and are dropped per-date — no wrap-around, no look-ahead. The
    factor itself is never shifted, so no future factor value ever enters.

    Returns a Series indexed by horizon h, value = mean over dates of the per-date
    cross-sectional IC at that horizon. A real signal shows IC highest at short
    horizons and decaying toward 0 as h grows.
    """
    f = factor_df.sort_index()
    fr = fwd_returns.reindex_like(f).sort_index()
    out = pd.Series(np.nan, index=pd.Index(list(horizons), name="horizon"), dtype=float)
    for h in horizons:
        if h < 1:
            raise ValueError("horizons must be >= 1")
        shifted = fr.shift(-(int(h) - 1))  # row t now holds the t+h-1 -> t+h return
        ic_h = information_coefficient(f, shifted, method=method)
        out.loc[h] = float(ic_h.dropna().mean()) if ic_h.notna().any() else np.nan
    return out


def ic_half_life(decay: pd.Series) -> float:
    """Horizon at which mean IC first falls to <= half its h=1 (peak) value.

    Linearly interpolates between the bracketing horizons for a fractional
    half-life. Expects the Series returned by ic_decay (index = horizons). Returns
    NaN if the first horizon's IC is non-positive (no decay to measure) or if IC
    never reaches half within the supplied horizons.

    Purely descriptive summary of an IC-decay curve; not a tradable signal.
    """
    d = decay.dropna()
    if d.empty:
        return float("nan")
    hs = np.asarray(d.index, dtype=float)
    vals = d.to_numpy(dtype=float)
    order = np.argsort(hs)
    hs, vals = hs[order], vals[order]
    peak = vals[0]
    if peak <= 0:
        return float("nan")
    target = peak / 2.0
    for i in range(1, len(vals)):
        if vals[i] <= target:
            # interpolate between (hs[i-1], vals[i-1]) and (hs[i], vals[i])
            v0, v1 = vals[i - 1], vals[i]
            h0, h1 = hs[i - 1], hs[i]
            if v0 == v1:
                return float(h1)
            frac = (v0 - target) / (v0 - v1)
            return float(h0 + frac * (h1 - h0))
    return float("nan")


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


def _quantile_weights(
    factor_df: pd.DataFrame,
    q: int,
    weight: str,
    mktcap_df: pd.DataFrame | None,
):
    """Per-date long-top / short-bottom weights for the q-bucket spread portfolio.

    Yields (t, w_t) where w_t is a Series over that date's valid assets summing to
    0 (long top bucket, short bottom bucket). 'equal' splits each leg evenly;
    'cap' weights each leg PROPORTIONAL to that name's market cap within its
    bucket (a value-weighted spread, as published factor returns usually are).

    LEAK-SAFE: weights for date t use only the factor and the market cap known AS
    OF date t; nothing from t+1 enters. mktcap_df is aligned (reindexed) to the
    factor grid and a name is dropped from a date whenever its cap is missing or
    non-positive (a cap weight needs a strictly positive cap).
    """
    if weight not in ("equal", "cap"):
        raise ValueError("weight must be 'equal' or 'cap'")
    if weight == "cap" and mktcap_df is None:
        raise ValueError("weight='cap' requires an aligned mktcap_df panel")

    cap = mktcap_df.reindex_like(factor_df) if mktcap_df is not None else None
    labels = list(range(1, q + 1))

    for t in factor_df.index:
        f = factor_df.loc[t]
        if cap is not None:
            block = pd.concat([f, cap.loc[t]], axis=1).dropna()
            block = block[block.iloc[:, 1] > 0.0]  # caps must be strictly positive
            if len(block) < q:
                continue
            f = block.iloc[:, 0]
            c = block.iloc[:, 1]
        else:
            f = f.dropna()
            if len(f) < q:
                continue
            c = None

        ranks = f.rank(method="first")
        try:
            buckets = pd.qcut(ranks, q, labels=labels)
        except ValueError:
            continue
        top = f.index[buckets == q]
        bot = f.index[buckets == 1]
        if len(top) == 0 or len(bot) == 0:
            continue

        w = pd.Series(0.0, index=f.index)
        if weight == "equal":
            w.loc[top] = 1.0 / len(top)
            w.loc[bot] = -1.0 / len(bot)
        else:
            ct, cb = c.loc[top], c.loc[bot]
            w.loc[top] = (ct / ct.sum()).to_numpy()
            w.loc[bot] = (-cb / cb.sum()).to_numpy()
        yield t, w


def weighted_quantile_spread(
    factor_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
    q: int = 5,
    weight: str = "equal",
    mktcap_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Per-date top-minus-bottom spread return under equal- OR cap-weighting.

    Builds, per date, a dollar-neutral long-top / short-bottom portfolio (the same
    extremes as quantile_returns' 'spread' column) and returns its forward return
    series. weight='equal' equal-weights each leg (matches quantile_returns'
    spread up to qcut tie handling); weight='cap' value-weights each leg by an
    aligned market-cap panel — the convention behind most published factor
    premia, since equal weighting over-loads tiny illiquid names.

    LEAK-SAFE: weights at date t come only from factor_t and mktcap_t (known as of
    t); they multiply fwd_ret_t (realized after t). The factor and caps are never
    forward-shifted. Returns a Series indexed by date.
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    fr = fwd_ret_df.reindex_like(factor_df)
    out: Dict[pd.Timestamp, float] = {}
    for t, w in _quantile_weights(factor_df, q, weight, mktcap_df):
        r = fr.loc[t]
        pair = pd.concat([w.rename("w"), r.rename("r")], axis=1).dropna()
        if pair.empty:
            continue
        out[t] = float((pair["w"] * pair["r"]).sum())
    s = pd.Series(out, dtype=float)
    s.index.name = factor_df.index.name
    return s.sort_index()


def turnover(
    factor_df: pd.DataFrame,
    q: int = 5,
    weight: str = "equal",
    mktcap_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Two-sided per-date turnover of the long-top/short-bottom spread portfolio.

    Turnover_t = sum_a |w_t(a) - w_{t-1}(a)|, the total absolute weight traded to
    move from yesterday's target weights to today's. For a dollar-neutral spread
    with +1 long notional and -1 short notional, a full rebalance into disjoint
    names tends toward ~4 (sell 1 long + cover 1 short + buy 1 long + short 1).
    Multiply by per-unit cost to charge the spread its trading cost.

    LEAK-SAFE / CAUSAL: weights on each date use only information known AS OF that
    date (factor_t, mktcap_t); turnover_t differences today's target against the
    PRIOR date's target, so it consumes no future data and is safe to charge in a
    backtest. The first traded date has no predecessor and is omitted.

    Missing assets are treated as zero weight on the dates they are absent, so an
    entry/exit counts its full weight as traded. Returns a Series indexed by date
    (one shorter than the number of traded dates).
    """
    weights = list(_quantile_weights(factor_df, q, weight, mktcap_df))
    if len(weights) < 2:
        return pd.Series(dtype=float, name="turnover")
    dates = [t for t, _ in weights]
    full = (
        pd.DataFrame({t: w for t, w in weights})
        .T.reindex(columns=factor_df.columns)
        .fillna(0.0)
    )
    delta = full.diff().abs().sum(axis=1)
    to = delta.iloc[1:]  # first date has no prior target
    to.index = pd.Index(dates[1:], name=factor_df.index.name)
    to.name = "turnover"
    return to


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

    # --- HAC / Newey-West IC t-stat on an OVERLAPPING panel -------------------
    # Build an IC series with genuine POSITIVE serial correlation: a noisy factor
    # (so per-date IC is modest and varies) paired with an overlapping 5-step
    # cumulative forward return. Overlapping windows share component returns, which
    # induces positive autocorrelation in the IC series. The Newey-West SE then
    # exceeds the iid SE, so |t_stat_hac| <= |t_stat|.
    rng_h = np.random.default_rng(321)
    Th, Nh = F.shape
    Fz_h = cross_sectional_zscore(F)
    # weak signal + large noise so ICs are small and noisy (not pinned near 1)
    fwd_h = 0.01 * Fz_h + pd.DataFrame(
        0.08 * rng_h.standard_normal((Th, Nh)), index=F.index, columns=F.columns
    )
    overlap = 5
    fwd_overlap = fwd_h.rolling(overlap).sum().shift(-(overlap - 1))  # t..t+4 cum, leak-safe
    ic_ov = information_coefficient(F, fwd_overlap, method="spearman")
    # confirm the planted positive autocorrelation actually exists (lag-1 rho > 0)
    icv = ic_ov.dropna().to_numpy(float)
    icv_c = icv - icv.mean()
    rho1 = float((icv_c[1:] @ icv_c[:-1]) / (icv_c @ icv_c))
    assert rho1 > 0.0, f"overlapping IC must be positively autocorrelated: rho1={rho1}"
    sh = ic_summary_hac(ic_ov, nw_lags=overlap)
    s_ov = ic_summary(ic_ov)
    assert np.isclose(sh["t_stat"], s_ov["t_stat"], equal_nan=True), "HAC must reuse iid t_stat"
    assert np.isfinite(sh["t_stat_hac"]), "HAC t-stat must be finite"
    assert abs(sh["t_stat_hac"]) <= abs(sh["t_stat"]) + 1e-9, (
        f"HAC t {sh['t_stat_hac']} must be <= iid t {sh['t_stat']} on overlapping panel"
    )
    # nw_lags=0 collapses to the iid SE up to the ddof convention: the HAC long-run
    # variance uses the divide-by-n (biased) autocovariance gamma_0 (= var ddof=0)
    # while the iid t_stat uses std(ddof=1), so the HAC SE is smaller by
    # sqrt((n-1)/n) and t_stat_hac = t_stat * sqrt(n/(n-1)) at lag 0.
    sh0 = ic_summary_hac(ic_ov, nw_lags=0)
    n_ov = ic_ov.dropna().shape[0]
    assert np.isclose(
        sh0["t_stat_hac"], sh0["t_stat"] * np.sqrt(n_ov / (n_ov - 1))
    ), "nw_lags=0 must match iid t-stat up to the ddof factor"

    # --- IC decay: declines with horizon on a decaying planted signal ---------
    # Plant fwd[t] driven by factor[t] with a fast geometric fade across the next
    # few one-step returns, so longer holding horizons see weaker IC.
    rng_d = np.random.default_rng(99)
    Td, Nd = F.shape
    Fz_d = cross_sectional_zscore(F)
    decay_fac = 0.45
    fwd_decay = pd.DataFrame(0.0, index=F.index, columns=F.columns)
    for k in range(6):  # factor at t-k feeds returns k steps later, geometrically fading
        fwd_decay += (decay_fac ** k) * 0.05 * Fz_d.shift(k)
    fwd_decay += pd.DataFrame(0.01 * rng_d.standard_normal((Td, Nd)), index=F.index, columns=F.columns)
    horizons = [1, 2, 3, 5, 8]
    dec = ic_decay(F, fwd_decay, horizons, method="spearman")
    assert dec.loc[1] > dec.loc[8], f"IC must decay: h1 {dec.loc[1]} !> h8 {dec.loc[8]}"
    assert dec.loc[1] >= dec.loc[3] >= dec.loc[8] - 1e-9, "IC should decline (near-)monotonically"
    hl = ic_half_life(dec)
    assert np.isfinite(hl) and hl > 1.0, f"half-life should be finite and > 1: {hl}"

    # --- cap- vs equal-weighted quantile spread: distinct, both signed --------
    rng_c = np.random.default_rng(2024)
    mktcap = pd.DataFrame(
        np.exp(rng_c.standard_normal((Td, Nd))) * 1e6,  # lognormal caps, strictly positive
        index=F.index, columns=F.columns,
    )
    sp_eq = weighted_quantile_spread(F, fwd, q=5, weight="equal")
    sp_cap = weighted_quantile_spread(F, fwd, q=5, weight="cap", mktcap_df=mktcap)
    assert sp_eq.mean() > 0 and sp_cap.mean() > 0, "both spreads should be positive on planted signal"
    # equal-weighted spread reproduces quantile_returns' spread (same extremes)
    assert np.isclose(sp_eq.mean(), qr["spread"].dropna().mean(), atol=1e-9), "equal spread == qr spread"
    # cap weighting changes the realized spread series materially
    common = sp_eq.index.intersection(sp_cap.index)
    assert not np.allclose(sp_eq.loc[common].to_numpy(), sp_cap.loc[common].to_numpy()), (
        "cap-weighted spread must differ from equal-weighted"
    )

    # --- turnover: two-sided, non-negative, ~4 for a full disjoint rebalance ---
    to_eq = turnover(F, q=5, weight="equal")
    assert (to_eq >= -1e-12).all(), "turnover must be non-negative"
    assert len(to_eq) == F.shape[0] - 1, "turnover is one shorter than the number of traded dates"
    # equal-weight dollar-neutral spread fully reshuffles each date -> ~4
    assert 0.0 < to_eq.mean() <= 4.0 + 1e-9, f"two-sided turnover in (0,4]: {to_eq.mean()}"
    to_cap = turnover(F, q=5, weight="cap", mktcap_df=mktcap)
    assert (to_cap >= -1e-12).all() and to_cap.mean() > 0, "cap turnover non-negative and positive"

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
