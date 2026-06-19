# Cross-Sectional Factor Research

Reference for the claude-quant skill. How to build, evaluate, and combine cross-sectional factors so that a measured edge is real alpha and not leakage, an unintended risk bet, or the luckiest of many tries.

This file is the cross-sectional companion to `references/research-backtest.md` (time-series strategies, costs, fills) and `references/stats-risk.md` (multiple-testing deflation, purged CV, exact metrics). Start signal work from `templates/factor_research.py`.

Conventions used throughout:
- **Panel layout.** Data is a long panel indexed by `(date, asset)`, or a wide frame `factor[date, asset]`. "Cross-section at *t*" = the vector across assets on date *t*.
- **Simple returns** `r_t = P_t/P_{t-1} - 1`, compound multiplicatively. PPY: daily ≈ 252 (state 365 for 24-7 crypto).
- **Forward return** over horizon *h*: `fwd_h[t] = prod(1 + r[t+1 : t+h+1]) - 1` — strictly the returns of bars **after** the signal bar. The signal bar's own return is never in the forward return.
- **Information Coefficient (IC)** = per-date cross-sectional rank correlation of `factor_t` vs `fwd_h`. Spearman unless stated.
- **All cross-sectional fits (standardization, neutralization) are done PER DATE**, using only that date's cross-section. Never pool across dates, never fit on the full panel — both leak future information into the past.
- Coerce to NumPy / align indexes explicitly before arithmetic; mismatched `(date, asset)` alignment silently produces NaN or wrong pairings.

---

## 1. What a factor is; cross-sectional vs time-series alpha

A **factor** (signal, characteristic) is a number computed for each asset on each date from point-in-time information, intended to rank assets by expected forward return. Examples: 12-1 momentum, book-to-market, 5-day reversal, earnings-revision breadth, funding rate (crypto), carry (FX/futures), implied-vol skew (options).

Two distinct ways a signal can pay, and they require different evaluation:

| | **Cross-sectional alpha** | **Time-series alpha** |
|---|---|---|
| Question | Which assets out/under-perform *their peers* on date *t*? | Should I be long/short/flat *this asset* now? |
| Bet | Relative: long high-factor, short low-factor, ~dollar-neutral | Directional: net long/short exposure to the asset/market |
| Eval metric | **Information Coefficient**, quantile spread, Fama-MacBeth premium | Sharpe of the asset's own timing rule |
| Neutral to | The market (by construction, if balanced long-short) | Nothing — you are taking the market bet on purpose |
| Typical universe | Many comparable assets (equities, a futures complex) | One asset or a few |
| This file covers | **Yes — the whole document** | See `references/research-backtest.md` |

A factor can be a strong *cross-sectional* ranker (high IC) yet useless as a *time-series* timer, and vice-versa. Pick the evaluation that matches how you will trade it. The rest of this file assumes the cross-sectional case: you will rank the cross-section each rebalance and trade the spread.

**Sign convention.** Define every factor so that **higher = predicted higher forward return**. Flip the raw sign at construction (e.g. book-to-market is positive value, so high book/price is "cheap/good"; short-term reversal uses *negative* trailing return). Doing this once at the source means IC, quantile spreads, and combinations all share one orientation and you never chase a sign bug downstream.

---

## 2. Construction hygiene

### 2.1 Point-in-time inputs (the prerequisite)

Every input must be knowable at the **close of the signal date**. The two killers specific to factor research:

- **Fundamentals lagged to publication, not period-end.** A Q1 (Mar 31) earnings figure is typically known mid-May. Joining it on Mar 31 is look-ahead. Use as-of joins on the *publication/filing date* (see `references/data.md` and `templates/data_loader.py`).
- **Survivorship-correct, point-in-time universe.** The cross-section on date *t* must be the assets that were *tradable and in-universe on t* — including names later delisted/acquired, and excluding names not yet listed or not yet index members. Ranking against today's survivors is fiction (Section 10).

### 2.2 Cross-sectional standardization (per date)

Raw factor values are not comparable across dates or across factors (different units, scales, and dispersion). Standardize **within each date's cross-section** so that ranks/exposures are comparable and combinable.

**Z-score** (de-mean and scale by the cross-sectional std of that date):
```python
def cs_zscore(f):                      # f: Series indexed by asset, one date
    return (f - f.mean()) / f.std(ddof=1)
```
Sensitive to outliers — winsorize first (2.3). After z-scoring, a dollar-neutral long-short weight is simply `w_i = z_i / sum(|z|)`.

**Rank / quantile transform** (distribution-free, robust to outliers and monotone transforms):
```python
def cs_rank(f):                        # uniform ranks in [0, 1]
    return f.rank(pct=True)

def cs_rank_centered(f):               # centered to [-0.5, +0.5]
    return f.rank(pct=True) - 0.5
```
Rank-based factors throw away cardinal information (the gap between #1 and #2) but are far more robust and are what Spearman IC implicitly uses. A common production choice is a **Gaussian rank transform** (rank → uniform → normal quantile via `Φ⁻¹`), which gives a well-behaved, outlier-immune, roughly-normal exposure.

Apply per date with a groupby:
```python
# long panel: columns [date, asset, raw]
panel["z"] = panel.groupby("date")["raw"].transform(
    lambda f: (f - f.mean()) / f.std(ddof=1))
```

### 2.3 Winsorization

Clip extreme values *before* z-scoring so a single bad print or a true outlier doesn't dominate the cross-section. Clip in the cross-section of each date, by quantile or by MAD:

```python
def winsorize_quantile(f, lo=0.01, hi=0.99):
    return f.clip(f.quantile(lo), f.quantile(hi))

def winsorize_mad(f, k=5):
    med = f.median()
    mad = (f - med).abs().median() * 1.4826      # ~std for normal data
    return f.clip(med - k * mad, med + k * mad)
```
MAD-based clipping is preferable when the cross-section is itself fat-tailed (quantile clips still let in heavy tails inside the kept range). Winsorize, *then* z-score — not the reverse.

### 2.4 Log / sign transforms

Many raw characteristics are heavily skewed (market cap, volume, dollar turnover, B/M). Compress before standardizing so the tail doesn't swamp the rank:
- **`log`** for strictly-positive, multiplicatively-distributed quantities: `size = log(market_cap)`, `log(dollar_volume)`.
- **Signed log** for variables spanning zero with fat tails: `sign(x) * log1p(|x|)`.
- **Ratios over levels.** Prefer scale-free ratios (B/M, earnings yield, momentum as a return) so the factor doesn't just re-encode size.

Order of operations for one factor: `point-in-time raw → transform (log/sign/ratio) → winsorize → cross-sectional standardize → (optional) neutralize`.

---

## 3. Neutralization

A raw factor often carries **unintended bets**. Value loads on certain sectors; momentum drifts net long the market in trends; almost everything correlates with size. If you don't remove these, your "value IC" is partly a sector timing bet and your backtested P&L is partly market beta. Neutralization strips known exposures so the residual factor is a *pure* relative bet.

**Mechanism: per-date cross-sectional regression, keep the residual.** On each date, regress the factor across assets on the exposures you want to remove; the residual is the neutralized factor.

```python
import numpy as np

def neutralize_cs(f, exposures):
    """f: (n_assets,) factor for ONE date. exposures: (n_assets, k) design
    matrix for that SAME date (include an intercept column of ones).
    Returns the residual: factor orthogonal to the exposures, this date only.
    """
    f = np.asarray(f, float)
    X = np.asarray(exposures, float)
    beta, *_ = np.linalg.lstsq(X, f, rcond=None)
    return f - X @ beta
```

Common things to neutralize against:
- **Sector / industry** — dummy variables (GICS sector one-hots). Removes "this is just a sector tilt." For dummies, the residual is the factor demeaned *within each sector*.
- **Market beta** — the asset's estimated β as a regressor; residual is beta-neutral so long-short P&L isn't market exposure in disguise.
- **Size** — `log(market_cap)`; removes the pervasive small-cap tilt that most factors smuggle in.
- **Other style factors** you don't want to double-count (e.g. neutralize a new factor against value+momentum to test for *incremental* alpha — see Section 8).

Convenience for dummy/group neutralization (demean within group, per date):
```python
panel["f_neutral"] = (panel.groupby(["date", "sector"])["z"]
                           .transform(lambda s: s - s.mean()))
```

### 3.1 Leakage risk in neutralization (critical)

**Fit the neutralization cross-sectionally, PER DATE, using only that date's data.** The classic, silent bug is fitting one regression on the *pooled* panel (all dates stacked) or, worse, on the full sample including the future — this lets future cross-sections determine today's residual.

```python
# WRONG — pooled across all dates: the betas are estimated using future
# cross-sections, so today's neutralized factor depends on the future.
beta = np.linalg.lstsq(X_all_dates, f_all_dates, rcond=None)[0]
resid = f_all_dates - X_all_dates @ beta

# RIGHT — independent regression within each date's cross-section.
# (_design(g) builds that date's exposure matrix, intercept included.)
panel["f_neutral"] = panel.groupby("date", group_keys=False).apply(
    lambda g: pd.Series(neutralize_cs(g["z"], _design(g)), index=g.index))
```

The same rule governs the **exposures themselves**: a beta or sector membership used on date *t* must be estimated/known from data up to *t* (rolling beta on a trailing window; point-in-time sector classification — GICS reclassifications happen). Using a full-sample beta or *today's* sector label is look-ahead.

Standardization (Section 2) has the identical requirement — `mean`/`std`/quantiles are per-date cross-sectional statistics, never full-panel. Using `sklearn.StandardScaler().fit(panel)` over the whole panel is the same leak.

---

## 4. Information Coefficient (IC)

The IC measures how well the factor *ranks* the cross-section against what actually happens next. It is the primary, cost-free first screen for a cross-sectional factor.

### 4.1 Definition

For each date *t*, take the cross-section of the factor `factor_t` (one value per asset) and the cross-section of **forward** returns `fwd_h` over *t+1 … t+h*. The IC on date *t* is their cross-sectional correlation:

```
IC_t = corr_assets( factor_t , fwd_h(t) )
```

This produces a **time series of ICs**, one per rebalance date. The forward return must start the bar *after* the signal date (Section 10 pitfall: including the signal bar).

### 4.2 Pearson vs Spearman

- **Spearman (rank IC)** — correlation of ranks. Robust to outliers and to monotone non-linearity; matches how you trade if you sort into quantiles. **Default for factor screening.**
- **Pearson (linear IC)** — correlation of raw (standardized) values. Captures cardinal information and is the natural metric if you size positions *proportionally* to the factor. More sensitive to outliers, so winsorize first.

Report Spearman by default; report Pearson too when positions are proportional rather than ranked.

```python
import pandas as pd
from scipy.stats import spearmanr

def ic_timeseries(panel, factor_col, fwd_col, method="spearman", min_names=20):
    """panel long-indexed by [date, asset] with factor and forward-return cols
    (date is an index LEVEL, not a column — avoids the pandas 2.x
    grouping-column deprecation in groupby.apply).
    Returns a Series of per-date cross-sectional ICs.
    """
    def _ic(g):
        x, y = g[factor_col], g[fwd_col]
        ok = x.notna() & y.notna()
        if ok.sum() < min_names:           # too few names -> unstable, drop
            return np.nan
        if method == "spearman":
            return spearmanr(x[ok], y[ok]).correlation
        return x[ok].corr(y[ok])           # Pearson
    return panel.groupby("date").apply(_ic).dropna()
```

### 4.3 Summary statistics of the IC series

Let `IC` be the per-date series of length `n_dates`.

- **Mean IC** — `IC.mean()`. The average ranking power. Equity daily/weekly single-factor mean ICs are typically small: **0.02–0.05 is already a useful factor; 0.05–0.10 is strong.** Anything ≥ 0.15 on a single raw factor over many names invites a leakage audit.
- **IC IR (information ratio of the IC)** — consistency of the edge:
  ```
  IC_IR = mean(IC) / std(IC, ddof=1)
  ```
  This is the breadth-adjusted quality of the signal (closely related to the strategy IR via the fundamental law of active management, `IR ≈ IC · sqrt(breadth)`). IC_IR around 0.3–0.5 (per period) is good.
- **IC t-stat** — significance of a non-zero mean IC, treating dates as the sample:
  ```
  t = IC_IR * sqrt(n_dates)          # = mean(IC) / (std(IC)/sqrt(n_dates))
  ```
  This is a one-sample t-test that mean IC = 0. **Caveat:** ICs are autocorrelated when forward horizon *h* > rebalance spacing (overlapping windows), which inflates the t-stat — use non-overlapping forward windows, or apply a Newey-West / HAC correction (Section 7), or annualize honestly.
- **Risk-adjusted IC** — the same `mean/std` ratio is itself the risk-adjusted IC; some desks report it annualized as `IC_IR * sqrt(periods_per_year)` to compare across rebalance frequencies. State the frequency. (Annualizing this way assumes roughly iid ICs — the same overlap caveat applies.)

```python
def ic_summary(ic, ppy=252):
    ic = ic.dropna()
    n = len(ic)
    mean, sd = ic.mean(), ic.std(ddof=1)
    icir = mean / sd
    return {
        "mean_ic": mean,
        "ic_std": sd,
        "ic_ir": icir,                       # per-period, risk-adjusted IC
        "ic_t_stat": icir * np.sqrt(n),      # H0: mean IC = 0 (iid assumption)
        "ic_ir_ann": icir * np.sqrt(ppy),    # comparable across frequencies
        "hit_rate": (ic > 0).mean(),         # fraction of dates with IC > 0
        "n_dates": n,
    }
```

A factor that passes here (positive, significant, consistent mean IC) earns a costed quantile backtest. A factor that fails here will not be rescued by clever portfolio construction.

---

## 5. IC decay and signal horizon

The IC depends on the forward horizon *h*. Computing **IC as a function of h** tells you *how fast the signal pays out* and therefore *how often to rebalance*.

```python
def fwd_return_per_asset(panel, ret_col, h):
    """Forward h-bar simple return, labelled at the SIGNAL bar t and covering
    bars t+1 … t+h only (signal bar excluded). panel must be sorted by
    (asset, date). For each asset:
        fwd_h[t] = prod(1 + r[t+1 : t+h+1]) - 1
    Built as a trailing h-bar rolling product (labelled at the right edge,
    i.e. bar t+h) then shifted back by h rows to land on bar t.
    """
    g = panel.groupby("asset", group_keys=False)[ret_col]
    return (g.apply(lambda r: (1 + r).rolling(h).apply(np.prod, raw=True)
                                      .shift(-h) - 1))

def ic_decay(panel, factor_col, ret_col, horizons=(1, 2, 3, 5, 10, 21, 42, 63)):
    """ret_col = per-bar simple return. panel long-indexed by [date, asset]
    and SORTED by (asset, date). Builds forward returns per horizon PER ASSET
    (signal bar excluded, correctly aligned to the signal bar), then mean IC
    per h.
    """
    out = {}
    for h in horizons:
        tmp = panel.assign(_fwd=fwd_return_per_asset(panel, ret_col, h))
        out[h] = ic_timeseries(tmp, factor_col, "_fwd").mean()
    return pd.Series(out, name="mean_ic")
```
**Alignment, the easy place to get this wrong.** `rolling(h)` labels each window at its *right* edge (bar `t+h`), so you must `.shift(-h)` to move the `t+1 … t+h` product back onto the signal bar `t`. The tempting one-liner `(1+r).shift(-1).rolling(h).prod()` is **wrong for `h > 1`**: it labels the product at bar `t+1` (off by `h-1`) and the window it actually covers is `t-h+2 … t+1`, which *includes the signal bar* — exactly the leakage Section 10 warns about. The two forms coincide only at `h = 1`. Also: the rolling product is per-asset and order-dependent, so the panel must be sorted by `(asset, date)` first, or windows will straddle assets.

Read the decay curve:
- **Rising then flat / slow decay** — slow-moving value/quality factor; longer horizon, less frequent rebalancing, lower turnover. The IC may keep rising for weeks.
- **Peaks at *h*=1–3 then collapses** — fast signal (short-term reversal, intraday, news/flow). Must trade quickly; turnover and costs dominate the decision (Section 6, and `references/research-backtest.md` for cost modeling).
- **IC half-life** — the horizon at which IC falls to half its peak; a compact summary of "how long the alpha lives." Fit an exponential `IC(h) ≈ IC_0 · exp(-h/τ)` and report `τ` (and `half_life = τ·ln 2`), or just read it off the curve.

**Choosing rebalance frequency** is a trade-off: rebalancing far *more often* than the signal's horizon mostly churns turnover (cost) for little fresh information; rebalancing far *less* often than the half-life lets the edge decay away before you harvest it. Rebalance near the IC half-life, then confirm net-of-cost in the quantile backtest. (For a formal cost-vs-decay optimum and turnover control, see `references/research-backtest.md`.)

---

## 6. Quantile / decile portfolios

The IC says the factor ranks; quantile portfolios say *how the payoff is shaped* and *what it earns net of costs*. Sort the cross-section into *q* buckets each rebalance and track each bucket's forward return.

```python
def quantile_returns(panel, factor_col, fwd_col, q=5, weight="equal"):
    """Per-date: bucket assets by factor quantile, average forward return per
    bucket. weight='equal' or 'cap' (needs a 'mktcap' column).
    Returns DataFrame indexed by date, columns = quantiles 1..q (1 = lowest
    factor, q = highest). Long-short = Qq - Q1.
    """
    def _bucket(g):
        g = g.dropna(subset=[factor_col, fwd_col])
        if len(g) < q * 2:
            return None
        g = g.assign(_b=pd.qcut(g[factor_col], q, labels=False,
                                duplicates="drop") + 1)
        if weight == "cap":
            r = g.groupby("_b").apply(
                lambda b: np.average(b[fwd_col], weights=b["mktcap"]))
        else:
            r = g.groupby("_b")[fwd_col].mean()
        return r
    # Build per-date rows explicitly. Do NOT use groupby(...).apply(_bucket):
    # when every date yields the same buckets, apply returns a DataFrame (not a
    # Series-of-Series), and the .dropna()/.unstack() chain then mangles the
    # shape. A dict -> DataFrame.T is shape-stable and skips short dates.
    rows = {d: r for d, g in panel.groupby("date")
            if (r := _bucket(g)) is not None}
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "date"
    out.columns = [f"Q{int(c)}" for c in out.columns]
    return out
```

What to look at:
- **Long-short spread (top-minus-bottom).** `Qq − Q1` is the dollar-neutral factor portfolio. Its mean, Sharpe (annualize with the rebalance PPY), and drawdown are the headline. This is your tradable proxy.
- **Monotonicity.** A real factor's bucket means should increase ~monotonically from Q1 to Qq. A non-monotone pattern where only the extreme buckets pay (or only the short leg works) is fragile and often a small-cap/illiquidity artifact. Check with a monotonicity score (e.g. Spearman of bucket index vs bucket mean return, or count of correctly-ordered adjacent steps).
- **Equal vs cap weight.** Equal-weight overweights small/illiquid names and usually *overstates* the spread; cap-weight is closer to tradable. Report both — a spread that only exists equal-weighted lives in the micro-caps and won't survive costs/capacity (Sections 9–10).
- **Long vs short leg.** Decompose the spread. If all the alpha is in the short leg, borrow availability/cost and short-sale constraints decide whether it's real (crypto/FX differ — shorting may be cheap or structurally hard depending on venue).
- **Turnover.** Per rebalance, the notional traded as a fraction of book:
  ```python
  def turnover(weights):          # weights: DataFrame [date x asset], rows sum to 1 abs
      # sum|Δw| = total notional traded = sells + buys (TWO-sided / gross turnover).
      return weights.diff().abs().sum(axis=1).iloc[1:]
  ```
  `sum|Δw|` is **two-sided (gross) turnover** — it counts both the sells and the buys, i.e. the total notional that crosses the spread. One-sided turnover is half of this. Be explicit about which you mean when costing: if your cost is quoted *per unit traded one way* (a half-spread + fee), multiply by **two-sided** turnover; if it is a *round-trip* cost (full spread for a buy-then-sell), multiply by **one-sided** turnover (`0.5 · sum|Δw|`). Mixing the two double-counts or halves the drag. **A high-IC factor with high turnover can be net-negative after costs.** This is the bridge to the full costed backtest in `references/research-backtest.md`.

Quantile P&L uses the **same forward-return alignment** as the IC: positions formed from `factor_t` earn returns over *t+1…*. Forming buckets and applying the *same-bar* return is the look-ahead bug from Section 10.

---

## 7. Fama-MacBeth

Fama-MacBeth (1973) estimates the **premium** earned per unit of factor exposure, with a clean treatment of cross-sectional correlation. Two steps:

1. **Cross-sectional regression per period.** On each date *t*, regress forward returns on the factor exposure(s) across assets:
   ```
   fwd_i,t = a_t + λ_t · factor_i,t + ε_i,t        (one regression per date)
   ```
   Collect the slope series `λ_t` (the per-date factor premium) — and the intercept `a_t`.
2. **Average over time with proper standard errors.** The estimated premium is `λ̄ = mean_t(λ_t)`. Its t-stat uses the time-series variability of the slopes; because slopes are **autocorrelated** (overlapping forward windows, persistent exposures), use **Newey-West (HAC)** standard errors rather than the naive `std/sqrt(T)`.

```python
import numpy as np, pandas as pd
import statsmodels.api as sm

def fama_macbeth(panel, exog_cols, fwd_col, nw_lags=None):
    """Stage 1: per-date cross-sectional OLS of forward return on exposures.
    Stage 2: average the slopes, Newey-West t-stats on the slope time series.
    exog_cols: list of exposure columns (intercept added automatically).
    """
    lambdas = []
    for date, g in panel.groupby("date"):
        g = g.dropna(subset=exog_cols + [fwd_col])
        if len(g) < len(exog_cols) + 5:
            continue
        X = sm.add_constant(g[exog_cols])
        res = sm.OLS(g[fwd_col], X).fit()
        lambdas.append(res.params.rename(date))
    lam = pd.concat(lambdas, axis=1).T          # rows = dates, cols = params
    T = len(lam)
    if nw_lags is None:
        nw_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))   # rule of thumb
    out = {}
    for c in lam.columns:                       # Newey-West mean t-stat per coef
        m = sm.OLS(lam[c].values, np.ones(T)).fit(
            cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        out[c] = {"premium": m.params[0], "t_stat": m.tvalues[0],
                  "nw_se": m.bse[0]}
    return pd.DataFrame(out).T, lam
```

Notes:
- Fama-MacBeth and the IC are close cousins: the per-date slope on a *standardized* single factor is a scaled per-date IC. FM generalizes cleanly to **multiple exposures at once** (estimate each factor's premium controlling for the others) and gives you the HAC-correct t-stat directly.
- **Newey-West lags** should cover the overlap in the forward return (at least `h − 1` lags if using overlapping *h*-period forward returns) plus exposure persistence. Under-lagging re-introduces the inflated t-stat the method is meant to fix.
- The same per-date / no-pooling discipline applies: stage 1 is strictly within-date.

---

## 8. Combining factors

Most single factors are weak; the edge comes from combining several into a composite that ranks better and more consistently than any one. Methods, roughly in order of robustness:

**1. Standardize-then-average (equal-weight composite).** Z-score each factor *per date* (Section 2), then average. Simple, robust, hard to overfit, and a strong baseline:
```python
composite = panel[["z_value", "z_mom", "z_quality"]].mean(axis=1)
```
Equal weights are remarkably hard to beat out-of-sample because estimated optimal weights are noisy.

**2. IC-weighting.** Weight each factor by its (rolling, *trailing*) IC or IC_IR so better/steadier signals count more:
```
w_k ∝ IC_IR_k        (estimated on a trailing window, never the full sample)
composite = sum_k w_k · z_k
```
The weights must be estimated **only from past ICs** (rolling window, lagged), or you leak future performance into the historical composite. This adds parameters; prefer light shrinkage toward equal weight.

**3. Orthogonalization / Gram-Schmidt.** When factors overlap (value and quality both load on profitability; momentum and low-vol co-move), naive averaging double-counts the shared component. Orthogonalize so each factor contributes only its *incremental* information — sequentially regress each factor on the already-included ones (per date) and keep residuals:
```python
def gram_schmidt_cs(F):
    """F: (n_assets, k) standardized factors for ONE date, ordered by priority.
    Returns residualized factors, each orthogonal to all earlier ones."""
    F = np.asarray(F, float)
    out = np.empty_like(F)
    out[:, 0] = F[:, 0]
    for j in range(1, F.shape[1]):
        X = out[:, :j]
        beta, *_ = np.linalg.lstsq(X, F[:, j], rcond=None)
        out[:, j] = F[:, j] - X @ beta          # incremental part only
    return out
```
Order matters (the first factor keeps its full variance); order by economic priority or IC. This is also how you test a *new* factor for incremental alpha: residualize it against the established factors and check whether the residual still has IC.

**4. Multicollinearity.** Highly correlated exposures make any *fitted* weighting (regression, mean-variance, FM with many factors) unstable — small data changes flip signs and blow up weights. Diagnose with the cross-sectional correlation matrix and VIF; treat by dropping/merging near-duplicates, orthogonalizing, or shrinking weights toward equal. When in doubt, equal-weight a deduplicated set.

**Multiple testing across many factors.** Screening dozens or hundreds of candidate factors and keeping the high-IC ones is exactly the data-snooping problem: the best IC out of *N* tried is biased high even if all are worthless. **The number of factors tried is part of `n_trials`.** Deflate accordingly — Benjamini-Hochberg / Bonferroni on the IC t-stats, and the **Deflated Sharpe Ratio** on any backtested composite. See `references/stats-risk.md` (Sections on FWER/FDR, PSR/DSR, and PBO). Report the *full* count of factors and variants examined, not just the survivors.

---

## 9. Factor decay, crowding, and capacity

A factor that worked historically can erode going forward for reasons unrelated to any coding error:

- **Crowding / arbitrage decay.** Once a factor is published and traded, capital flows in and the premium compresses; the post-publication Sharpe of academic factors is materially lower than in-sample (McLean & Pontiff). Detect via a declining trailing IC / rolling spread, rising correlation with known factor indices, and crowded-positioning metrics (short interest, factor-ETF flows). Don't extrapolate an in-sample premium forward unchanged.
- **Capacity.** The dollar-neutral spread you can *actually* harvest is bounded by liquidity: the small/illiquid names that often drive the gross spread can't absorb size without moving the price. Estimate capacity from per-name ADV and your participation cap, and re-run the quantile backtest with a tradable (liquidity-screened, cap-weighted) universe and realistic costs. A spread that lives in micro-caps has near-zero real capacity.
- **Regime dependence.** Many factors are conditional (value and momentum have long, painful drawdowns; low-vol depends on the rate regime). Examine IC/spread by sub-period and regime, not just the full-sample average.

These move a factor from "interesting IC" to "tradable, sized, net-of-cost edge." The costing and capacity machinery lives in `references/research-backtest.md`.

---

## 10. Pitfalls (detect / fix)

| Pitfall | How it sneaks in | Detect | Fix |
|---|---|---|---|
| **Look-ahead in factor inputs** | Fundamentals joined on period-end, not publication date; restated/PIT-incorrect data; today's sector/beta used for the past | Suspiciously high mean IC (≥ 0.15 single raw factor); IC barely changes when you add the publication lag | As-of join on filing/publication date; rolling/PIT exposures; `templates/data_loader.py` |
| **Forward-return misalignment** | Forward return mislabelled by `h-1` rows (e.g. `shift(-1).rolling(h)` instead of `rolling(h).shift(-h)`); off-by-one `shift`; date/asset index misalignment in the join | IC collapses (often inverts) when you correctly align; mean IC drifts oddly across horizons; sanity-check a few `(date, asset)` rows by hand | `fwd_h(t)` uses returns `t+1 … t+h` only, labelled at `t`: `rolling(h).prod().shift(-h)` (Section 5); align on `(date, asset)` explicitly |
| **Including the signal bar in the forward return** | `rolling(h).prod()` with no forward shift (window ends at `t`), or `shift(-1).rolling(h)` for `h>1` (window covers `t-h+2 … t+1`), so the *t*-bar return (correlated with the *t* factor) leaks into `fwd` | A spike in IC at *h*=1 that vanishes after excluding the current bar; spuriously high IC at small `h` | Build forward returns from strictly future bars `t+1 … t+h` (Section 5 code) |
| **Neutralization leakage** | Regressing the factor on exposures using the **pooled** panel or full sample instead of per date | Residual factor's IC depends on data outside the as-of date; pooled vs per-date residuals differ a lot | Fit neutralization **within each date's cross-section only** (Section 3.1) |
| **Standardization leakage** | `StandardScaler.fit` on the whole panel; `center=True` rolling stats; full-sample mean/std/quantiles used per row | Same look-ahead test as above; full-sample stats used to scale past rows | Per-date cross-sectional mean/std/quantiles; or fit scalers on train only |
| **Quantile look-ahead** | Bucket thresholds (`qcut` breakpoints) computed on the full sample / all dates at once | Bucket boundaries on date *t* depend on future cross-sections | Compute `qcut` breakpoints **within each date's cross-section** (Section 6) |
| **Survivorship bias** | Ranking against *today's* universe / index members; missing delisted names and delisting returns | Spread weakens sharply on a PIT universe; short leg loses its "winners" (the blowups that delisted) | Point-in-time, survivorship-correct universe with delisting returns (`references/data.md`) |
| **Micro-cap / illiquid artifact** | Equal-weight spread driven by tiny names that can't be traded | Spread collapses cap-weighted or after a liquidity screen; capacity is tiny | Liquidity-filter the universe; report cap-weighted spread; estimate capacity (Section 9) |
| **Multiple testing across factors** | Keeping the best IC of many candidates/variants without counting trials | Survivor's IC t-stat doesn't survive BH/Bonferroni; backtest DSR < ~0.95 | Count *every* factor/variant in `n_trials`; deflate (BH/Bonferroni, DSR) — `references/stats-risk.md` |
| **Overlapping-window inflated t-stats** | Using overlapping *h*-period forward returns then treating dates as iid | IC/FM t-stat shrinks under HAC or with non-overlapping windows | Newey-West (≥ `h−1` lags) or non-overlapping forward windows (Sections 4.3, 7) |
| **Sign / orientation bug** | Raw factor not oriented "higher = better"; flipped somewhere downstream | Negative mean IC where theory says positive; long leg underperforms short | Fix the sign once at construction so higher = predicted higher return (Section 1) |

**Pre-flight before trusting any factor result:** (1) every input is point-in-time; (2) forward return excludes the signal bar and is aligned on `(date, asset)`; (3) every cross-sectional fit (standardize, neutralize, quantile breakpoints) is per-date, no pooling; (4) universe is survivorship-correct and liquidity-screened; (5) IC and quantile P&L use the same lagged alignment; (6) t-stats are HAC-corrected for overlap; (7) `n_trials` is counted and the result is deflated. See `references/pitfalls.md` for the cross-cutting trap checklist and `references/stats-risk.md` for deflated metrics.

---

## See also

- `templates/factor_research.py` — runnable, self-checking implementations of the IC time series, IC decay, per-date neutralization/standardization, quantile/long-short construction, and Fama-MacBeth from this file.
- `references/stats-risk.md` — multiple-testing deflation (FWER/FDR, Probabilistic & Deflated Sharpe, PBO), purged+embargoed CV, exact Sharpe/IR conventions and the Lo (2002) autocorrelation correction.
- `references/research-backtest.md` — turning the long-short factor into a costed, capacity-aware backtest; rebalance/turnover trade-offs; cost models.
- `references/data.md` + `templates/data_loader.py` — point-in-time joins, survivorship-correct universes, corporate-action adjustment.