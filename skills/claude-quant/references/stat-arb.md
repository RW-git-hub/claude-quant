# Statistical Arbitrage & Pairs Trading

Market-neutral relative-value trading: profit from the *relative* mispricing of
related instruments while hedging out shared (market/sector) risk. The core
statistical object is **cointegration**, not correlation. This file gives the
formulas, the testing machinery, spread construction, the mean-reversion model,
trading rules, basket extensions, and a detect/fix pitfalls table.

Conventions (consistent with the rest of this skill): simple returns compound
multiplicatively; positions are lagged versus the returns they earn
(`pnl_t = pos.shift(1) * ret_t`); annualized Sharpe = `mean(excess)/std(excess, ddof=1)*sqrt(ppy)`,
daily `ppy=252` (matches `templates/metrics.py`); any estimated quantity (hedge
ratio, spread mean/std, cointegration test) must be computed on data strictly
**prior** to the bar it is used on — rolling or expanding, never full-sample.

**Pandas safety note (applies to every snippet below).** Prices in this skill
arrive as pandas Series/DataFrames. `statsmodels` `fit().params` is a pandas
Series with *string* labels (`const`, `x1`/column name), so `params[1]` does
**label** lookup and raises `KeyError`. Likewise positional indexing `r[t]` on a
Series is label-based and silently wrong. Coerce inputs with `np.asarray(...)`
before positional work, and use `.iloc[1]` (or `.params.iloc[1]`) when you must
index a results Series. This mirrors the convention in
`references/stats-risk.md`.

See also: `templates/validation.py` (purged + embargoed walk-forward),
`templates/costs.py` (borrow/funding/slippage), `templates/metrics.py`,
`templates/factor_research.py` (PCA factors for basket residual reversion),
`references/stats-risk.md` (multiple testing, deflated/probabilistic Sharpe),
and `references/transaction-costs.md`. A companion `templates/pairs_trading.py`
(end-to-end pair workflow) is the intended home for the routines sketched here;
treat the snippets below as the authoritative spec rather than assuming a
finished file exists.

---

## 1. Correlation vs Cointegration

These are different things, and conflating them is the most common conceptual
error in pairs trading.

**Correlation** is a property of *returns*. `corr(r_x, r_y)` measures whether two
series tend to move up and down together over short horizons. Two assets can be
highly correlated yet drift arbitrarily far apart in *level* — correlation says
nothing about whether the price *spread* stays bounded.

**Cointegration** is a property of *price levels*. Two (or more) series that are
each individually non-stationary (integrated of order 1, `I(1)` — i.e. random
walks whose levels wander without reverting) are *cointegrated* if some linear
combination of their levels is **stationary** (`I(0)`, mean-reverting):

```
spread_t = y_t - beta * x_t   is stationary (mean-reverting)
```

That stationary spread is exactly what makes mean-reversion trading possible: when
the spread deviates from its long-run mean, there is a statistical tendency to
revert, and you trade that reversion.

Why correlation is not enough:

- **Correlated, not cointegrated:** two stocks both trend up with the market.
  High return correlation, but the spread `y - beta*x` itself trends (random
  walk) — no stable level to revert to. Trading it is trading a drift, not a
  mean reversion.
- **Cointegrated, not (highly) correlated:** the spread reverts on a horizon
  different from the daily return window, so daily return correlation can be
  modest while the levels are tightly tethered.

Detect/fix in one line: **test the spread for stationarity (cointegration); do
not select pairs on return correlation alone.** Correlation is a fine *prior* for
choosing candidates (same sector, same factor exposure), but the trade thesis
must be a cointegration test.

---

## 2. Testing for Cointegration

### Engle–Granger two-step (for a single pair, one cointegrating vector)

1. **Estimate the hedge ratio** by OLS on *levels*:
   `y_t = alpha + beta * x_t + e_t`. The fitted `beta` is the hedge ratio.
2. **Test the residual `e_t` for stationarity** with an Augmented Dickey–Fuller
   (ADF) test. Null `H0`: a unit root (non-stationary, *not* cointegrated).
   Reject `H0` (small p-value) => residual is stationary => the pair is
   cointegrated.

Critical: because `beta` was *estimated*, you cannot use standard ADF critical
values on the residual — use **cointegration-specific** critical values
(Engle–Granger / MacKinnon). `statsmodels.tsa.stattools.coint(y, x)` does this
correctly and returns a 3-tuple `(eg_t_stat, p_value, crit_values)`; prefer it
over manually running `adfuller` on an OLS residual.

```python
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import numpy as np

y = np.asarray(y, float); x = np.asarray(x, float)   # coerce: avoid label indexing

# Step 1: hedge ratio (note OLS asymmetry, section 3)
beta = sm.OLS(y, sm.add_constant(x)).fit().params[1]   # numpy in -> positional [1] is safe
# If x/y are pandas Series, use .params.iloc[1] instead (params is a labelled Series).

# Step 2: Engle-Granger cointegration test (correct EG critical values)
t_stat, p_value, crit = coint(y, x)        # H0: no cointegration; 3-tuple
# p_value < 0.05 -> reject no-cointegration -> tradeable spread (IN-SAMPLE ONLY)
```

EG is **asymmetric**: regressing `y~x` vs `x~y` can give different test outcomes
near the boundary. Report both orderings or use Johansen, which is symmetric.

### Johansen test (>2 series, multiple cointegrating vectors)

For a basket of `n` series, there can be up to `n-1` independent cointegrating
relationships. The Johansen test (a VECM-based likelihood-ratio test) estimates
the **number of cointegrating vectors** (the *rank* `r`) and returns the vectors
themselves (the eigenvectors), which are the hedge weights.

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# prices: T x n array of LEVELS; det_order=0 (no deterministic trend), k_ar_diff=1
res = coint_johansen(np.asarray(prices, float), det_order=0, k_ar_diff=1)
# Trace statistic vs critical values -> infer rank r (number of cointegrating vectors)
# res.lr1 = trace stat (length n); res.cvt = trace crit values, shape (n, 3) at 90/95/99%;
# res.evec columns = cointegrating vectors. (res.lr2 / res.cvm are the max-eigenvalue test.)
hedge_weights = res.evec[:, 0]   # first cointegrating vector (largest eigenvalue)
```

Use Johansen when trading a basket (e.g. one stock vs a weighted combination of
peers, or an ETF vs its replicating basket).

### Caveats (these bite in production)

- **Low power.** Cointegration tests routinely fail to reject `H0` even when a
  real (but weak/slow) relationship exists, and they reject spuriously on short
  samples. Need long, clean samples; treat borderline p-values skeptically.
- **Structural breaks.** A merger, index reconstitution, capital-structure
  change, or regime shift breaks the relationship. Standard tests assume a
  *constant* relationship over the window; a single break makes a truly stable
  pair look non-cointegrated and a broken pair look fine on a stale window.
- **Multiple testing.** Screening hundreds of candidate pairs and keeping the
  ones with `p<0.05` guarantees false positives (~5% of *random* pairs pass at
  the 5% level). This is the dominant failure mode of pair-mining — see section 7
  and `references/stats-risk.md`.
- **Estimated `beta` => use EG critical values**, not plain ADF (above).
- **Non-stationary inputs only.** Cointegration is defined for `I(1)` series.
  Confirm each leg is `I(1)` (ADF fails to reject unit root on the level, rejects
  on the first difference) before interpreting a cointegration test on the pair.

---

## 3. Spread Construction & the Hedge Ratio

### OLS hedge ratio (and its asymmetry)

The hedge ratio `beta` is the number of units of `x` to short per unit of `y` so
that the spread has reduced exposure to the common factor:

```
spread_t = y_t - beta * x_t          (often also subtract intercept alpha)
```

Note: an OLS/TLS hedge ratio gives a *unit*-neutral spread (1 unit `y` vs `beta`
units `x`). That is **not** automatically dollar-neutral or beta-neutral — for
dollar neutrality scale legs by price (`notional_x = notional_y`); for
factor/beta neutrality use the factor betas (section 6). Be explicit about which
neutrality you are imposing.

OLS minimizes squared errors in **one direction only**. Regressing `y~x` puts all
the noise on `y`; regressing `x~y` puts it on `x`. The two betas are **not**
reciprocals: they satisfy `beta_{y~x} * beta_{x~y} = R^2`, i.e.
`beta_{y~x} = R^2 / beta_{x~y}`. So they differ from being reciprocals by a
factor of `R^2` (verified numerically). With two noisy legs (the usual case),
neither is "right."

**Total least squares / orthogonal (Deming) regression** (first principal
component of the two centered series) is the symmetric alternative — it minimizes
*perpendicular* distance and treats both legs as noisy:

```python
import numpy as np

def tls_hedge_ratio(y, x):
    """Orthogonal (Deming/TLS) regression via PCA of the 2-col matrix.
    NOTE: PCA on raw centered data is SCALE-SENSITIVE. This is the standard TLS
    estimator and assumes the two legs have comparable error variances. If the
    legs live on very different scales/vols, standardize first (divide each by
    its std) and rescale the resulting slope back, or use a known error-variance
    ratio (general Deming). Do NOT blindly trust the slope on mismatched scales.
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    M = np.column_stack([x - x.mean(), y - y.mean()])
    _, _, vh = np.linalg.svd(M, full_matrices=False)
    vx, vy = vh[0]          # first principal direction (x-comp, y-comp)
    return vy / vx          # slope dy/dx = hedge ratio of y on x
```

Use TLS when both legs are comparably noisy and you care about the *economic*
hedge ratio. Use OLS when EG cointegration testing is the goal (its critical
values assume the OLS residual). Whichever you pick, **estimate it
out-of-sample only** — see the look-ahead pitfall in section 7.

### Dynamic / time-varying hedge ratio

A fixed full-sample `beta` is both look-ahead-biased and wrong when the
relationship drifts. Two standard remedies:

- **Rolling regression:** re-estimate `beta` over a trailing window (window length
  ~ a few half-lives, section 4). Simple, transparent, but laggy and window-size
  sensitive.
- **Kalman filter:** treat `beta_t` (and optionally `alpha_t`) as a latent
  random-walk state updated each bar. Smoother and adaptive, with the
  observation `y_t = alpha_t + beta_t * x_t + noise`. The filter is causal by
  construction (state at `t` uses data up to `t`), which makes it convenient for
  avoiding look-ahead — but it has tuning parameters (process vs observation
  variance) that themselves must not be fit on the future.

```python
# Kalman filter for a time-varying [alpha, beta]; causal, no look-ahead.
import numpy as np

def kalman_hedge(y, x, delta=1e-4, obs_var=1e-3):
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(y)
    beta = np.zeros((n, 2))                 # [alpha_t, beta_t]
    P = np.zeros((2, 2))                    # state covariance
    theta = np.zeros(2)                     # state mean [alpha, beta]
    Vw = delta / (1.0 - delta) * np.eye(2)  # state (process) covariance
    spread = np.full(n, np.nan)
    for t in range(n):
        F = np.array([1.0, x[t]])           # observation row vector
        R = P + Vw                          # predict: prior state covariance
        yhat = F @ theta                    # predicted y_t (uses past state only)
        e = y[t] - yhat                     # innovation = realized spread
        S = F @ R @ F + obs_var             # innovation variance (scalar)
        K = R @ F / S                       # Kalman gain (length-2 vector)
        theta = theta + K * e               # update state mean
        P = R - np.outer(K, F @ R)          # update state covariance
        beta[t] = theta
        spread[t] = e                       # trade the innovation / z-score it
    return beta, spread
# delta (process-to-observation variance ratio) and obs_var are tuning knobs:
# choose them on a TRAIN window, never the test window.
```

---

## 4. Mean-Reversion Model & Half-Life

Model the spread as an **Ornstein–Uhlenbeck (OU)** process (the continuous-time
mean-reverting process):

```
d(spread_t) = theta * (mu - spread_t) dt + sigma dW_t
```

`theta>0` is the speed of reversion, `mu` the long-run mean, `sigma` the
volatility. The discrete-time analogue is an **AR(1)** on the spread:

```
spread_t = c + phi * spread_{t-1} + eps_t           with phi = exp(-theta*dt)
```

Estimate `phi` by regressing the spread on its own one-period lag. The
**half-life** of mean reversion — the expected time for a deviation to decay
halfway back to `mu` — is, exactly:

```
half_life = -ln(2) / ln(phi)                         (0 < phi < 1)
```

This is only meaningful for `0 < phi < 1` (genuine, smooth mean reversion). If
`phi >= 1` the spread is a random walk or explosive — no reversion, no trade.
For `phi <= 0` the AR(1) is oscillatory/alternating, not the smooth reversion the
OU model assumes; treat such estimates as unusable rather than as a "fast" trade.

**Caution on the change-on-lag form.** A common variant regresses the *change*
on the lagged level (`d_spread_t = c + lambda * spread_{t-1} + eps_t`) where
`lambda = phi - 1`, and reports `half_life = -ln(2)/lambda`. This is an
**approximation**, not an identity: it relies on `ln(phi) ≈ phi - 1 = lambda`,
which holds only when `phi` is close to 1 (slow reversion). For `phi = 0.5` the
exact half-life is `1.0` but `-ln(2)/lambda` gives `1.39` (≈39% too long). For
fast-reverting spreads, prefer the exact `-ln(2)/ln(phi)` form.

```python
import numpy as np, statsmodels.api as sm

def half_life(spread):
    """Half-life from the change-on-lag regression. Uses the APPROXIMATION
    -ln(2)/lambda, accurate only for slow reversion (lambda near 0 / phi near 1).
    For fast reversion, fit AR(1) for phi and use -ln(2)/ln(phi) instead.
    """
    s = np.asarray(spread, float)
    s = s[~np.isnan(s)]
    ds = np.diff(s)
    lag = s[:-1]
    lam = sm.OLS(ds, sm.add_constant(lag)).fit().params[1]   # d_s = c + lam*s_{t-1}
    if lam >= 0:
        return np.inf          # no mean reversion (random walk / explosive)
    return -np.log(2) / lam

def half_life_exact(spread):
    """Exact discrete half-life via AR(1) phi = corr-style slope of s_t on s_{t-1}."""
    s = np.asarray(spread, float)
    s = s[~np.isnan(s)]
    phi = sm.OLS(s[1:], sm.add_constant(s[:-1])).fit().params[1]
    if not (0.0 < phi < 1.0):
        return np.inf          # random walk / explosive / oscillatory: no clean reversion
    return -np.log(2) / np.log(phi)
```

**Use the half-life to set horizons.** It is the natural time scale of the trade:

- Rolling estimation windows for `beta`, `mu`, `sigma` ~ a small multiple of the
  half-life (long enough to estimate, short enough to track drift).
- Expected holding period ~ on the order of the half-life. If the half-life is
  120 days but you intend to hold for 5, the edge will rarely realize before
  costs and noise dominate.
- Sanity gate: discard pairs whose half-life is implausibly long (slower than
  your capital/holding horizon) or implausibly short (likely microstructure
  noise, will be eaten by costs).

---

## 5. Trading Rules (z-score bands)

Standardize the spread to a **z-score** using a *trailing* mean and std (rolling
window ~ a few half-lives), never the full-sample mean/std:

```
z_t = (spread_t - rolling_mean_t) / rolling_std_t
```

Symmetric band rules:

- **Entry** when `|z| >= entry_band` (e.g. 2.0): the spread is stretched.
  - `z <= -entry_band` (`y` cheap relative to `x`): **long the spread** =>
    long 1 unit `y`, short `beta` units `x`.
  - `z >= +entry_band` (`y` rich): **short the spread** => short `y`, long
    `beta` units `x`.
- **Exit** when the spread reverts near the mean, `|z| <= exit_band` (e.g. 0.0–0.5).
- **Stop** when `|z| >= stop_band` (e.g. 3.0–3.5): the spread is moving *against*
  you and may have broken (structural change / regime shift). Cut it. Without a
  stop, a broken cointegration relationship produces an unbounded loss — the
  classic "the spread will surely revert" trap.

```python
import numpy as np, pandas as pd

def zscore_signals(spread, lookback, entry=2.0, exit=0.5, stop=3.5):
    s = pd.Series(np.asarray(spread, float))      # positional index 0..n-1
    # rolling stats use ONLY the trailing window (causal); .std defaults to ddof=1.
    mu = s.rolling(lookback).mean()
    sd = s.rolling(lookback).std(ddof=1)
    z = (s - mu) / sd
    pos = np.zeros(len(s)); cur = 0
    for t in range(len(s)):
        if np.isnan(z.iloc[t]) or sd.iloc[t] == 0:
            pos[t] = cur                           # carry state through NaN/zero-vol bars
            continue
        if cur == 0:
            if z.iloc[t] <= -entry:  cur =  1      # long the spread
            elif z.iloc[t] >=  entry: cur = -1     # short the spread
        else:
            if abs(z.iloc[t]) <= exit or abs(z.iloc[t]) >= stop:
                cur = 0                            # take profit OR stop out
        pos[t] = cur
    # position is the SPREAD position; lag before earning returns.
    return pd.Series(pos, index=s.index).shift(1)  # pnl_t = pos.shift(1)*spread_ret_t
```

Notes:

- `z_t` uses `rolling_mean_t` / `rolling_std_t` computed on the window ending at
  `t` (inclusive). That is causal for *signal generation*, but the position is
  still lagged (`.shift(1)`) before it earns the next bar's return, so the bar-`t`
  spread value is never used to earn the bar-`t` return.
- The position returned is the position *in the spread*; the per-leg positions are
  `+pos` in `y` and `-pos*beta` in `x` (use the bar-appropriate, lagged `beta`,
  not a full-sample `beta`).
- Define "spread return" consistently with how the spread is built. If you trade
  fixed unit weights, the spread PnL per unit is `Δspread_t = spread_t -
  spread_{t-1}` (a *dollar* change), not a simple percentage return — do not feed
  a level-difference into return-based Sharpe machinery without converting to a
  per-unit-capital return first. Mixing level-changes and simple returns silently
  corrupts Sharpe.
- Choosing `lookback`, `entry`, `exit`, `stop` is a multi-parameter search —
  validate with purged + embargoed walk-forward (`templates/validation.py`) and
  account for the search in your significance test (`references/stats-risk.md`).
- Wider entry bands => fewer, higher-conviction trades and lower turnover;
  tighter exits => faster recycling but more cost drag. The right point depends on
  the half-life and on costs (section 7).

---

## 6. Portfolio / Basket Stat-Arb

Beyond a single pair, the same idea generalizes to **residual reversion** across a
universe.

**Factor/PCA residual reversion.** Regress each asset's returns on a set of common
factors (market, sector, or statistical factors from PCA of the return
covariance), and trade the **residual** — the part not explained by common
factors. Residuals are, by construction, approximately market/sector-neutral and
tend to mean-revert (this is the engine behind classic statistical-arbitrage work
such as Avellaneda–Lee, 2010).

```
r_i = sum_k beta_ik * F_k + residual_i
signal_i = -zscore(cumulative residual_i)   # short rich residuals, long cheap
```

- **PCA factors:** eigen-decompose the (often shrunk) return covariance or
  correlation matrix; the top components are the common factors, the rest is
  idiosyncratic. Trade the reconstructed residual return. Re-estimate factors on
  a rolling window (the factor structure drifts — see
  `references/factor-research.md` and `templates/factor_research.py`). Use a
  shrinkage estimator (e.g. the constant-correlation Ledoit-Wolf in
  `templates/validation.py`) before inverting/eigen-decomposing on a large
  universe.
- **Sector/explicit factors:** use observable factors (sector ETFs, size/value)
  when you want interpretable neutrality constraints.

**Neutrality construction.** Build the portfolio so net factor exposure is ~0:
gross long ≈ gross short (dollar neutral), and `sum_i w_i * beta_ik ≈ 0` for each
factor `k` (beta/sector neutral). This is what makes the book a *relative-value*
bet rather than a directional one. Enforce it in the optimizer/weights, not by
hoping it nets out.

**Why baskets over single pairs:** diversification across many weakly-reverting
residuals smooths PnL and reduces single-name break risk, at the cost of more
estimation (covariance/factor matrices) and more turnover.

---

## 7. Risks & Pitfalls

The recurring theme: a relationship that is real *in sample* is not necessarily
real *out of sample*, and the edges are thin enough that costs and crowding
matter as much as the signal.

- **In-sample-only cointegration.** A pair passes EG/Johansen on the backtest
  window and falls apart live. This is partly low test power, partly multiple
  testing, partly genuine instability. *Always* validate cointegration on a
  held-out, walk-forward basis, not just on the full history.
- **Regime change / structural breaks.** Mergers, index changes, business-model
  shifts, central-bank regime changes. Re-test cointegration on a rolling basis
  and enforce a hard stop (section 5) so a broken spread can't run unbounded.
- **Crowding.** Well-known pairs/residual-reversion signals are traded by many
  desks; the edge decays and, worse, crowded unwinds (e.g. the Aug-2007 quant
  quake) cause correlated drawdowns precisely when you expect neutrality. Monitor
  decay and don't assume historical edge persists.
- **Look-ahead from the hedge ratio.** Estimating `beta` (or `mu`/`sigma` for the
  z-score, or the cointegration relationship itself) on the **full sample** and
  then "backtesting" leaks the future into every bar. This single mistake
  manufactures gorgeous, fake equity curves. Use rolling/expanding/Kalman
  estimation only.
- **Thin edges vs costs & borrow.** Mean-reversion profits per round-trip are
  small; spread, commissions, market impact, and **short-borrow / financing
  fees** on the short leg routinely exceed the edge — especially on hard-to-borrow
  names. Model costs and borrow explicitly (`templates/costs.py` —
  `borrow_cost`, `funding_cost`, `slippage_total`; `references/transaction-costs.md`)
  before believing any pairs Sharpe.
- **Multiple testing across candidate pairs.** Screening `N` pairs and keeping
  `p<0.05` finds false positives by construction. Apply a multiple-testing
  correction (Bonferroni / BH FDR — see `benjamini_hochberg` in
  `references/stats-risk.md`) or a probabilistic/deflated-Sharpe adjustment, and
  reserve a truly out-of-sample window.
- **Non-stationarity assumed away.** The whole framework assumes the legs are
  `I(1)` and the spread `I(0)`. If those don't hold (e.g. both legs are already
  stationary, or the spread is itself `I(1)`), the half-life and z-score are
  meaningless. Verify integration orders first.

### Pitfalls table (detect / fix)

| Pitfall | Detect | Fix |
|---|---|---|
| Selecting pairs on return correlation | Pairs chosen by `corr(r_x,r_y)`; spread itself trends | Require a cointegration test on price *levels*; correlation only as a candidate prior |
| Full-sample hedge ratio (look-ahead) | `beta`/`mu`/`sigma` fit once on all data, then "backtested" | Rolling/expanding/Kalman estimation; every bar uses only prior data |
| Plain ADF on OLS residual | `adfuller` run on `y~x` residual with standard crit values | Use `coint()` (EG critical values) or Johansen; account for estimated `beta` |
| OLS direction asymmetry | `beta_{y~x} != 1/beta_{x~y}` (they satisfy `beta_{y~x}*beta_{x~y}=R^2`); results flip with ordering | Report both orderings; prefer TLS/orthogonal regression or Johansen (symmetric) |
| `params[1]` on pandas results | `KeyError: 1` or wrong value when x/y are Series | Coerce to numpy, or use `.params.iloc[1]`; avoid label indexing |
| Half-life `-ln2/lambda` misused | Fast-reverting spread; `-ln2/lambda` far from `-ln2/ln(phi)` | Use exact `-ln(2)/ln(phi)` when `phi` not near 1; gate on `0<phi<1` |
| Half-life on non-reverting spread | `half_life` computed when `phi>=1` (random walk) or `phi<=0` (oscillatory) | Gate on `0<phi<1`; reject non-reverting/oscillatory spreads |
| TLS on mismatched scales | PCA slope unstable; legs have very different vol | Standardize legs (or use known error-variance ratio) before TLS; rescale slope back |
| No stop on the spread | "It must revert" — position held as `|z|` grows | Hard stop at `|z|>=stop_band`; re-test cointegration on a rolling window |
| Level-change vs return mixup | Spread PnL fed as `Δspread` into return-based Sharpe | Convert spread PnL to a per-unit-capital return before Sharpe machinery |
| Costs/borrow ignored | Backtest gross; short leg assumed free to borrow | Subtract spread+impact+commission+borrow per round-trip; check hard-to-borrow names |
| Pair-mining false positives | Hundreds of pairs screened, keep `p<0.05` | Bonferroni/FDR or deflated/probabilistic Sharpe; out-of-sample holdout; cap candidates |
| Non-stationary spread | Spread itself wanders (`I(1)`), z-score never reverts | Confirm legs are `I(1)` and spread `I(0)` before trading; otherwise discard |
| Regime/structural break | Live PnL diverges from backtest after a corporate event | Monitor rolling cointegration; flatten on events (M&A, index changes) |
| In-sample-only cointegration | Great on full history, fails live | Walk-forward validation (purge+embargo, `templates/validation.py`) on the test |
| Position not lagged | `pnl_t = pos_t * ret_t` (uses same-bar signal) | `pnl_t = pos.shift(1) * ret_t`; signal at close of `t` earns return of `t+1` |

---

### Pointers

- `templates/validation.py` — purged + embargoed walk-forward CV (`PurgedKFold`,
  `CombinatorialPurgedKFold`) for spread parameters and out-of-sample
  cointegration checks; constant-correlation covariance shrinkage for baskets.
- `templates/costs.py` — `borrow_cost`, `funding_cost`, `slippage_total`,
  square-root/linear impact, and break-even helpers for thin mean-reversion edges.
- `templates/metrics.py` — Sharpe / drawdown / annualization with stated
  conventions (matches the Sharpe definition above).
- `templates/factor_research.py` — PCA/factor construction underlying basket
  residual reversion.
- `references/stats-risk.md` — multiple-testing corrections (`benjamini_hochberg`),
  probabilistic/deflated Sharpe for pair-mining; autocorrelation caveats on Sharpe.
- `references/transaction-costs.md` — modeling spread, impact, and borrow that
  eat thin mean-reversion edges.
- `references/factor-research.md` — factor construction details for basket
  residual reversion.
- `templates/pairs_trading.py` — companion end-to-end pair workflow (shipped, self-tested)
  (selection, cointegration test, rolling/Kalman hedge ratio, half-life, z-score
  signal, costed backtest). The snippets above are the spec; do not assume the
  file exists yet.