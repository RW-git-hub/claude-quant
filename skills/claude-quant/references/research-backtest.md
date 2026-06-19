# Strategy Research & Backtesting

Reference for the claude-quant skill. How to go from idea to a backtest you can trust, and the traps that make a dead strategy look alive.

---

## 1. The research loop

```
hypothesis -> data -> signal -> backtest -> analysis -> decision
                ^                                          |
                +------------------ iterate ---------------+
```

The single most important discipline: **write down the hypothesis, the test, and the kill criteria BEFORE you look at out-of-sample results.** This is pre-registration. Without it you will p-hack — running 50 variants and keeping the one that looked good is selection bias, and its in-sample Sharpe is meaningless.

Pre-registration template (commit this to the repo before running OOS):

```yaml
# experiment.yaml
hypothesis: "12-1 month cross-sectional momentum in liquid US equities earns a
             positive long-short premium net of 10bps round-trip costs."
universe:    "S&P 500 members as of each rebalance date (point-in-time)"
signal:      "rank by trailing 12m return skipping most recent month"
sample:      {in_sample: "2005-2015", out_of_sample: "2016-2023"}
costs:       {commission_bps: 1, spread_bps: 4, slippage_bps: 5}
benchmark:   "equal-weight universe"
success:     "OOS net Sharpe > 0.5 AND OOS maxDD > -25% AND beats benchmark"
kill:        "OOS net Sharpe < 0.3, OR turnover-adjusted alpha t-stat < 2"
n_trials:    "count every variant tested; deflate Sharpe accordingly"
```

**Why it matters.** With enough trials, some random strategy clears any fixed bar. Track `n_trials` and apply the Deflated Sharpe Ratio (Bailey & López de Prado) or at minimum a Bonferroni-style haircut. A backtest is a hypothesis test with a multiple-comparisons problem.

**Don't hand-wire the sweep — that's where the trial count gets under-counted and the OOS gets peeked.** Run the parameter grid through `templates/validation.py`'s `walk_forward_evaluate(strategy_fn, param_grid, data, train=…, test=…, label_horizon=…, embargo_pct=…)`: it marches a purged+embargoed expanding/rolling window over the data, calls your `strategy_fn(train_idx, test_idx, params) -> oos_return_series` for *every* grid point, and returns the (config × time) OOS performance matrix plus an honest `n_trials = len(param_grid)` (survivors **and** failures, not just the variants you liked). The harness owns the purge/embargo and the trial count so neither can be forgotten. Then `summarize_search(result)` returns the one-line verdict — `{best_config, oos_sharpe, dsr, pbo, n_trials, n_eff, performance_degradation}` — wiring the matrix into the Deflated Sharpe (`metrics.deflated_sharpe_ratio`) and the PBO/CSCV overfitting probability (`overfitting.pbo_cscv`). This is the canonical sweep entry point; see §8.

**Effective vs raw trial count.** Grid points are not independent — neighbouring parameters produce highly correlated return streams — so the relevant multiple-testing budget is the *effective* number of trials, not the grid size. `summarize_search` deflates by `n_eff = effective_n_trials(perf_matrix)` (the participation ratio `(Σλ)²/Σλ²` of the config-correlation eigenvalues: `= N` when configs are independent, `→ 1` as they collapse onto one bet). Report both `n_trials` (the honest raw count you ran) and `n_eff` (what you deflated by).

**Decision rule.** A strategy graduates only on *out-of-sample, net-of-cost* numbers that you committed to in advance. In-sample fit and gross returns are for debugging, never for go/no-go. Concretely: `dsr > 0.95` (the deflated Sharpe survives the effective trial count) **and** `pbo < 0.5` (selecting the in-sample best is not anti-predictive out-of-sample) are necessary gates, not nice-to-haves.

---

## 2. Vectorized vs event-driven backtests

| | Vectorized | Event-driven |
|---|---|---|
| Model | array math over the whole series at once | bar-by-bar / tick-by-tick event loop |
| Speed | very fast (great for parameter sweeps) | slow (Python loops) but parallelizable |
| Realism | weak: hard to model partial fills, queue position, path-dependent sizing | strong: order types, fills, margin, intrabar logic |
| Leakage risk | **high** — easy to use future data accidentally | lower — causality enforced by the loop |
| Use when | factor/cross-sectional research, sweeps, prototyping | execution-sensitive, intraday, complex order logic, capital constraints |
| Libraries | vectorbt, polars/pandas by hand | backtrader, zipline-reloaded, bt, nautilus |

**Validity rule:** a vectorized backtest is valid only if every position is a *pure function of past data* and fills are simple (full fill at a known price). The moment sizing depends on realized P&L, or you cap by available liquidity, or you use limit orders, you need an event loop (or a much more careful vectorized formulation).

### The classic vectorized leakage trap

Decision is made at the close of bar `t`; you can only earn the return of bar `t+1` onward. Positions must be **shifted** relative to the returns they earn.

```python
import numpy as np, pandas as pd

ret = px["close"].pct_change()                 # simple returns
signal = (px["close"] > px["close"].rolling(50).mean()).astype(float)

# BUG: same-bar execution. signal at close t multiplies return OF bar t,
# which already happened before you knew the signal. Look-ahead.
pnl_bug = signal * ret

# CORRECT: decide at close t, hold over t+1. Shift the position by 1.
pos = signal.shift(1)
pnl = pos * ret
```

**Detect same-bar leakage:** if removing `.shift(1)` barely changes (or *improves*) results, you were trading on the bar you used to decide. A legit strategy gets *worse* gross when you lag it correctly. Also: a backtest with Sharpe > 4 on daily data is almost always leakage, not alpha.

Other vectorized leak sources to grep for:
- `.rolling(...).mean()` / z-scores computed with `center=True` (uses future).
- Full-sample `StandardScaler.fit` / `df.mean()` used to normalize each row (uses the whole series mean, including future).
- Resampling that labels a bar with its `close` then trades at that bar's `open`/`high`.
- Survivorship: using *today's* index constituents to backtest the past.
- Reindexing/`ffill` of fundamentals to the *report date* instead of the *publication date* (point-in-time data needed).

---

## 3. Transaction costs & frictions

**Ignoring costs is the #1 way to fake alpha.** High-turnover signals (mean-reversion, short-horizon momentum) often have huge gross Sharpe that evaporates entirely net of cost. Always report net first.

Cost components (model the ones that bind for your asset/frequency). Be explicit about whether your inputs are quoted full-spread or half-spread:

```python
def round_trip_cost_bps(spread_bps, commission_bps, slippage_bps):
    # One full entry+exit. spread_bps is the FULL quoted bid-ask spread.
    # Crossing the spread once costs half the spread (you trade at the touch),
    # so a round trip crosses it twice = one full spread.
    # Per-side cost  = spread_bps/2 + commission_bps + slippage_bps
    # Round-trip cost = 2 * per-side = spread_bps + 2*commission_bps + 2*slippage_bps
    return spread_bps + 2 * commission_bps + 2 * slippage_bps


def per_side_cost_bps(spread_bps, commission_bps, slippage_bps):
    # Cost charged per unit of notional traded (one leg).
    return spread_bps / 2 + commission_bps + slippage_bps
```

| Friction | Equities | Futures | Crypto spot/perp | FX/options |
|---|---|---|---|---|
| Commission | per-share/notional | per-contract | bps taker fee, maker rebate | tight FX; per-contract options |
| Spread | tight on large caps | tight on front month | wide on alts, thin books | FX tight; options spreads wide |
| Slippage/impact | scales with ADV % | depth-dependent | severe on alts | gap risk |
| Borrow/financing | short borrow fee, can be huge for hard-to-borrow | embedded in roll | margin interest | swap points (carry) |
| Funding | n/a | n/a | **perp funding** every 8h | n/a |

### Slippage models (increasing realism)

```python
# 1. Fixed bps — fine for liquid, small size
slip = notional_traded * (slippage_bps / 1e4)

# 2. Participation / volume-aware — cost grows when you trade a big % of ADV
adv_frac = abs(shares_traded) / adv_shares
slip = notional_traded * (base_bps + k * adv_frac) / 1e4

# 3. Square-root market impact (Almgren-style); the empirical standard.
# impact (as a fraction of price) ~ c * sigma * sqrt(Q / ADV)
#   Q = order size in shares, sigma = daily return vol (fractional, not %),
#   c = O(1) dimensionless constant. Multiply by 1e4 to express in bps.
impact_bps = c * daily_vol * np.sqrt(abs(shares_traded) / adv_shares) * 1e4
```

`c` is typically O(1) (often ~0.5–1). The sqrt law matters because impact *per share* grows with the square root of size, so total impact cost is *superlinear in size* (cost ∝ Q^{3/2}) while marginal impact is concave — that concavity in marginal cost is what creates **capacity limits**.

### Apply costs to turnover, not just trades

```python
# pos is the target weight series, NOT yet lagged here.
turnover = pos.diff().abs()                       # fraction of book traded per bar
per_side = per_side_cost_bps(spread_bps=4, commission_bps=1, slippage_bps=5) / 1e4
cost = turnover * per_side                        # charge each traded leg once
# Lag the position vs the returns it earns; charge the cost on the bar it trades.
pnl_net = pos.shift(1) * ret - cost
```

Note: `pos.diff().abs()` already counts both legs of a round trip (one unit on entry, one
on exit across two rebalances), so multiplying it by the **per-side** cost reproduces the
full round-trip cost over a complete entry+exit. Do not also divide by 2.

### Crypto perp funding

```python
# funding paid by longs to shorts (or reverse) every funding interval
funding_pnl = -position_notional * funding_rate   # sign: long pays when rate>0
```
Funding can dominate carry strategies. A "free" basis trade is often just harvesting funding — model it explicitly.

### Cost sensitivity

Never report one number. Sweep costs and show where the strategy dies:

```python
for c_bps in [0, 5, 10, 20, 40]:
    sharpe = backtest(cost_bps=c_bps).sharpe()
    print(c_bps, round(sharpe, 2))
```
If realistic costs (your venue's actual fees + measured spread) push Sharpe below your kill threshold, the strategy is not real. A robust edge degrades gracefully; a fake one falls off a cliff at the first basis point.

---

## 4. Realistic fills

- **Next-bar open vs close.** Decide at close `t`; the *earliest realistic fill* is open of `t+1`, not close of `t`. Filling at close `t` is look-ahead (you can't trade at a price you used to decide). For daily strategies, executing at next open (and earning open→open returns) is the honest default; using close→close with `shift(1)` is acceptable if you also assume you can transact at the close (MOC orders) — be explicit.
- **Limit vs market.** Market orders fill but pay the spread/impact. Limit orders may save the spread but suffer **adverse selection** (they fill when the market moves against you, not when it moves your way). A vectorized backtest that assumes limit fills at the touch with 100% fill rate is fantasy. Model a fill probability or require the bar's range to cross the limit.
- **Partial fills / ADV caps.** Cap per-bar traded size at a fraction of ADV (e.g. 5–10%). If desired trade exceeds the cap, fill partially and carry the remainder.

```python
max_trade = participation_cap * adv_shares       # e.g. 0.05 * ADV
fill = np.clip(desired_trade, -max_trade, max_trade)
```

- **Capacity.** As AUM grows, impact (sqrt law) eats the edge. Estimate capacity as the AUM where marginal impact cost equals marginal gross alpha. Report it — a 3-Sharpe strategy that holds $2M is a hobby, not a fund.

---

## 5. Position sizing

```python
# Fixed fraction: constant exposure
pos = signal_direction * target_gross

# Volatility targeting: scale to a target annualized vol -> stabler Sharpe, controls DD.
# Lag the scaler: today's position must be sized from vol known as of yesterday's close.
realized_vol = (ret.rolling(20).std(ddof=1) * np.sqrt(252)).shift(1)
pos = signal_direction * (target_ann_vol / realized_vol)
pos = pos.clip(-max_leverage, max_leverage)
```
Vol targeting is the single highest-value sizing improvement for most strategies: it equalizes risk contribution across regimes and usually raises Sharpe and tames drawdowns. **Lag the vol estimate** — using today's (close-of-bar) realized vol to size today's position is look-ahead. Note `ret.rolling(20).std()` already only sees data through bar `t`; the extra `.shift(1)` enforces that the position for bar `t` is decided at the close of `t-1`.

**Risk budgeting / risk parity:** allocate so each asset (or sub-strategy) contributes equal risk: `w_i ∝ 1/σ_i` (naive) or solve for equal marginal risk contribution using the covariance matrix. Better diversification than equal-dollar weighting.

### Kelly — and why full Kelly is dangerous

For a single bet, Kelly fraction `f* = edge / variance`; for a continuous return stream `f* ≈ μ / σ²` (μ, σ per period). Full Kelly maximizes *expected log growth* but:
- It assumes you know μ and σ exactly. You don't — estimation error means your "Kelly" is biased high, and overbetting destroys capital geometrically.
- Full-Kelly drawdowns routinely exceed 50%; few survive them.
- The growth curve is extremely flat near `f*`: **half-Kelly gives ~75% of the growth at ~half the volatility.**

Practical rule: size at **quarter- to half-Kelly**, and cap leverage independently.

---

## 6. Cross-sectional / multi-asset backtests

Construction for a market-neutral long-short factor:

```python
# scores: DataFrame [dates x assets], factor value known at close t.
# fwd_ret: DataFrame [dates x assets] of returns earned over (t, t+1] — i.e.
#          fwd_ret.loc[t] = ret.loc[t+1]. Using fwd_ret with an un-lagged weight
#          is equivalent to lagging the weight; do exactly ONE of the two, never both.
z = scores.sub(scores.mean(axis=1), axis=0).div(scores.std(axis=1, ddof=1), axis=0)

# neutralize: demean within sector so you bet the factor, not the sector tilt.
# pandas 2.x removed groupby(axis=1); transpose, group rows, transpose back.
sector_mean = z.T.groupby(sector).transform("mean").T
z = z.sub(sector_mean)                                    # sector-neutral

# dollar-neutral long-short weights, gross = 1
w = z.div(z.abs().sum(axis=1), axis=0)                    # sum|w| = 1, sum w ≈ 0

# Because fwd_ret is already the NEXT-bar return, do NOT also shift w here.
port_ret = (w * fwd_ret).sum(axis=1)
```

- **Rebalancing schedule.** Daily rebal maximizes signal freshness but maximizes turnover/cost. Match rebalance frequency to signal decay (IC half-life). Often weekly/monthly nets more than daily.
- **Turnover control.** Add a no-trade band / cost-aware optimizer so you only trade when the expected alpha from rebalancing exceeds the cost. Cheap heuristic: only move toward target if `|w_target − w_current| > threshold`.
- **Neutralization.** Beta-, sector-, size-neutralize to isolate the factor. Otherwise your "momentum" P&L may just be a tech-sector or market-beta bet.
- **Long-short construction.** Dollar-neutral (Σw=0) removes market direction; beta-neutral removes market risk more precisely. Cap single-name and sector weights to avoid concentration.
- **IC as the health check.** IC = cross-sectional rank corr between the factor at `t` and **forward** returns over `t..t+h`. A stable positive IC (even ~0.03–0.05) with low decay is the real evidence; the equity curve is downstream of it.

```python
from scipy.stats import spearmanr
ic = pd.Series(
    [spearmanr(scores.loc[d], fwd_ret.loc[d], nan_policy="omit")[0]
     for d in scores.index],
    index=scores.index,
)
# spearmanr returns NaN when a cross-section is degenerate; ignore those dates.
ir = ic.mean() / ic.std(ddof=1)                  # information ratio of the signal
```

---

## 7. Libraries — when to reach for each

| Library | Paradigm | Reach for it when | Watch out |
|---|---|---|---|
| **hand-rolled** (pandas/polars/numpy) | vectorized | full control, transparency, factor research, you must trust every line | you own every leakage bug |
| **vectorbt** | vectorized, numba-backed | huge parameter sweeps, fast indicator/portfolio grids, intraday vectorized | API is dense; still vectorized-realism limits; verify lag semantics |
| **backtrader** | event-driven | realistic order types, multi-asset, live-ish logic, education | slow, largely unmaintained, awkward for cross-sectional |
| **zipline-reloaded** | event-driven | pipeline factor API, point-in-time discipline, US equities | data ingestion is heavy; equities-centric |
| **bt** | tree-of-algos | weight-based allocation, rebalancing logic, multi-asset portfolios | less tick/order realism |
| **nautilus_trader** | event-driven, Rust core | HFT/intraday, true order book, live+backtest parity | steep learning curve |

Guidance: **prototype and sweep in vectorbt or by hand; validate the winner in an event-driven engine** before committing capital. If the two disagree materially, the vectorized version was hiding a fill/leakage assumption — trust the event-driven one.

For a perf layer on hand-rolled loops: numba `@njit` for the bar loop, polars for the data wrangling, or a Rust/Cython core for tick-level. Don't optimize until a clean version proves the edge.

---

## 8. Reproducibility

A backtest you can't reproduce is an anecdote.

- **Fixed seeds** everywhere stochastic: `np.random.seed`, `random.seed`, framework seeds, and seed any CV split / model.
- **Versioned, immutable data.** Snapshot raw inputs with a content hash; store point-in-time data so re-runs use what was knowable then. DVC / parquet snapshots / a hash in the log.
- **Config-driven runs.** All params in a YAML/TOML, no magic numbers in code. Log the resolved config with every run.
- **Log the full experiment:** git commit hash, config, data hash, library versions, `n_trials`, timestamps, and the metrics. MLflow/W&B or a plain JSON-per-run is fine.

```python
import json, hashlib, subprocess, time
def log_run(config, metrics, data_df):
    rec = {
        "ts": time.time(),
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "config": config,
        "data_sha": hashlib.sha256(
            pd.util.hash_pandas_object(data_df, index=True).values.tobytes()
        ).hexdigest()[:16],
        "metrics": metrics,
    }
    with open(f"runs/{int(rec['ts'])}.json", "w") as f:
        json.dump(rec, f, indent=2, default=str)
```

For model-based signals, validate with **purged, embargoed time-series CV** (López de Prado): remove training samples whose label windows overlap the test window (purge), and drop a further block of training samples immediately *after* the test window (embargo) to defend against serial correlation leaking across the boundary. Plain `KFold`/shuffle leaks across time. `sklearn.model_selection.TimeSeriesSplit` enforces train-before-test ordering but does **not** purge or embargo — you must add those yourself when labels span multiple bars. `templates/validation.py` ships `PurgedKFold` and `CombinatorialPurgedKFold` (CPCV) that do both, with the same `.split` API as sklearn but no sklearn dependency.

### The canonical sweep harness: `walk_forward_evaluate` → `summarize_search`

The splitters give you leak-free folds, the Deflated Sharpe (`metrics.py`) gives you a haircut for trials, and PBO/CSCV (`overfitting.py`) tells you whether the *ranking* generalizes — but those are primitives. `templates/validation.py` provides the harness that runs the actual sweep through them and hands you one verdict, operationalizing Iron Laws 4 (OOS sacred) and 5 (deflated, honest stats) end-to-end:

```python
from validation import walk_forward_evaluate, summarize_search

# strategy_fn receives integer positional indices into `data` for the purged
# train rows and the test rows, plus this grid point's params. It returns the
# per-period OOS returns realized on the TEST rows (already lagged/costed by you).
def strategy_fn(train_idx, test_idx, params):
    model = fit(X[train_idx], y[train_idx], **params)     # train only on past
    signal = model.predict(X[test_idx])
    return (np.sign(signal) * fwd_ret[test_idx]            # fwd_ret is next-bar => no look-ahead
            - turnover(signal) * cost_per_side)            # cost BEFORE Sharpe (Iron Law 3)

grid = [{"lookback": L, "thresh": t} for L in (20, 60, 120) for t in (0.5, 1.0, 1.5)]
res  = walk_forward_evaluate(strategy_fn, grid, data,
                             train=750, test=125,          # ~3y train, ~6m test
                             anchored=False,               # rolling window; True = expanding
                             label_horizon=5, embargo_pct=0.01)
verdict = summarize_search(res)
# -> {best_config, oos_sharpe, dsr, pbo, n_trials=9, n_eff, performance_degradation}
```

What the harness guarantees so you can't get it wrong:
- **No look-ahead across the fold.** Every `train_idx` lies strictly before its `test_idx`, and the last `label_horizon` train bars before the test block are purged (plus an embargo buffer). A test index appearing in its own train fold raises immediately — the purge/embargo assertion runs on every call.
- **Honest trial counting.** `n_trials == len(param_grid)`: every config you ran is counted, survivors and failures alike. There is no path that silently drops the variants that "didn't work."
- **No OOS peeking.** `strategy_fn` only ever sees `train_idx` for fitting; the test rows are passed solely to realize returns. You cannot accidentally normalize, select features, or tune on the test window because you never receive it as training data.
- **One deflated verdict.** `summarize_search` ranks configs by per-period OOS Sharpe, deflates the winner by `n_eff` (correlation-aware effective trials) via the Deflated Sharpe, and runs the full OOS matrix through CSCV for the PBO and the IS→OOS degradation diagnostics. Read `dsr` and `pbo` together: DSR deflates the *headline number* for the trial budget; PBO asks whether the *selection rule* generalizes. They fail in different ways — a strategy can have a respectable DSR yet a PBO > 0.5 (the winner is a different config each split), or a clean PBO yet a DSR < 0.95 (real but too small to clear the trial count). (`performance_degradation` is an optional enrichment populated when the `overfitting.py` sibling is importable — the production layout; in a bare standalone `validation.py` it is `None` while `dsr`/`pbo` still compute from in-file fallbacks.)

Use CPCV instead of the single walk-forward path when you want the full *distribution* of OOS outcomes (many backtest paths) rather than one chronological path; collect each path's per-period returns into the columns of a performance matrix and pass it straight to `summarize_search` / `overfitting.pbo_cscv`.

---

## 9. Worked example: cross-sectional momentum, done correctly

Long-short, dollar-neutral, lagged, vol-targeted, costed. Comments flag exactly where leakage usually sneaks in. This snippet is self-contained and runnable given a price panel `px`.

```python
import numpy as np, pandas as pd

# px: DataFrame [dates x tickers] of point-in-time adjusted closes.
# Universe must be point-in-time (no survivorship) — LEAK #1 if you use today's list.

ret = px.pct_change()                                  # simple returns; aggregate via prod(1+r)-1

# --- SIGNAL: 12-1 momentum (return from t-252 to t-21, skipping the last ~month) ---
mom = px.shift(21) / px.shift(252) - 1                 # uses only past prices

# --- CROSS-SECTIONAL RANK -> dollar-neutral weights ---
z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1, ddof=1), axis=0)
w_raw = z.div(z.abs().sum(axis=1), axis=0)             # gross 1, ~market neutral

# --- LAG: decide at close t, hold over t+1. LEAK #2 if you forget this. ---
w = w_raw.shift(1)

# --- VOL TARGET the whole book; lag the scaler. LEAK #3 if you use same-bar vol. ---
# w is already lagged, so (w * ret) earns next-bar returns with no look-ahead.
gross_ret = (w * ret).sum(axis=1)
realized = gross_ret.rolling(20).std(ddof=1) * np.sqrt(252)
scale = (0.10 / realized).shift(1).clip(upper=3)       # 10% ann vol target, scaler lagged
w = w.mul(scale, axis=0)

# --- COSTS on turnover (per-side bps). LEAK #4: omitting costs = fake alpha. ---
turnover = w.diff().abs().sum(axis=1)                  # fraction of book traded (both legs counted)
# Per-side cost = half-spread + commission + slippage = 4/2 + 1 + 5 = 8 bps.
cost_per_side = (4 / 2 + 1 + 5) / 1e4
strat_ret = (w * ret).sum(axis=1) - turnover * cost_per_side

# --- METRICS (conventions: simple returns, 252/yr, ddof=1) ---
# Drop the warmup NaNs from shift/rolling before annualizing, or eq.iloc[-1] is NaN.
strat_ret = strat_ret.dropna()
ppy = 252
eq = (1 + strat_ret).cumprod()
n = len(strat_ret)
ann_ret = eq.iloc[-1] ** (ppy / n) - 1                # geometric (CAGR)
ann_vol = strat_ret.std(ddof=1) * np.sqrt(ppy)
sharpe  = strat_ret.mean() / strat_ret.std(ddof=1) * np.sqrt(ppy)   # rf ~ 0; excess = strat_ret
dd      = eq / eq.cummax() - 1
maxdd   = dd.min()
calmar  = ann_ret / abs(maxdd)

print(f"AnnRet {ann_ret:.2%}  AnnVol {ann_vol:.2%}  Sharpe {sharpe:.2f} "
      f"MaxDD {maxdd:.2%}  Calmar {calmar:.2f}  AnnTurnover {turnover.mean()*ppy:.1f}x")
```

**Where leakage sneaks in (checklist):**
1. **Universe** built from current constituents → survivorship bias. Use point-in-time membership.
2. **No `shift(1)` on weights** → same-bar execution, look-ahead. Removing the lag should make gross results *worse*; if it doesn't, you've found the leak.
3. **Vol scaler not lagged** → uses today's realized vol to size today.
4. **Costs omitted** → momentum's turnover is moderate; mean-reversion variants will look spectacular gross and die net. Always sweep cost.
5. **`mom` using `.shift(0)`** or centered windows → future prices in the signal. Confirm every term in the signal references only `px.shift(k>=1)`.

**Sanity gates before believing it:** Sharpe in a plausible range (a daily L/S equity factor at net Sharpe > 2.5 is suspicious); positive, stable IC; graceful degradation under the cost sweep; results survive purged/embargoed CV; and the numbers reproduce from the logged config + data hash. For a multi-config sweep, run it through `walk_forward_evaluate` / `summarize_search` (§1, §8) and graduate only on `dsr > 0.95` with `pbo < 0.5`.
