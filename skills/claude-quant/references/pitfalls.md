# Quant Pitfalls Catalog

Fast-reference list of traps that wreck backtests and live trading. Each entry: **why it kills you**, **detect**, **fix**. Scan the summary table first; jump to the section you're working in. Pre-flight checklist at the bottom.

> Mental model: most blown backtests die from one bug — **information from the future leaking into a decision in the past**. Look-ahead, leakage, survivorship, and CV leakage are all the same disease. The rest is cost realism and statistical self-deception.

---

## Summary table

| # | Trap | Section | One-line tell |
|---|------|---------|---------------|
| 1 | Look-ahead bias | Data/Backtest | Equity curve too smooth; Sharpe > 3 daily |
| 2 | Survivorship bias | Data | Universe = "today's" tickers |
| 3 | Delisting bias | Data | No -100% / partial-recovery rows for dead names |
| 4 | Corporate-action errors | Data | Price gaps on split/dividend dates |
| 5 | Continuous-contract level errors | Data | Returns computed on stitched futures price level |
| 6 | Timezone / calendar bugs | Data | Bars off by one; signal "sees" close before it prints |
| 7 | NaN / fillna leakage | Data | `ffill` of future, `fillna(mean)` of full series |
| 8 | Log-vs-simple / compounding errors | Data | Summing simple returns; mixing the two |
| 9 | Same-bar / next-bar execution | Backtest | `pnl = position * return` (no shift) |
| 10 | Unrealistic fills | Backtest | Filled at exact close/VWAP, full size, no queue |
| 11 | Ignoring transaction costs & slippage | Backtest | Zero-cost PnL; high turnover strategy "works" |
| 12 | Transaction-cost underestimation | Backtest | Flat bps on illiquid/large/crypto fills |
| 13 | Capacity & liquidity blindness | Risk | Trades > % of ADV; no participation cap |
| 14 | Currency & financing/funding omissions | Backtest | FX/futures/crypto carry ignored |
| 15 | Data-snooping / multiple testing | Stats | Many variants tried, best reported |
| 16 | In-sample optimization / overfitting | Stats | Many params, one period, no OOS |
| 17 | p-hacking | Stats | Stop testing when p < 0.05 |
| 18 | Leakage in normalization/scaling | Stats | `fit` scaler/PCA on full sample |
| 19 | CV leakage (no purge/embargo) | Stats | `KFold(shuffle=True)` on time series |
| 20 | Regime overfitting | Stats | Tuned to one regime; fails OOS regime |
| 21 | Sample-period cherry-picking | Stats | Start/end chosen to flatter |
| 22 | Benchmark gaming | Risk | Beat cash, not the right benchmark |
| 23 | Vol-targeting / risk-scaling look-ahead | Backtest | Size today uses today's (or full-sample) vol |
| 24 | Signal / alpha decay | Production | Live Sharpe << backtest, trending down |
| 25 | Factor crowding | Production | Correlated to known factors; drawdowns co-move |
| 26 | Intrabar stop/target fill (path-dependent) | Backtest | Stop *and* target both "hit" the same bar; filled at the trigger level, not the gap |

---

## Data

### 1. Look-ahead bias (many flavors)
**Why it kills you:** You use information not yet available at decision time. Produces gorgeous, fake backtests; reverts to zero or negative live.
**Flavors to hunt:**
- **Execution look-ahead:** deciding bar t position/fill using bar t data you couldn't have observed until the bar closed (see #9).
- **Point-in-time look-ahead:** using restated fundamentals / index membership / analyst estimates as they look *today*, not as published then.
- **Full-sample statistics:** any `mean`, `std`, `quantile`, `min/max`, scaler, PCA, or rank computed over the *entire* series then applied historically (see #18).
- **Resampling/aggregation look-ahead:** a daily bar's close/high/low isn't known until session end — don't act on it intraday.
- **Timestamp look-ahead:** data labeled at the *start* of its window but only knowable at the *end* (e.g. economic releases, bar OHLC).
- **Bar-close vs bar-open:** computing a signal from bar t's close and assuming you also traded at bar t's close.

**Detect:**
- Sharpe implausibly high (> ~2-3 daily, > ~4-5 intraday with realistic costs) → assume leakage until proven otherwise.
- Equity curve unnaturally smooth / monotonic.
- Add one extra bar of lag between signal and the return it earns (`position.shift(2)` vs `shift(1)`). If a single extra bar of delay destroys most of the edge, you almost certainly had same-bar look-ahead (real, slow-decaying alpha should not evaporate from one bar of latency).
- Audit every aggregation: does each value depend only on data with timestamp <= decision time?

**Fix:**
- Vectorized rule: `pnl_t = position.shift(1) * return_t`. Positions decided at end of t earn the return of t+1.
- Use point-in-time / as-of databases; lag fundamentals by reporting delay (e.g. quarterly data available 45-90 days after period end).
- Compute all rolling stats causally (`rolling`, `expanding`), never global.
- Stamp every series with the time it became *knowable*, and join on that.

### 2. Survivorship bias
**Why it kills you:** Backtesting only names that exist today silently drops the losers, inflating returns and hiding tail risk.
**Detect:** Your universe is pulled from a *current* index/exchange listing. Count tickers per year — flat or growing only? Suspicious. Compare results on a point-in-time universe vs today's universe; large gap = bias.
**Fix:** Use a point-in-time universe with historical index constituents and listing/delisting dates. Include names that later died. For crypto, include delisted/depegged/rugged tokens and dead exchanges.

### 3. Delisting bias
**Why it kills you:** Even with delisted names present, omitting the *terminal return* hides bankruptcies. Delisted-for-cause equities often realize large negative returns (CRSP performance-related delisting returns average roughly -30%, with a long tail toward -100%); dropping them overstates returns, especially for value/distress/small-cap and short books.
**Detect:** Check that delisted securities have a final return row. Short strategies with implausibly clean profits are a red flag.
**Fix:** Apply delisting returns (CRSP delisting return codes for US equities). If unknown, use conservative assumptions (e.g. -100% for bankruptcy, merger value for M&A). For crypto, model exchange halts and zero-recovery events.

### 4. Corporate-action errors
**Why it kills you:** Splits, dividends, spin-offs, rights issues create artificial price jumps. Unadjusted prices fake huge gains/losses; naively back-adjusted prices distort dividend yield and notional.
**Detect:** Scan for single-bar returns beyond a threshold (e.g. |r| > 50% for liquid equities) coinciding with ex-dates. Check that total return ≈ price return + reinvested dividends.
**Fix:**
- For signals: use **split-and-dividend-adjusted total-return** series.
- For PnL/position sizing: trade on *actual* traded prices, account dividends as cash, never on a back-adjusted level as if it were the tradable price.
- Keep both adjusted (for returns/signals) and raw (for fills/notional) and know which you're using at each step.

### 5. Continuous-contract level errors (futures)
**Why it kills you:** Stitched continuous contracts have roll gaps. Back-adjusted (additive) series can go *negative* (notoriously crude oil in 2020) → ratios, logs, and % returns become garbage. Computing returns across a roll without handling the gap fabricates PnL.
**Detect:** Negative or near-zero values in a back-adjusted series; suspicious returns on roll dates; `log(price)` errors/NaNs.
**Fix:**
- Compute **returns from individual contract series**, not from the stitched level; roll PnL is realized by closing the old and opening the new contract (model the roll cost / calendar spread).
- Use ratio-adjusted (Panama/proportional) series for *signal* shape, back-adjusted only with awareness of sign, and never treat a synthetic level as a tradable price.
- Track roll calendar explicitly (volume/OI-based roll, not just expiry).

### 6. Timezone / calendar bugs
**Why it kills you:** Off-by-one-bar errors and cross-venue misalignment silently introduce look-ahead or destroy signals. A US economic print joined to the wrong local date "predicts" the prior session.
**Detect:** Mixed tz-naive and tz-aware timestamps; daily bars that don't match the exchange session; signals that work suspiciously well around session boundaries or DST changes.
**Fix:**
- Store everything in UTC; convert to exchange-local only for session logic.
- Use exchange trading calendars (`pandas-market-calendars` / `exchange_calendars`) for holidays, half-days, session open/close.
- Crypto is 24/7 — define your "day" boundary explicitly (often 00:00 UTC) and use 365 for annualization.
- Align multi-asset/multi-venue data on a common, knowable-time index; be explicit about DST.

### 7. NaN / fillna leakage
**Why it kills you:** `fillna(method='ffill')` is fine; `fillna(mean)` / `bfill` / `interpolate` over the whole series injects future info. Dropping NaNs mid-series can also misalign shifts.
**Detect:** Any global-statistic fill; any `bfill`/interpolation in a feature pipeline; check whether a filled value used data dated after its timestamp.
**Fix:** Forward-fill only (carries last *known* value). Compute fill statistics causally (expanding/rolling). Treat missing data as missing for the period or mask it; never back-fill features used for prediction. Note: even `ffill` can stale-mark a series (carrying a price through a halt) — bound the fill horizon and be aware it understates realized vol.

### 8. Log-vs-simple / compounding errors
**Why it kills you:** Mixing return types corrupts every downstream metric. Summing simple returns overstates; treating log returns as simple misstates compounding and position math.
**Conventions (be consistent):**
- Simple returns aggregate **multiplicatively**: `total = prod(1 + r) - 1`.
- Log returns **add**: `total_log = sum(log_r)`; convert: `simple = exp(sum(log_r)) - 1`.
- Geometric annualized return: `(prod(1+r))**(periods_per_year/n) - 1` (252 daily / 52 weekly / 12 monthly; 365 for crypto/24-7).
**Detect:** Cumulative return via `.cumsum()` on simple returns; Sharpe/vol computed on log returns but reported as simple; portfolio-level returns summed from log asset returns (log returns are *not* additive across assets).
**Fix:** Pick one convention per computation and document it. Use simple returns for cross-sectional/portfolio aggregation; log returns are fine for time-series stats of a single series. Cross-sectional aggregation (portfolio return = weighted sum of asset returns) requires **simple** returns.

---

## Backtest / Execution

### 9. Same-bar / next-bar execution errors
**Why it kills you:** The single most common look-ahead. `pnl = position * return` (unshifted) trades on the same bar that generated the signal — you can't act on a close at that close.
**Detect:** No `.shift()` between signal and return. Edge that vanishes when you add a one-bar lag.
**Fix:** `pnl_t = position.shift(1) * return_t`. A position held over bar t+1 must be decided using only info available at end of bar t. If the signal uses bar t's close and you fill at bar t+1's open, model that explicitly (and be consistent about the open-to-open vs close-to-close return convention — the return you earn must match where you actually entered and exited). Flag any same-bar execution as a bug.

### 10. Unrealistic fills
**Why it kills you:** Assuming you always get the close, the VWAP, the touch, or unlimited size ignores queue position, partial fills, and market impact. Limit orders assumed filled when price merely *touches* the level overstate fill rate badly.
**Detect:** Fills at exact OHLC values; 100% fill rate on limits; orders larger than bar volume; no spread crossed on market orders.
**Fix:**
- Market orders: cross the spread (pay half-spread+); add slippage.
- Limit orders: require price to *trade through* your level (or model queue/fill probability), not just touch it.
- Cap order size vs bar volume (participation cap); model partial fills.
- For backtest realism use conservative assumptions; reconcile against live fills once trading.

### 11. Ignoring transaction costs & slippage
**Why it kills you:** Costless backtests make high-turnover strategies look profitable. Costs scale with turnover; many "alphas" are pure cost mirages.
**Detect:** Cost model is zero or missing. Re-run with realistic costs and watch Sharpe collapse — especially intraday/high-turnover.
**Fix:** Model per-trade cost = commission + half-spread + slippage + (for size) impact. Compute as bps on notional traded. Report net AND gross; track turnover and break-even cost (the cost level that zeroes the edge). If the strategy dies at plausible costs, it's not real.

### 12. Transaction-cost underestimation
**Why it kills you:** A flat bps assumption understates costs for illiquid names, large orders, volatile periods, and crypto (taker fees, funding, wide spreads, thin books).
**Detect:** Same bps applied across liquidity tiers, order sizes, and vol regimes. Compare modeled cost to realized cost (slippage = fill vs decision/arrival price).
**Fix:** Spread + impact that scales with size/ADV and volatility (e.g. square-root impact `~ sigma * sqrt(Q/ADV)`, where Q is order size and ADV is average daily volume in the same units). Use venue-specific maker/taker fees for crypto; include borrow fees for shorts. Stress costs ±50-100% and confirm survival.

### 13. Capacity & liquidity blindness
**Why it kills you:** A strategy that works at \$1M dies at \$100M — impact and partial fills eat the edge. Microcap/illiquid alphas have near-zero real capacity.
**Detect:** Orders as % of ADV unbounded; no AUM scaling test; signal concentrated in illiquid names.
**Fix:** Cap participation (e.g. <= 5-10% of ADV per day per name). Estimate capacity = AUM at which net Sharpe drops below threshold. Re-run at target AUM with size-dependent impact. Report capacity alongside returns.

### 14. Currency & financing/funding omissions
**Why it kills you:** Omitting FX conversion, carry, borrow, and funding flatters returns and hides risk. These are not rounding errors for leveraged/cross-currency books.
**Detect:** Multi-currency PnL summed without FX conversion or hedge cost; futures with no roll/carry; shorts with no borrow fee or no borrow-availability check; crypto perps with no funding; leverage with no financing cost.
**Fix:**
- **Equities:** dividends as cash, short borrow fees, borrow availability (hard-to-borrow names may be uninvestable), margin financing.
- **FX/rates:** carry = interest-rate differential (rolls/swap points); convert PnL to base currency at correct rates.
- **Futures:** roll cost / cost-of-carry; mark-to-market financing.
- **Crypto:** perpetual funding (can dominate PnL), staking/lending yield, gas, withdrawal fees.

---

## Statistics / Validation

### 15. Data-snooping / multiple testing
**Why it kills you:** Try 100 variants, report the best — the best is mostly luck. The naive p-value and Sharpe are inflated by the number of trials.
**Detect:** Count how many strategies/params/universes you actually tried (including informally). Any "we tested a bunch and picked the winner" without correction.
**Fix:** Track trial count. Apply a multiple-testing haircut: **Deflated Sharpe Ratio** / Probability of Backtest Overfitting (Bailey & López de Prado), or Bonferroni/Benjamini-Hochberg on p-values. Reserve a truly untouched holdout. Prefer ex-ante economic hypotheses over search.

### 16. In-sample optimization / overfitting
**Why it kills you:** Many free parameters + one sample = curve-fitting noise. Peak in-sample performance is the worst predictor of OOS.
**Detect:** Sharp, spiky parameter-sensitivity surface (best params sit on a knife-edge); large in-sample vs OOS gap; more params than the data can support.
**Fix:** Minimize parameters; prefer robust plateaus over peaks. Use walk-forward analysis (rolling re-fit + step-forward OOS). Penalize complexity. Report OOS / walk-forward results as the headline, never in-sample.

### 17. p-hacking
**Why it kills you:** Tweaking until p < 0.05, optional stopping, and selective reporting manufacture significance from noise.
**Detect:** Significance appears only after many tweaks; analysis decisions made *after* seeing results; thresholds/universe/period adjusted to cross 0.05.
**Fix:** Pre-register hypotheses, universe, period, and metric before testing. Fix the test plan up front. Report all variants tried, not just the survivor. Use effect sizes and economic significance, not just p-values.

### 18. Leakage in normalization / scaling
**Why it kills you:** Fitting a scaler, PCA, winsorization bounds, or rank on the full sample leaks future distribution info into the past — a subtle, pervasive look-ahead.
**Detect:** `StandardScaler().fit(X)` / `PCA().fit(X)` / global `quantile`/`clip` applied to the whole dataset before splitting; any *time-series* normalization that uses future rows.
**Fix:** Fit transforms on train only, apply to test (`fit` on train, `transform` on test). Inside CV, fit per-fold via a `Pipeline`. For time-series features use causal (expanding/rolling) normalization. Cross-sectional standardization at a single time t (z-score across names using only that timestamp's cross-section) is allowed — it uses only contemporaneous info.

### 19. CV leakage — missing purge/embargo
**Why it kills you:** Plain `KFold(shuffle=True)` or random train/test split on time series leaks: overlapping label windows and serial correlation put near-duplicate info on both sides. OOS looks great, live doesn't.
**Detect:** Any shuffled CV on temporal data; labels built from forward windows (e.g. h-day forward return) with train samples whose label window overlaps the test set; no gap between train and test.
**Fix:** Use time-ordered splits. **Purge** training samples whose label windows overlap the test set, and add an **embargo** gap after the test set (López de Prado, *Advances in Financial ML*). Use `TimeSeriesSplit` as a floor; PurgedKFold/CombinatorialPurgedCV when labels span multiple bars. Walk-forward for final validation.

### 20. Regime overfitting
**Why it kills you:** Tuned to one regime (e.g. the post-2009 bull, low-vol, ZIRP), it breaks when the regime changes. Single-regime backtests overstate robustness.
**Detect:** Backtest spans only one macro regime; performance concentrated in a single period; no drawdown comparable to known stress events.
**Fix:** Test across regimes (rate cycles, vol regimes, bull/bear, liquidity crises: 2008, 2015, 2018, 2020, 2022). Stratify metrics by regime. Stress-test on out-of-regime data. Prefer mechanisms with regime-independent economic rationale.

### 21. Sample-period cherry-picking
**Why it kills you:** Choosing start/end dates that flatter results (skip the drawdown, start after the crash) is silent overfitting to the calendar.
**Detect:** Odd, non-round start/end dates; results highly sensitive to trimming the first/last N months; period conveniently excludes a known stress event.
**Fix:** Use the longest available clean history. Report rolling-window and sub-period stats. Show sensitivity to start/end date. Include all major stress periods in-sample.

### 22. Benchmark gaming
**Why it kills you:** Comparing to cash (or no benchmark) when you should compare to a risk-matched benchmark hides the fact that you're just delivering beta. Alpha vanishes once you net out the right factors.
**Detect:** Benchmark is cash/0% for a long-equity strategy; no beta/factor adjustment; high return that disappears after regressing on market/size/value/momentum.
**Fix:** Choose a risk- and exposure-matched benchmark (e.g. SPX for US long equity, the relevant index for the universe). Report alpha net of factor exposures (regress excess returns on Fama-French / common factors). Report tracking error and information ratio, not just raw return.

### 23. Vol-targeting / risk-scaling look-ahead
**Why it kills you:** Scaling positions to a volatility target is standard, but estimating that vol with the contemporaneous (or full-sample) return — including the bar you are about to trade — leaks the future into your sizing. It quietly deleverages right before bad bars and levers up before good ones, manufacturing Sharpe.
**Detect:** Position size at t uses a vol/covariance estimate that includes bar t's (or later) returns; vol target computed on the whole sample; realized-vol window not lagged relative to the return it scales.
**Fix:** Size from a vol/covariance estimate that uses only data through t-1 (e.g. `vol.shift(1)` or an EWMA updated before the trade). Apply the same causal lag to any risk model, beta hedge, or covariance used for sizing.

---

## Risk / Portfolio (metrics conventions)

State these explicitly so downstream code is consistent (excess = return − per-period risk-free):

- **Annualized vol** = `std(returns, ddof=1) * sqrt(periods_per_year)`.
- **Sharpe** = `mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)`. Annualize **only** by `sqrt(periods_per_year)`. Assumes iid; under autocorrelation apply the **Lo (2002)** correction (positive autocorrelation inflates the naive annualized Sharpe).
- **Sortino** = `mean(excess) / downside_deviation * sqrt(periods_per_year)`; downside deviation = `sqrt( mean( min(excess - MAR, 0)**2 ) )` — sum the squared shortfalls of returns below the target (MAR, usually 0) but divide by the **total** number of observations N, not just the count below target. (Dividing by the below-target count is a common error that inflates Sortino.)
- **Max drawdown** from equity curve: `dd_t = equity_t / cummax(equity)_t - 1`; `maxDD = min(dd_t)` (negative).
- **Calmar** = `annualized_return / abs(maxDD)`.

**Risk-specific traps:**
- **Annualizing an autocorrelated Sharpe by plain sqrt(T):** overstates risk-adjusted return for trend/momentum (positive autocorr) and for illiquid/smoothed marks. *Detect:* significant return autocorrelation (Ljung-Box). *Fix:* Lo (2002) adjustment.
- **`ddof=0` vs `ddof=1`:** numpy/pandas differ — `numpy.std` defaults to population (`ddof=0`), `pandas.Series.std` defaults to sample (`ddof=1`). Use sample (`ddof=1`) for vol/Sharpe and set it explicitly so you don't depend on the default. *Detect:* small-sample metric mismatch between numpy and pandas code paths. *Fix:* set `ddof=1` everywhere.
- **Wrong `periods_per_year`:** 252/52/12, but 365 for crypto/24-7 and ~252 (or ~260 if counting all weekdays) confusion for FX. *Fix:* pin it per asset class.
- **Drawdown on wrong curve:** computing DD on returns not the compounded equity curve. *Fix:* build equity = `(1+r).cumprod()` first.

---

## Production

### 24. Signal / alpha decay
**Why it kills you:** Alphas erode as others discover them, as markets adapt, and as your own trading moves prices. Backtest Sharpe is the *peak*; live is lower and trending down.
**Detect:** Live performance materially below backtest; rolling IC/Sharpe trending toward zero; edge concentrated in early sample.
**Fix:** Monitor live IC and net Sharpe vs backtest on a rolling basis. Set a decay/kill threshold. Discount backtest Sharpe for live expectations (haircut). Re-research and refresh the signal library continuously.

### 25. Factor crowding
**Why it kills you:** If your "alpha" is a known factor (value, momentum, carry, low-vol, size), you share crowded positioning — synchronized deleveraging causes sharp, correlated drawdowns (e.g. the August 2007 quant quake).
**Detect:** High correlation of returns to standard factor portfolios; valuation spreads of your factor at extremes; crowding metrics (short interest, factor-portfolio flows, days-to-cover).
**Fix:** Decompose returns into known factors; isolate residual (idiosyncratic) alpha. Monitor factor valuation/crowding. Diversify across uncorrelated signals. Size down crowded factors; have a deleveraging plan for correlated unwinds.

**IC reminder:** Information Coefficient = cross-sectional correlation (Spearman rank or Pearson) between a factor at time t and **forward** returns over t..t+h. Computing IC against contemporaneous or past returns is look-ahead-flavored nonsense.

---

## Pre-flight checklist (run before trusting any backtest)

**Data**
- [ ] Universe is point-in-time (survivorship-free); delisted names + delisting/terminal returns included.
- [ ] Corporate actions handled; signals on total-return adjusted series, fills/notional on raw prices.
- [ ] Futures returns from individual contracts (not stitched level); roll calendar + roll cost modeled.
- [ ] All timestamps UTC + exchange calendar; no off-by-one; correct `periods_per_year` for the asset.
- [ ] Only `ffill` used (with a bounded horizon); no `bfill`/interpolation/global-statistic fills in features.

**Execution / leakage**
- [ ] `pnl_t = position.shift(1) * return_t` — no same-bar execution anywhere.
- [ ] Extra-lag stress test: edge does not collapse from one additional bar of latency.
- [ ] No full-sample stats (mean/std/quantile/rank/scaler/PCA) used historically; all transforms fit on train only / causally.
- [ ] Position sizing / vol-targeting / risk model use only data through t-1 (no contemporaneous-vol leakage).
- [ ] Fundamentals/membership/estimates lagged by real reporting delay.

**Costs**
- [ ] Commissions + spread + slippage + size/vol-dependent impact modeled; net AND gross reported.
- [ ] Break-even cost computed; survives ±50-100% cost stress.
- [ ] Financing/carry/funding/borrow/FX (and borrow availability) included as relevant to the asset class.
- [ ] Fills realistic (spread crossed, participation-capped, limit-fill logic, partials).

**Statistics / validation**
- [ ] Trial count tracked; multiple-testing haircut (Deflated Sharpe / PBO) applied.
- [ ] Walk-forward / purged+embargoed CV — no shuffled KFold on time series.
- [ ] Headline metrics are OOS; an untouched holdout exists.
- [ ] Tested across regimes and major stress periods; longest clean history; start/end sensitivity shown.

**Risk / reporting**
- [ ] Metrics use `ddof=1`, correct annualization, excess returns; Lo (2002) check if returns autocorrelate.
- [ ] Max drawdown from compounded equity curve; Calmar/Sortino consistent (Sortino downside dev divides by total N).
- [ ] Compared to a risk/exposure-matched benchmark; alpha reported net of factor exposures.
- [ ] Capacity estimated at target AUM; participation vs ADV bounded.
- [ ] Factor decomposition done; residual (idiosyncratic) alpha and crowding assessed.

**Sanity gate:** if daily Sharpe > ~2-3 (or intraday > ~4-5) net of realistic costs, assume a bug — re-audit leakage, fills, and costs before believing it.
