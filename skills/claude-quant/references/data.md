# Data Ingestion & Feature Engineering

The single largest source of fake alpha is bad data discipline, not bad models. Most "amazing" backtests die on PIT, survivorship, or look-ahead bugs. Treat this file as the checklist you run *before* trusting any result.

Conventions used here: simple returns aggregate multiplicatively (`prod(1+r)-1`), log returns add; positions are decided at end of bar `t` and earn `return_{t+1}` (`pnl_t = position.shift(1) * ret_t`). All raw timestamps stored in UTC. Annualization: daily simple/excess returns -> `* sqrt(252)` for vol/Sharpe (crypto 24/7 -> 365, see §8).

---

## 1. Point-in-Time (PIT) discipline

**Rule:** a feature timestamped `t` may use only data that was *observable* at `t`. The enemy is data that was *revised*, *announced late*, or *back-stamped to the period it describes*.

### Why it matters
Fundamentals (earnings, GDP, analyst estimates, index membership) are the worst offenders. A Q4 figure has a `period_end` of Dec 31 but is *announced* in late January / February. If you join on `period_end` you let the strategy "know" earnings ~6 weeks early. This alone can manufacture an entire fake factor.

### As-of joins (the core tool)
Use `merge_asof` / `join_asof` to attach the *most recent value known as of* each price timestamp. Never a regular join on period.

`merge_asof` requires **both** frames sorted by the as-of key (globally, not just within `by` groups), and the left/right keys must share the same dtype and timezone-awareness.

```python
import pandas as pd

# prices: daily, columns include date, symbol
# fund: fundamentals with an explicit ANNOUNCEMENT date (not period_end)
fund = fund.sort_values("announce_date")
prices = prices.sort_values("date")

merged = pd.merge_asof(
    prices, fund,
    left_on="date", right_on="announce_date",
    by="symbol",                 # match within symbol
    direction="backward",        # last value announced AT OR BEFORE date  <-- critical
    allow_exact_matches=True,    # set False if announce is intraday after the close
)
```

`direction="backward"` is the PIT direction. `direction="forward"` or `"nearest"` leaks the future. Flag any `merge_asof` with non-backward direction in a feature pipeline as a bug.

Polars equivalent (lazy-friendly, fast). `join_asof` also requires both frames sorted by the as-of key; default strategy is `"backward"`:

```python
import polars as pl
merged = prices.sort("date").join_asof(
    fund.sort("announce_date"),
    left_on="date", right_on="announce_date",
    by="symbol", strategy="backward",       # backward == PIT; allow_exact_matches=True by default
)
```

### Realistic lag when announcement date is missing
If you only have `period_end`, do **not** assume same-day availability. Apply a conservative lag, then as-of join on the lagged date.

```python
# Common conservative defaults (verify per dataset/jurisdiction):
#   quarterly earnings: period_end + 45 calendar days (10-Q), + 90 for 10-K
#   macro (GDP, CPI):   use the official release calendar, not the reference month
fund["available_date"] = fund["period_end"] + pd.Timedelta(days=45)
```

R note: `data.table` rolling joins (`X[Y, roll=TRUE]`) give as-of semantics; `roll=TRUE` carries the last observation forward and is the backward/PIT direction. `roll=-Inf` looks forward (leaky). Match keys must be sorted.

### Detect PIT leakage
- Shift a known fundamental factor's IC: if IC is large *before* the realistic announcement date and collapses after, you were leaking.
- Compare `period_end` vs `announce_date`: if a pipeline keys on `period_end`, it's almost certainly leaking.
- Vendor "PIT" tables expose `as_of` snapshots — prefer them over "latest" tables.

---

## 2. Survivorship & delisting bias

**Survivorship bias:** building a universe from *today's* listed names (e.g. current S&P 500 members) and backtesting it historically. You silently exclude every company that went bankrupt, was delisted, or dropped from the index. The survivors did better than average by construction → backtest is inflated, often massively (mean-reversion / value / small-cap strategies are hit hardest).

### Build a correct universe
- Use **historical index membership** with effective add/drop dates. Reconstruct the universe *as it was* on each rebalance date.

```python
# membership: symbol, start_date, end_date (end_date NaT if still a member)
def universe_on(date, membership):
    m = membership
    active = (m["start_date"] <= date) & (m["end_date"].isna() | (m["end_date"] > date))
    return set(m.loc[active, "symbol"])
```

- Include **delisted names** with their **delisting return**. A bankruptcy delisting is often a ~ -100% (or large negative) return on the final period. Dropping it entirely is what creates the bias.
- Symbols get reused/recycled. Key on a permanent security id (CRSP `permno`, Bloomberg `figi`, vendor permid), not the ticker. Tickers change on rename/M&A.

### Detect survivorship
- Does your symbol list change over the backtest? If it's constant, you almost certainly have survivorship bias.
- Count delistings in your data over the period and compare to known base rates (US equities: thousands over a decade). Zero delistings = biased source.
- Free sources (e.g. naive Yahoo pulls of "current tickers") are survivorship-biased by default. Use CRSP, Norgate, vendor PIT universes, or reconstruct membership.

---

## 3. Corporate actions: splits & dividends

Three price series exist; know which you have:
1. **Raw / unadjusted close** — what traded that day.
2. **Split-adjusted** — continuity across splits only.
3. **Total-return / fully-adjusted** — adjusts for splits *and* reinvested dividends.

### Why dividend adjustment rewrites history
Adjusting for dividends scales *all prior prices* by a cumulative factor, so historical levels printed in the file change every time a new dividend is paid. Two consequences:
- A "fully adjusted" file is not stable across vintages — re-pulling changes old values. Snapshot it.
- Level-based signals (e.g. "price crossed $50", round-number levels, absolute strike logic) are meaningless on dividend-adjusted prices.

### Which to use
- **Return computation / Sharpe / equity curve:** use total-return (adjusted) prices, or compute returns from raw prices plus an explicit dividend/cash flow. Equity strategies should earn dividends.
- **Signal levels, support/resistance, options strikes, anything comparing to an absolute price:** use raw prices, and handle splits explicitly.
- Best practice: store **raw prices + an adjustment factor series**, and derive adjusted prices on demand. Never throw away raw.

```python
# Vendor adjustment conventions differ: some publish a back-adjusted "adj_close"
# directly; others publish a cumulative factor. Confirm yours before using it.
ret = adj_close.pct_change(fill_method=None)     # simple total return (dividends included)
# back out raw return for level logic:
raw_ret = raw_close.pct_change(fill_method=None)
```

> Always pass `fill_method=None` to `pct_change`. The pandas default (`'pad'`) forward-fills NaNs *before* differencing, fabricating spurious 0% returns across gaps — exactly the leak warned about in §6.

### Back-adjustment pitfall (equities)
If you apply split factors but the factor series itself has a gap/error, you get a phantom jump that looks like a return. See §6 (splits-as-jumps).

---

## 4. Futures continuous contracts

Individual futures contracts expire, so you stitch them into a continuous series. The roll method materially changes prices and signals.

### Roll triggers (when to switch front → next)
- **Calendar:** N days before expiry / first notice day. Simple, deterministic, but can roll while liquidity is still in the front month.
- **Volume:** roll when next contract's volume exceeds front's. Tracks where liquidity actually is.
- **Open interest:** roll on OI crossover. Similar intent, slightly smoother.

Volume/OI rolls are generally preferred for tradability; calendar is fine for research if consistent. **Store the roll schedule** (date, from_contract, to_contract, roll_method) so PnL is reproducible and you can map continuous-series signals back to the actual contract you'd trade.

### Adjustment methods
- **Back-adjustment (difference / panama):** shift the historical series by the price *gap* at each roll so the curve is continuous. **Prices are not real and can go negative** if cumulative roll gaps exceed the price level.
- **Ratio / proportional adjustment:** multiply by the price *ratio* at each roll. Keeps prices positive and keeps *percentage* returns intact across the roll (except on the roll day itself).
- **Unadjusted (just stitch):** continuous but has jumps at each roll.

### Why this breaks signals — detect/fix
- **Back-adjusted prices can be negative or near-zero.** Then `pct_change()` is garbage (division by ~0, sign flips) and any level-based or percentage signal (RSI, %-bands, log returns) is invalid.
  - **Fix:** compute **returns from the underlying contracts** (per-contract returns, with the roll handled by switching which contract is held), not from `pct_change` of a back-adjusted level. For research signals that need a level, prefer **ratio-adjusted** series.
- **Roll PnL must be modeled:** on the roll date you exit one contract and enter another. The per-contract return method below earns each held contract's own return and does **not** book the inter-contract gap as a return (no spurious roll PnL). The only real roll cost is transaction cost/slippage on the exit+entry — model that separately. Back-adjustment instead bakes the gap into history; if you use it, don't *also* charge the gap as a return.
- Log returns on a back-adjusted series that crosses zero produce NaN/complex — a clear red flag you're on the wrong series.

```python
import pandas as pd

# Robust continuous return: per-contract simple returns, no leakage across roll.
# panel: long frame with columns contract, date, settle
# roll_sched: Series indexed by date giving the contract HELD that day (decided in
#             advance / known at the prior close), so it carries no look-ahead.
held = panel.pivot(index="date", columns="contract", values="settle").sort_index()
dates = held.index

active = roll_sched.reindex(dates).ffill()           # which contract is held each day

# fill_method=None is essential: each contract is NaN outside its own life, and the
# pandas default would forward-fill stale prices and fabricate 0% returns in the gaps.
ret = held.pct_change(fill_method=None)

# return of whatever contract was held that day; the roll day is NaN for the newly
# entered contract (its first observation) -> no fake gap return, drop or treat as 0.
cont_ret = pd.Series(
    [ret.at[d, active.at[d]] for d in dates], index=dates, name="ret"
)
```

---

## 5. Timestamp alignment, calendars, resampling

### Timezones
- **Store everything in UTC.** Keep the exchange/local tz as metadata. DST shifts session times in local terms; UTC is unambiguous.
- Daily bars are tz-tricky: a "2020-03-15" US equity bar means the *US/Eastern* session. Don't compare it date-for-date with a Tokyo bar of the same label — they're different real-time windows.

```python
ts = pd.to_datetime(raw_ts, utc=True)               # always tz-aware UTC
local = ts.tz_convert("America/New_York")           # for session logic
```

### Exchange calendars
Use a real calendar library; never assume Mon–Fri = trading days.

```python
import pandas_market_calendars as mcal
nyse = mcal.get_calendar("NYSE")
sched = nyse.schedule(start_date="2020-01-01", end_date="2024-12-31")
# sched has market_open / market_close per session, including EARLY CLOSES (half-days)
sessions = mcal.date_range(sched, frequency="1D")
```

`exchange_calendars` (the `xcals` package) is the other standard; `pandas-market-calendars` can use `exchange_calendars` definitions under the hood. Both encode holidays, half-days (e.g. day after Thanksgiving, Christmas Eve), and special closures. Ignoring half-days corrupts intraday resampling and VWAP.

### Resampling without leakage
- **Label bars by their close** and beware pandas' `label`/`closed` defaults. A bar covering `[t, t+Δ)` must be timestamped at its *close* and only become available *after* its close.
- `df.resample("5min")` by default uses **left** label and **left** closed for most freqs → a bar labeled `09:30` actually contains `[09:30, 09:35)` data not fully known until `09:35`. If you then treat the `09:30` label as "known at 09:30", you leak. (Verified on pandas 2.2: the default-labeled bars start at `09:30`; right-labeled at `09:35`.)

```python
# Make the timestamp mean "data complete as of here":
bars = df.resample("5min", label="right", closed="right").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
)
# Now bars.loc[t] is fully observable at t. Trade on bars.shift(1) to be safe.
```

- Honor session boundaries: resample within a session, don't let a bar span the overnight gap or merge across half-day closes. Group by trading session/date first when needed.

### Detect alignment bugs
- Plot bar timestamps vs known open/close — off-by-one-bar = leakage.
- Check for "trades" or volume on holidays/weekends in equities/futures → calendar or tz error.

---

## 6. Data quality

### Bad ticks / outliers
- Detect: returns beyond N MADs (robust) or implausible single-bar moves; price ≤ 0; high < low; close outside [low, high]; zero-volume bars with price changes.

```python
r = close.pct_change(fill_method=None)
med = r.median()
mad = (r - med).abs().median()
suspect = (r - med).abs() > 10 * 1.4826 * mad        # robust z > 10 (MAD->sigma scaling)
```

- Fix: cross-check against a second source; for a confirmed bad tick, drop or replace — but log it and **never forward-fill a price into a return as if it traded.**

### Gaps
- Distinguish *missing data* (vendor gap) from *no trading* (holiday/halt). Reindex to the **exchange calendar**, not to a naive business-day range, so real non-sessions don't appear as gaps.

### Splits-as-jumps
A 2:1 split halves the raw price overnight → a naive `pct_change()` reads as -50% return. Detect large overnight jumps that coincide with split events and confirm against a corporate-actions table; apply the adjustment factor rather than letting it flow into returns.

```python
overnight = close / close.shift(1) - 1
flag = overnight.abs() > 0.30                          # candidate split/error; verify vs CA table
```

### Forward-fill & interpolate dangers (leakage)
- `ffill` is PIT-safe (carries past forward) **only** if the value was truly known and stale-carry is acceptable. `ffill` of a price into a *return* fabricates a 0% return on a day nothing traded — it understates vol and inflates Sharpe. The same trap hides inside `pct_change()`'s default `fill_method='pad'`; always pass `fill_method=None`.
- `bfill`, `interpolate()`, and `fillna(method="bfill")` use **future** values → look-ahead. Never use them in a feature pipeline.
- `interpolate()` (linear, time, spline) is two-sided and leaks by construction. Banned for features.

```python
features = features.ffill()          # OK (carries known-past), document staleness
# features = features.interpolate()  # BUG: uses future points
# features = features.bfill()        # BUG: uses future values
```

### Validation checks (run on every dataset)
- Monotonic, unique, tz-aware timestamps; no duplicate (symbol, ts).
- `low <= open,close <= high`; price > 0; volume >= 0.
- Returns within sane bounds; flag/quarantine outliers.
- Row counts per session ≈ expected bars (catches gaps/half-days).
- No timestamps on non-sessions.

---

## 7. Alt data alignment & revisions

- **PIT lag is mandatory.** Credit-card panels, web traffic, satellite, app downloads etc. are *collected* over a window and *delivered* with latency. Align to the **vendor delivery/availability timestamp**, never the activity date. If unknown, apply a conservative lag and stress-test sensitivity to it.
- **Revisions / restatements:** many alt and macro series are revised after first release. Backtest on the *first-release vintage* (what you'd have seen), not the final revised value. Prefer vendors with vintage/`as_of` snapshots.
- **Entity mapping leakage:** mapping alt-data entities to tickers using *today's* corporate structure (post-M&A, post-rename) leaks. Map using the structure as of `t`.
- **Detect:** if an alt factor's IC is suspiciously front-loaded relative to the realistic availability date, you're aligned to activity date, not delivery date.

---

## 8. Crypto specifics

- **24/7, no official close.** Pick and *document* a daily cutoff (e.g. 00:00 UTC) and use it everywhere; "daily" is arbitrary, so consistency matters. Use **365** periods/year for annualization (24/7 trading), not 252.
- **No single price.** Each exchange has its own book; use a consistent venue or a composite index. Cross-exchange arbitrage means prices differ — don't mix venues within one series.
- **Perp funding rates:** perpetual swaps pay/receive funding (typically every 8h). For a perp strategy, funding is a real cash flow — include it in PnL. A long-perp carry that ignores funding is fake.

```python
# Per-period perp PnL = price PnL + funding cash flow.
# Convention below: positive funding_rate => longs PAY shorts, so a long position
# (position > 0) loses funding. Verify the sign against your venue's spec.
pos = position.shift(1)                                 # held into the period (no lookahead)
pnl = pos * price_ret - pos * funding_rate
```

- **Clock skew & idiosyncrasies:** exchange timestamps drift and use different conventions (ms vs s, local vs UTC, trade-time vs receive-time). Normalize to UTC, sanity-check ordering, dedupe. Maintenance windows and outages create gaps/halts that aren't holidays.
- **Symbol churn:** tokens get renamed/redenominated; handle like equity ticker reuse.

---

## 9. Feature engineering

### Time-series vs cross-sectional normalization
- **Cross-sectional** (across names at one timestamp): rank/z-score within the universe each day. Used for relative-value / factor models. Naturally market-neutral-ish. Uses only same-day data → no time leakage.
- **Time-series** (one name over its own history): z-score vs its rolling/expanding history. Must use **trailing** windows only.

```python
# cross-sectional z at each date (no time leakage: uses only the same-day cross section).
# ddof=0 (population) is conventional cross-sectionally since you observe the whole
# universe that day; use ddof=1 if you prefer sample std — just be consistent.
cs_z = df.groupby("date")["factor"].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

# time-series z, trailing only. rolling() includes the current point; if "factor"
# itself is the model input for predicting a FUTURE return that is fine (the value
# is known at t). Shift the inputs by one bar if any component is not yet observable.
g = df.sort_values(["symbol", "date"]).groupby("symbol")["factor"]
roll_mean = g.transform(lambda s: s.rolling(252).mean())
roll_std = g.transform(lambda s: s.rolling(252).std(ddof=1))
df["ts_z"] = (df["factor"] - roll_mean) / roll_std
```

### Winsorization (outlier control, not removal)
Clip extremes (e.g. to 1st/99th pct or ±k MADs) before normalizing so a few prints don't dominate z-scores. Cross-sectionally, winsorize within each date (thresholds from the same-day cross section, so no leakage). Time-series winsorization must use trailing quantiles only.

```python
def winsorize(s, lo=0.01, hi=0.99):
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)

cs = df.groupby("date")["factor"].transform(winsorize)
```

### Sector / beta neutralization
Remove unwanted exposure so the factor is orthogonal to it (otherwise you're just betting on the sector or on market beta).
- **Sector-neutral:** demean (and optionally z-score) the factor *within each sector each day*.
- **Beta-neutral:** regress factor (or returns) on beta cross-sectionally each day and keep the residual.

```python
import numpy as np
import pandas as pd

# sector-neutral cross-sectional factor (within sector, within date)
df["fac_sn"] = df.groupby(["date", "sector"])["factor"].transform(lambda x: x - x.mean())

# beta-neutralize via cross-sectional OLS residual (per date).
# Build the residual column aligned by the original index to avoid misalignment.
def residualize(g, y="factor", x="beta"):
    sub = g[[y, x]].dropna()
    if len(sub) < 2:
        return pd.Series(np.nan, index=g.index)
    X = np.c_[np.ones(len(sub)), sub[x].to_numpy()]
    b, *_ = np.linalg.lstsq(X, sub[y].to_numpy(), rcond=None)
    resid = sub[y].to_numpy() - X @ b
    return pd.Series(resid, index=sub.index).reindex(g.index)

df["fac_bn"] = df.groupby("date", group_keys=False).apply(residualize)
```

### Stationarity transforms
Most ML/stat models assume stationarity; raw price levels are non-stationary (unit root) and will overfit trends.
- **Simple returns** `p_t/p_{t-1}-1` or **log returns** `ln(p_t/p_{t-1})` — stationary but discard all memory/level info.
- **Fractional differencing (López de Prado):** difference by a fractional order `d ∈ (0,1)` to make the series stationary while retaining *maximum memory* (correlation with level). Choose the smallest `d` passing an ADF test.

```python
import numpy as np

def frac_diff_weights(d, thresh=1e-3):
    # Smaller thresh -> longer weight vector. With thresh=1e-3 and d~0.4 this is ~55
    # weights; thresh=1e-5 produces >1400 weights, which makes the fixed-width rolling
    # window below longer than typical series and return all-NaN. Tune thresh so that
    # len(weights) << len(series).
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_); k += 1
    return np.array(w[::-1])

def frac_diff(series, d, thresh=1e-3):
    w = frac_diff_weights(d, thresh)
    width = len(w)
    if width > len(series):
        raise ValueError(
            f"weight window ({width}) exceeds series length ({len(series)}); raise thresh"
        )
    # rolling window uses only the current and past `width-1` points -> trailing/PIT-safe.
    return series.rolling(width).apply(lambda x: np.dot(w, x), raw=True)
```

Test stationarity with `statsmodels.tsa.stattools.adfuller`. Sweep `d`, pick the smallest `d` where ADF rejects the unit root. (This fixed-width-window form is the LdP "FFD" variant; the alternative expanding-window form weights the full history.)

### CRITICAL: fit scalers on train only
Any statistic computed over the full sample leaks test information into training.
- `StandardScaler`, `MinMaxScaler`, PCA, quantile/winsor thresholds, target encoders, imputation means, fractional-diff thresholds — **fit on train, then `transform` test.** Never `fit_transform` on the full dataset.
- For time series, "train only" also means *past only*: even within training, prefer expanding/rolling fits over a single global fit if the model will be applied walk-forward.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)        # fit on train
X_train_s = sc.transform(X_train)
X_test_s = sc.transform(X_test)           # transform only -> no leakage
```

- **CV for time series:** use purged + embargoed walk-forward (López de Prado). Plain `KFold`/`shuffle` leaks because labels span overlapping windows and future rows train on past tests. Purge training samples whose label window overlaps the test window; add an embargo gap (a few % of the sample) *after* the test block so serially correlated rows just past the test set don't leak into training.
- **Detect leakage:** if CV score >> walk-forward / live score, suspect global scaler fit, shuffled CV, or PIT violations upstream.

---

## 10. Storage & performance

### Format & layout
- **Parquet** (columnar, compressed, typed) is the default for research data. Avoid CSV for anything large (slow, untyped, loses tz).
- **Partition** by `date` (or `year`/`month`) and optionally `symbol` so queries prune files via predicate pushdown.

```python
df.to_parquet("data/bars", partition_cols=["year", "symbol"], engine="pyarrow")
# read just what you need; filters push down to file/row-group level
sub = pd.read_parquet(
    "data/bars",
    filters=[("year", "=", 2023), ("symbol", "in", ["AAPL", "MSFT"])],
    engine="pyarrow",
)
```

Partition granularity tradeoff: too fine (per-symbol-per-day) → millions of tiny files, slow listing; too coarse → no pruning. By month or year, with symbol as a secondary partition for wide universes, is usually right.

### pandas vs polars
- **pandas:** ubiquitous, rich ecosystem (statsmodels, sklearn, market-calendar libs). Eager; whole frame in memory; slower and heavier on large panels.
- **polars (lazy):** multithreaded, Arrow-backed, much lower memory; `scan_parquet().filter(...).group_by(...).collect()` builds a query plan and only materializes the result — great for large universes and out-of-core-ish workloads.

```python
import polars as pl
out = (
    pl.scan_parquet("data/bars/**/*.parquet")        # lazy
      .filter(pl.col("date") >= pl.date(2023, 1, 1))
      .group_by("symbol")
      .agg((pl.col("ret").std(ddof=1) * (252 ** 0.5)).alias("ann_vol"))
      .collect()                                      # executes optimized plan
)
```

Use polars for heavy ETL/aggregation; convert to pandas (`.to_pandas()`) at the boundary where you need sklearn/statsmodels.

### Memory & perf
- Downcast dtypes (`float32` where precision allows, categorical for symbols/sectors) — large panels shrink several-fold. Beware `float32` for cumulative-product equity curves over long horizons (precision loss); keep returns/PnL in `float64`.
- Vectorize; if a genuine hot loop remains (path-dependent sim, custom indicator), drop to **numba** (`@njit`) or a **Rust/Cython** extension rather than a Python loop.
- Snapshot adjusted/PIT datasets to immutable parquet with a version/vintage tag so results are reproducible even after vendor revisions.

---

## Quick pre-trust checklist
- [ ] Features use only data observable at `t` (as-of backward joins; realistic fundamental lag).
- [ ] Universe is PIT (historical membership) and includes delisted names + delisting returns.
- [ ] Returns use total-return prices; level signals use raw prices; raw retained.
- [ ] `pct_change(fill_method=None)` everywhere (no padded-NaN fake returns).
- [ ] Futures: documented roll method + stored schedule; returns from contracts, not `pct_change` of back-adjusted levels; no negative-price log returns.
- [ ] Timestamps UTC; exchange calendar honored (half-days); bars labeled by close; positions shifted.
- [ ] No `bfill`/`interpolate` in features; `ffill` doesn't fabricate returns; outliers/splits validated.
- [ ] Crypto annualized with 365; perp funding booked into PnL.
- [ ] Scalers/PCA/thresholds fit on train only; CV is purged + embargoed walk-forward.
