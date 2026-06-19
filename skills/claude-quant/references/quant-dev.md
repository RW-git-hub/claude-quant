# Quant Development & Production Code

Reference for writing correct, fast, reproducible quant code and shipping signals to production. Detect/fix framing throughout. Conventions: simple returns compound multiplicatively (`prod(1+r)-1`), log returns add; positions earn the *next* bar (`pnl_t = position.shift(1) * return_t`).

---

## 1. Codebase structure

Separate concerns so research can move fast without contaminating live trading. The cardinal rule: **research code and live-execution code share libraries, not entry points.** A backtest that accidentally reads tomorrow's price is a bad afternoon; live code that does it loses money.

### src layout

```
myquant/
  pyproject.toml          # single source of deps + tool config (ruff, mypy, pytest)
  src/myquant/
    __init__.py
    config.py             # typed config loading (see below)
    data/                 # ingestion, cleaning, alignment. NO signal logic.
      loaders.py
      calendars.py        # trading sessions, holidays, session boundaries
    signals/              # pure: features in -> signal out. NO I/O, NO execution.
      momentum.py
      factors.py
    backtest/             # simulation engine, cost models, fills
      engine.py
      costs.py
    analysis/             # metrics, tearsheets, attribution
      metrics.py
      plots.py
    execution/            # LIVE ONLY: brokers, order mgmt, state, reconciliation
      broker.py
      oms.py
      runner.py           # the live job entry point
  tests/
  notebooks/              # exploration only; never imported by src
  scripts/                # CLI entry points (research runs, live runner)
  configs/                # yaml/toml configs per strategy/environment
```

Why src layout (not flat): installing the package (`pip install -e .`) means tests import the *installed* package, catching missing-module and packaging bugs that a flat layout hides. Notebooks `import myquant`, never the reverse.

**Dependency direction (enforce, don't just hope):** `signals` and `analysis` must not import `execution`. `backtest` imports `signals`, `data`, `costs` — not `execution`. The live runner imports `signals` + `execution`. This guarantees the same signal function feeds both backtest and live.

Detect: `grep -rn "import.*execution" src/myquant/signals src/myquant/backtest` should return nothing. Add an import-linter contract or a test that asserts the dependency graph.

### Config management

Use typed config objects, not loose dicts of magic numbers. Pydantic gives validation + clear errors; dataclasses are lighter if you don't need parsing.

```python
from pydantic import BaseModel, Field, field_validator

class CostModel(BaseModel):
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    half_spread_bps: float = 0.0

class StrategyConfig(BaseModel):
    name: str
    universe: list[str]
    lookback: int
    rebalance: str = "1d"          # pandas offset alias
    # Use default_factory so each config gets its own CostModel instance,
    # never a single shared mutable default.
    costs: CostModel = Field(default_factory=CostModel)
    seed: int = 0

    @field_validator("lookback")
    @classmethod
    def positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("lookback must be > 0")
        return v

def load_config(path: str) -> StrategyConfig:
    import tomllib   # Python 3.11+ stdlib; for <3.11 use the `tomli` backport
    with open(path, "rb") as f:
        return StrategyConfig(**tomllib.load(f))
```

Rules: no hardcoded params in signal functions — pass config in. Keep environment-specific values (API keys, DB hosts) in env vars / secrets, never in versioned config. One config file per (strategy, environment) so a backtest and its live deployment are the *same file path*, diffed by env.

---

## 2. Numerical correctness

Most "alpha" that evaporates in production was a numerical bug in research. These are the recurring ones.

### Simple vs log returns: cumprod vs cumsum

```python
import numpy as np

# Simple returns -> compound multiplicatively
equity = (1 + simple_rets).cumprod()          # NOT cumsum
total_return = (1 + simple_rets).prod() - 1

# Log returns -> add
cum_log = log_rets.cumsum()
equity = np.exp(cum_log)                       # equity (gross), normalize to 1 at t0 if desired
total_return = np.expm1(log_rets.sum())        # expm1 for small-value accuracy; = exp(sum) - 1
```

Detect: if your equity curve is suspiciously *linear* on a long horizon, you probably `cumsum`'d simple returns. Fix: pick one convention per pipeline and assert it. Converting: `log = np.log1p(simple)`; `simple = np.expm1(log)` (use `log1p`/`expm1`, not `log(1+x)`/`exp(x)-1`, for precision near zero).

Never average log returns and report it as "mean return" of a simple-return strategy — the *arithmetic* mean of log returns is ≤ log of the arithmetic mean of simple returns (Jensen); the gap is the volatility drag (~σ²/2 for small returns). The geometric mean simple return equals `expm1(mean(log_rets))`.

### Off-by-one in shift/diff

```python
ret_t        = price.pct_change()             # uses price_t / price_{t-1} - 1  (OK, info at t)
signal_t     = (sma_fast > sma_slow).astype(int)
# position decided at close of t earns return of t+1:
pnl          = signal_t.shift(1) * ret_t      # CORRECT
pnl_wrong    = signal_t * ret_t               # LOOK-AHEAD: same-bar execution
```

`diff(n)` and `shift(n)`: `x.diff(1).iloc[0]` is NaN; `x.shift(1)` introduces a NaN at the front and pushes the last observed value out only if you later truncate — the value isn't lost, the series stays the same length with a leading NaN. After any `shift`, the first row(s) are unusable — don't let them silently become 0 (e.g. via a careless `fillna(0)`), which would fabricate a flat position/return.

Detect look-ahead numerically: see the future-shift test in §5. Any signal whose value at time t changes when you alter data at t+1 is leaking.

### NaN propagation and alignment

pandas arithmetic aligns on the index/columns *labels*, not position. Misaligned indices silently produce NaN where labels don't match and broadcast where they do — a top source of "why is my Sharpe NaN" and "why did half my universe vanish."

```python
import pandas as pd

a = pd.Series([1, 2, 3], index=[0, 1, 2])
b = pd.Series([10, 20, 30], index=[1, 2, 3])
a + b      # index 0 and 3 -> NaN; only 1,2 add

# Fix: align explicitly and decide the fill
a2, b2 = a.align(b, join="inner")            # or join="outer", fill_value=0
```

Rules:
- After every merge/join/reindex, check shape and NaN count. `assert df.notna().all().all()` or count and log.
- `mean()`, `sum()`, `std()` *skip* NaN by default (`skipna=True`) — a column that's 80% NaN still returns a number, often biased. Decide explicitly: `min_count=`, or drop, or impute.
- `np.nan == np.nan` is False; use `pd.isna`. Sorting puts NaN last regardless of ascending/descending.
- Forward-filling prices is usually acceptable, but ffill *across a session/halt gap* can itself create stale-price look-ahead; forward-filling *returns* or *signals* fabricates look-ahead or stale positions. Forward-fill only what is genuinely "last known value still valid," and never across boundaries where the value should reset.

### Float pitfalls

- Never test floats with `==` (except an exact 0 you set yourself). Use `np.isclose` / `math.isclose` with an explicit tolerance.
- Catastrophic cancellation: computing variance as `E[x²]-E[x]²` loses precision; use `np.var`/Welford. For money, consider `Decimal` or integer minor-units (cents/satoshis) in execution code where rounding must be exact.
- `float32` saves memory but accumulates error in long cumulative sums (equity curves, rolling sums over millions of intraday bars). Keep prices/returns `float64` for accounting; downcast features only after you've validated tolerance.
- Summation order matters at `float32` over large N; `np.sum` uses pairwise summation, a naive Python loop does not. `math.fsum` is exact for a 1-D Python iterable when you need it.

---

## 3. pandas correctness and anti-patterns

### Chained indexing / SettingWithCopyWarning

```python
df[df.vol > 0]["signal"] = 1          # WRONG: writes to a temporary copy, silently no-op
df.loc[df.vol > 0, "signal"] = 1      # RIGHT: single .loc set
```

The warning means "I can't tell if you're writing to a view or a copy." Treat it as an error, not noise. Under pandas 2.x with Copy-on-Write opted in (`pd.options.mode.copy_on_write = True`), and by default in pandas 3.0, chained *assignment* never propagates to the parent — it reliably no-ops rather than sometimes working. Turn CoW on so the behavior is consistent and most defensive `.copy()` calls become unnecessary; the chained-assignment no-op then raises a `ChainedAssignmentError` so you catch it instead of guessing.

### iterrows vs vectorization

`iterrows` is roughly two to three orders of magnitude slower than vectorized ops on wide DataFrames, and it yields dtype-coerced Series (your int becomes float64, your bool becomes object) because each row is materialized as a single mixed-dtype Series. Reach for vectorization first.

```python
# Slow
out = []
for i, row in df.iterrows():
    out.append(row.close * row.qty)
# Fast
out = df["close"] * df["qty"]
```

When you *think* you need a loop (path-dependent: stops, position carry, inventory), the order is: (1) vectorize with `cumsum`/`cumprod`/`np.where`/`shift`; (2) `df.itertuples(index=False)` (namedtuples, fast, dtype-preserving per column) if truly sequential; (3) numba on numpy arrays for the proven hot loop (§4). Never `iterrows`. `apply(axis=1)` is also a Python loop in disguise — same cost.

### copy vs view, dtype issues

- Slicing may return a view or a copy; relying on which is fragile. Under CoW, mutating a slice never writes back to the parent (copy-on-write semantics). Pre-CoW (before 3.0, CoW off), write `df = df.copy()` when you intend an independent object.
- `object` dtype columns are slow and memory-heavy. Cast strings to `category`, datetimes to `datetime64[ns]`, and watch for accidental `object` after concatenating mixed types.
- Integer columns with any NaN become `float64` (or use nullable `Int64`). A volume column that silently goes float is a smell.
- Tz-naive vs tz-aware datetimes don't compare/align correctly (comparing them raises or, in arithmetic, misaligns). Pick UTC internally; localize on display only.

### When to switch to polars

Switch when: dataset is larger than memory or you're memory-bound; you need real speed on group-bys/joins/rolling over tens of millions of rows; you want lazy query optimization and predictable multi-threading; or you're fighting pandas alignment/CoW surprises. polars expressions are explicit (no hidden index alignment), columnar, multi-threaded, and the lazy API can stream.

```python
import polars as pl

lf = pl.scan_parquet("ticks/*.parquet")       # lazy, larger-than-memory, predicate pushdown
out = (
    lf.filter(pl.col("symbol") == "BTCUSDT")
      .sort("ts")
      .with_columns(
          ret = pl.col("close").pct_change(),
          sig = (pl.col("close").rolling_mean(10) > pl.col("close").rolling_mean(30)).cast(pl.Int8),
      )
      .with_columns(pnl = pl.col("sig").shift(1) * pl.col("ret"))   # shift = no look-ahead
      .collect(engine="streaming")              # newer polars; older versions: collect(streaming=True)
)
```

Note: rolling/shift correctness in polars depends on rows being sorted by time first — hence the explicit `.sort("ts")` before the rolling windows. Keep polars at the data/feature layer; convert to pandas/numpy at the boundary for libraries that expect them (`df.to_pandas()`, `df.to_numpy()`). polars has no row index — alignment is via explicit joins, which removes a whole class of pandas bugs. R note: `tidyverse`/`data.table` fill the same role; `data.table`'s in-place `:=` is the polars-like speed tier.

---

## 4. Performance discipline

Order of operations, never skip a step: **vectorize → profile → optimize the proven hot path.** Optimizing before profiling wastes time on code that isn't the bottleneck. (Vectorize first because it removes the most common bottleneck outright; profile to find what remains; only then hand-optimize.)

### Profile before optimizing

```bash
# Whole-program, function-level
python -m cProfile -o prof.out scripts/run_backtest.py
python -c "import pstats; pstats.Stats('prof.out').sort_stats('cumulative').print_stats(20)"

# Line-level on a suspect function (decorate it with @profile)
kernprof -l -v scripts/run_backtest.py        # line_profiler

# CPU + memory + GPU, low overhead, great for mixed numpy/IO
scalene scripts/run_backtest.py
```

Profile on representative data sizes — a bottleneck at 1k rows is often not the bottleneck at 10M.

### Then optimize the hot loop

Only after profiling proves a Python loop dominates:

```python
import numba as nb
import numpy as np

@nb.njit(cache=True, fastmath=False)            # fastmath off for financial accuracy
def trailing_stop_pnl(prices: np.ndarray, entry: int, stop_pct: float) -> float:
    peak = prices[entry]
    for i in range(entry, prices.shape[0]):
        peak = max(peak, prices[i])
        if prices[i] <= peak * (1.0 - stop_pct):
            return prices[i] / prices[entry] - 1.0
    return prices[-1] / prices[entry] - 1.0
```

`numba` is the cheapest win for numeric loops over numpy arrays (pass arrays, not DataFrames; no Python objects inside). `Cython` when you need C interop or fine control. `Rust` (via `pyo3`/`maturin`, or use polars which is Rust) for production-grade libraries and parsers. Caveat `fastmath=True`: it reorders float ops and assumes no NaN/inf — measure that it doesn't change results before trusting it in PnL math.

### Memory, dtypes, chunking, parallelism

- Downcast *after* validating: `float64→float32`, `int64→int32`, strings→`category`. Can cut memory 2-4x. Don't downcast the accounting layer (see §2).
- Chunk huge files: `pd.read_parquet(columns=[...])` to read only needed columns; `pl.scan_*` for streaming; process by date partition.
- Embarrassingly parallel work (per-symbol backtests, parameter sweeps, walk-forward folds): `joblib.Parallel(n_jobs=-1)` or `concurrent.futures.ProcessPoolExecutor`. Threads only help if the inner work releases the GIL (numpy/numba `nogil=True`, polars, I/O); otherwise use processes.

```python
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(delayed(backtest_symbol)(s, cfg) for s in cfg.universe)
```

- GPU note: `cudf` (pandas-like) / `cupy` (numpy-like) pay off for very large columnar transforms and matrix-heavy work (large factor regressions, deep nets). Transfer cost dominates for small data — only worth it when compute >> PCIe transfer. Don't reach for it before exhausting vectorization + numba.

---

## 5. Testing quant code

Quant bugs are usually silent (wrong number, not a crash). Tests must check *values*, not just "it ran."

### Analytic metric tests

Hand-compute small cases and assert the formula. These pin your conventions.

```python
import numpy as np, pandas as pd
from myquant.analysis.metrics import sharpe, max_drawdown, annualized_return

def test_sharpe_known():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)   # rf=0, sample std (ddof=1)
    assert np.isclose(sharpe(r, rf=0.0, periods=252), expected)

def test_max_drawdown_known():
    eq = pd.Series([100, 120, 90, 110, 80])              # peak 120 -> trough 80
    assert np.isclose(max_drawdown(eq), 80 / 120 - 1)    # = -0.3333...

def test_annualized_return_geometric():
    r = pd.Series([0.0] * 252)                           # flat year
    assert np.isclose(annualized_return(r, 252), 0.0)
```

Convention reminders these tests pin: Sharpe uses `std(ddof=1)` on excess returns × `sqrt(periods_per_year)`; max drawdown is `min(equity / equity.cummax() - 1)` (≤ 0); annualized return is geometric, `(1 + r).prod() ** (periods_per_year / n) - 1`.

### Golden / regression tests

Freeze a known-good output (equity curve, signal series) on a fixed input; fail if it changes. Catches accidental behavior drift during refactors.

```python
def test_backtest_golden():
    out = run_backtest(load_fixture("aapl_2020.parquet"), cfg)
    golden = pd.read_parquet("tests/golden/aapl_2020_equity.parquet")["equity"]
    pd.testing.assert_series_equal(out.equity, golden, rtol=1e-9)
```

Regenerate goldens deliberately (a make target), review the diff in PR — never auto-overwrite.

### Look-ahead tests (the most important quant-specific test)

A correct signal at time t depends only on data ≤ t. So mutating the future must not change the past.

```python
def test_no_lookahead_future_mutation():
    df = make_synthetic_ohlcv(n=500, seed=0)
    sig = compute_signal(df)
    # Corrupt everything strictly AFTER t0, then recompute.
    # Use a bounded, distinct perturbation (not *1e6, which can overflow to inf and
    # interact oddly with NaNs/rolling windows). A reproducible reshuffle or additive
    # shock is cleaner and still detects any forward read.
    t0 = 300
    df2 = df.copy()
    rng = np.random.default_rng(1)
    fut = df2.iloc[t0 + 1:]
    df2.iloc[t0 + 1:] = fut.sample(frac=1.0, random_state=1).to_numpy()  # shuffle the future
    sig2 = compute_signal(df2)
    pd.testing.assert_series_equal(sig.iloc[:t0 + 1], sig2.iloc[:t0 + 1])

def test_pnl_uses_shifted_position():
    # Zeroing future returns must not change realized pnl up to t, and pnl_t must use
    # position.shift(1): assert pnl.iloc[k] == position.iloc[k-1] * ret.iloc[k].
    df = make_synthetic_ohlcv(n=100, seed=0)
    pos, ret = build_positions_and_returns(df)
    pnl = run_pnl(pos, ret)
    assert np.isclose(pnl.iloc[5], pos.iloc[4] * ret.iloc[5])   # shifted by exactly one bar
    assert pd.isna(pnl.iloc[0]) or pnl.iloc[0] == 0.0           # first bar has no prior position
```

If a future-mutation test fails, your signal reads forward data (rolling-window centering, `bfill`, full-sample fit, leakage in feature scaling). Also test the CV layer: purge + embargo (Lopez de Prado). For each train/test split, assert (a) no training observation whose *label window* overlaps the test interval survives — purge removes them — and (b) an embargo gap of `h` bars after the test interval is dropped from training, so serially-correlated leakage past the boundary can't bleed in. A test should construct a split and assert both the purge and the embargo actually removed the offending indices.

### Property-based tests (hypothesis)

Assert invariants over many generated inputs.

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp

@given(hnp.arrays(np.float64, st.integers(2, 200),
                  elements=st.floats(-0.5, 0.5, allow_nan=False)))
def test_drawdown_is_nonpositive(rets):
    eq = (1 + pd.Series(rets)).cumprod()
    assert max_drawdown(eq) <= 1e-12          # ≤ 0 within float tolerance

@given(hnp.arrays(np.float64, st.integers(1, 200),
                  elements=st.floats(-0.9, 9.0, allow_nan=False)))
def test_log_simple_roundtrip(rets):
    assert np.allclose(np.expm1(np.log1p(rets)), rets)
```

Good invariants: drawdown ≤ 0; vol ≥ 0; weights sum to target. Note the annualized Sharpe scales by `sqrt(periods_per_year)` — that is the annualization factor, *not* a property of tiling: duplicating each return k× per period leaves the per-observation `mean/std` essentially unchanged (it does **not** multiply Sharpe by √k), so don't write a "tiling scales Sharpe by √k" test — it's false. The correct invariant: if you change the assumed `periods_per_year` from p to k·p on the *same* return series, the reported annualized Sharpe scales by √k.

### Synthetic-data and NaN/alignment tests

- Generate data with a *known* embedded edge (e.g. a deterministic momentum process); assert the strategy captures it and a shuffled version (edge destroyed) earns ~0.
- Feed a series with leading NaNs, gaps, and a duplicate timestamp; assert the pipeline either handles or raises — never silently returns a wrong-length result.
- Assert output index equals input index (no rows silently dropped), and `result.notna()` where you expect values.

Run tests in CI on every push; treat warnings (SettingWithCopy/ChainedAssignment, dtype, numpy `RuntimeWarning`) as failures (`filterwarnings = error` in pytest config).

---

## 6. Reproducibility and determinism

If you can't reproduce a backtest bit-for-bit, you can't trust a "fixed" bug or an A/B comparison.

- **Seeds:** seed every RNG you use. Prefer the Generator API `rng = np.random.default_rng(cfg.seed)` over global `np.random.seed`; also `random.seed`, and framework seeds (`torch.manual_seed`, plus `torch.use_deterministic_algorithms(True)` and `PYTHONHASHSEED` set in the environment before the interpreter starts). Pass the seed through config so it's recorded.
- **Pinned environments:** lock exact versions (`uv.lock`/`poetry.lock`/`pip-compile` → `requirements.txt` with hashes). Record the Python and BLAS/numpy build — different BLAS can change the last ULP of large linalg and break golden tests at tight tolerances. Containerize (Docker) for live.
- **Deterministic ordering:** dict iteration is insertion-ordered (3.7+) but `set` iteration order and `groupby` over unsorted keys are not stable across runs/platforms unless you sort. Always `sort_index()`/`sort` before reducing if order affects float sums. Sort the universe deterministically.
- **No wall-clock or `datetime.now()` in research code paths** — inject the timestamp. Hidden `now()` makes backtests non-reproducible and can leak look-ahead.
- **Experiment tracking:** log config + git commit hash + data snapshot id + seed + resulting metrics for every run (MLflow, Weights & Biases, or a plain append-only `runs.parquet`). A result you can't tie to a commit and a config is anecdote, not evidence. Hash input data so you detect silent upstream data changes.

---

## 7. Research → production

A backtested signal becomes a live job. The same `signals` function runs in both; everything around it changes.

### Shape of a live runner

```python
def trading_cycle(clock, data_feed, signal_fn, oms, cfg) -> None:
    bar = data_feed.latest_completed_bar()           # COMPLETED bar only
    if bar is None or bar.ts <= oms.last_processed_ts:
        return                                        # idempotent: skip duplicates/partials
    features = build_features(data_feed.history(cfg.lookback))
    target = signal_fn(features, cfg)                 # SAME fn the backtest used
    current = oms.current_positions()
    orders = diff_to_orders(current, target, cfg)     # trade the delta, not the target
    for o in orders:
        oms.submit(o)                                 # broker enforces idempotency key
    oms.persist_state(bar.ts)                          # durable checkpoint
```

### Production concerns (each is a fail mode if skipped)

- **Idempotency:** a cron retry, a restart, or a duplicated message must not double-trade. Process only *completed* bars; key the bar by timestamp; use client order IDs so resubmits are deduped broker-side; make `trading_cycle` safe to call twice for the same bar.
- **Real-time / streaming data:** never act on a forming (partial) bar — it's the live equivalent of same-bar look-ahead. Wait for bar close + a small grace for late ticks. Handle out-of-order and revised ticks. Distinguish "no data yet" from "value is 0."
- **State management:** positions, last-processed timestamp, and any path-dependent indicator state must be durable (DB/file), so a crash recovers to truth, not to a fresh in-memory zero. On startup, reconcile persisted state against the broker's actual positions before trading.
- **Monitoring & alerts:** heartbeat (job ran this cycle), data freshness (feed stalled?), order reject/latency, position vs target drift, realized PnL vs expectation. Alert on *silence* too — a dead job emits nothing.
- **Fail-safes / kill switch:** max position size, max daily loss, max order rate, fat-finger price bands, and a manual halt flag checked every cycle. Default to flat/no-trade on any uncertainty (stale data, failed reconciliation, exception) — losing an opportunity beats an uncontrolled position.
- **Reconcile live vs backtest:** run the backtest over the same dates the strategy traded live, with realized fills/costs, and compare. Persistent gaps point to slippage/cost-model error, fill timing, data revisions, or a look-ahead bug that only the live run exposes. Build this comparison as a scheduled report, not a one-off.

---

## 8. Style

- **Typed:** annotate signatures; run `mypy`/`pyright` in CI. Types document the data contract (`pd.Series` of returns vs prices vs an equity curve are different things — name and type them).
- **Small pure functions:** a signal function takes data + config and returns a signal, with *no* I/O, no global state, no `now()`. Pure functions are trivially testable and reusable in backtest and live alike. Push side effects (reads, writes, order submission) to the edges.
- **Clear interfaces:** stable function signatures and named, validated config at module boundaries. Prefer explicit args over kwargs-dicts. Make illegal states unrepresentable (e.g. an enum for `Side`, not a stray `+1/-1` int with no meaning).
- Run `ruff` (lint + format) and treat warnings as errors. Docstring the *convention* each function assumes (simple vs log returns, periods_per_year, whether positions are pre- or post-shift) — convention mismatches are the silent killers.
