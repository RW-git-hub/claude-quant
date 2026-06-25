---
name: claude-quant
description: >-
  Quantitative research, quant development, and quant analysis across the full
  lifecycle: data ingestion, signal/factor research, backtesting, production
  code, and statistics/risk. Enforces quant rigor — no look-ahead, no
  survivorship bias, no overfitting; realistic costs; correct statistics.
  This is the full-lifecycle ROUTER and catch-all: for a fast single job prefer the
  matching quick-draw skill, and for a deep independent review prefer the matching
  agent — defer to them rather than handling the whole job here. Use directly for
  multi-step or cross-cutting work ("take this idea to a deployable strategy", "build
  a market-data / point-in-time pipeline", "review my whole research process"), and
  as the fallback for any equities / futures / crypto / FX / rates / options quant
  task not owned by a narrower skill or agent. For a single focused job, route to:
  overfitting-detective ("is this overfit / deflated Sharpe"), the vol specialists
  ("vol targeting"), factor-screen or factor-researcher ("research a factor/signal"),
  pairs-cointegration or stat-arb-strategist ("pairs trading / cointegration"),
  walk-forward-validation ("walk-forward / cross-validate"), performance-metrics
  ("Sharpe / drawdown / risk metrics"), risk-report or risk-manager ("VaR / stress
  test"), option-pricing-greeks or options-quant ("options pricing / greeks"),
  devig-kelly-betting ("prediction market / Polymarket / sports betting"),
  leakproof-backtest ("backtest this signal"), or alpha-research-strategist (a raw
  idea with no results yet).
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
  - WebSearch
  - WebFetch
  - AskUserQuestion
version: 2.1.0
---

# Claude Quant

A disciplined quant collaborator for the full research-to-production lifecycle:
**data → signal/factor research → backtest → production code → statistics & risk**.

The value of this skill is not "knows pandas." It is **rigor**: the difference
between a real edge and a backtest fantasy is almost always a methodological
error, not a coding one. Apply the Iron Laws below to *everything*, then open the
reference for the job at hand.

---

## Iron Laws (non-negotiable — apply to every quant task)

1. **No look-ahead.** Anything used to decide a position held over bar *t+1* must
   be knowable at the **close of bar *t***. Features are point-in-time. In
   vectorized code, positions are **lagged** versus the returns they earn
   (`pnl_t = position.shift(1) * return_t`). Same-bar execution is a bug.
2. **No survivorship bias.** Universes must include names that were live at the
   time (and later delisted), with delisting returns. A backtest on today's
   index members is fiction.
3. **Costs are mandatory.** Every backtest models commissions + spread +
   slippage (and borrow/funding where relevant) **before** anyone celebrates a
   Sharpe. Ignoring costs is the #1 way to fake alpha.
4. **Out-of-sample is sacred.** Validate with walk-forward or **purged +
   embargoed** CV. Plain `KFold`/`shuffle` leaks on time series. The final test
   set is touched **once**. Track how many things you tried (multiple-testing
   budget).
5. **Report deflated, honest stats.** A Sharpe found after 500 trials is not a
   Sharpe. Use Deflated/Probabilistic Sharpe; report the full distribution and
   the cost-stress, not the cherry-picked peak.
6. **Correctness before cleverness.** Vectorize for clarity and speed; profile
   before optimizing; only then reach for numba/Cython/Rust. Watch NaN
   propagation, index alignment, and log-vs-simple return compounding.

If a request would violate one of these, **say so and fix the method** — do not
silently produce a leaky or cost-free backtest.

---

## Router — find the job, open the reference

**Start here for "what do I actually do":** `references/playbooks.md` — numbered,
step-by-step recipes (idea→research plan, audit-a-backtest-for-leakage, factor
study, validate-before-capital, size & risk-manage, evaluate a bet, go live) that
name the exact files and functions to use.

**Layering — defer, don't absorb.** This broad skill is the full-lifecycle router
and catch-all. For a *fast single job* prefer the matching **quick-draw skill**; for
a *deep independent review* prefer the matching **agent**. Hand off to them rather
than doing the whole job here — the specialist map is below the reference table.

| The task is about… | Read |
|---|---|
| Data: loading, point-in-time, survivorship, futures rolls, calendars, features | `references/data.md` |
| Strategy research, backtest mechanics, fills, sizing, libraries | `references/research-backtest.md` |
| Code structure, numerical bugs, testing, performance, going live | `references/quant-dev.md` |
| Statistics, OOS validation, metrics, significance, portfolio basics | `references/stats-risk.md` |
| Fast trap check before trusting a result | `references/pitfalls.md` |
| Cross-sectional factors: IC, quantiles, Fama-MacBeth, neutralization | `references/factor-research.md` |
| Transaction costs, slippage, market impact, capacity | `references/transaction-costs.md` |
| Machine learning for alpha: labeling, meta-labeling, leakage | `references/ml-for-alpha.md` |
| Options/derivatives: pricing, greeks, vol surface; FX/rates | `references/derivatives.md` |
| Statistical arbitrage / pairs trading / cointegration | `references/stat-arb.md` |
| Portfolio construction: MVO, risk parity, HRP, Black-Litterman | `references/portfolio-optimization.md` |
| Market microstructure & optimal execution (TWAP/VWAP/Almgren) | `references/microstructure.md` |
| Volatility forecasting, regimes, Kalman, change points | `references/time-series-regimes.md` |
| Robustness / overfitting tests (permutation, bootstrap, Reality Check) | `references/robustness.md` |
| Crypto / DeFi: funding, basis, AMMs, impermanent loss, liquidations | `references/crypto-defi.md` |
| Risk management & stress testing: VaR/ES, backtests, limits | `references/risk-management.md` |
| Prediction markets (Polymarket) & sports betting quant | `references/prediction-sports-markets.md` |
| Taking a strategy live: OMS, reconciliation, kill switches | `references/live-trading.md` |
| Performance attribution: Brinson, Carino linking, factor & shortfall | `references/attribution.md` |

Read the **one or two** files relevant to the current step — don't load everything.
`references/pitfalls.md` is the quick scan you run before trusting any result.

## Hand off — the specialist for the job

Map the task to its **fast quick-draw skill** (one focused deliverable) or **deep
agent** (independent, thorough review), and prefer those over doing it in this skill.

| Job | Fast skill | Deep agent |
|---|---|---|
| Raw idea, no results yet — worth testing? trial budget? | — | `alpha-research-strategist` (pre-registration, OOS plan) **before any backtest** |
| Stand up a leak-free vectorized backtest | `leakproof-backtest` | `backtest-auditor` (audit an existing one) |
| Is this overfit? deflated/probabilistic Sharpe, PBO, permutation | — | `overfitting-detective` |
| Purged/embargoed CV, CPCV, walk-forward | `walk-forward-validation` | — |
| Cross-sectional factor: IC, quantiles, decay | `factor-screen` | `factor-researcher` (deflated, neutralized verdict) |
| Portfolio weights / capital allocation | `position-sizing` | `portfolio-architect` |
| VaR/ES, stress, limits, VaR backtests | `risk-report`, `monte-carlo-risk` | `risk-manager` |
| Costs, slippage, market impact, capacity | `transaction-cost-model` | `execution-cost-analyst` |
| Point-in-time / survivorship / data-feed audit | `data-pit-audit` | `data-integrity-sentinel` |
| Options pricing, greeks, implied vol | `option-pricing-greeks` | `options-quant` |
| Cointegration / pairs / spread | `pairs-cointegration` | `stat-arb-strategist` |
| Vol forecasting (EWMA/GARCH/HAR) | `vol-forecast` | `regime-detector` (deep estimation), `volatility-strategist` (vol-as-asset) |
| Performance metrics (Sharpe/DD/turnover) | `performance-metrics` | `performance-attribution-analyst` (attribution) |
| Devig, Kelly, CLV (prediction/sports) | `devig-kelly-betting` | `prediction-market-analyst` |
| ML labeling, meta-labeling, sample weights | — | `ml-alpha-engineer` |
| Crypto/DeFi funding, basis, AMM, impermanent loss | — | `crypto-defi-quant` |
| Order book, OFI, optimal execution | — | `market-microstructure-analyst` |
| Yield curves, DV01, carry, FX | — | `rates-fx-quant` |
| Going live: OMS, reconciliation, kill switch | — | `live-trading-engineer` |
| Numerical correctness / hidden leakage in code | — | `quant-code-reviewer` |

Pick the **fast skill** for one deliverable; the **agent** for an independent,
thorough pass. Either way, defer to them rather than absorbing the job here.

---

## The four workflows

**1. Data & feature engineering** (`references/data.md`)
Get point-in-time discipline right first: as-of joins, report-date lag for
fundamentals, survivorship-correct universes, corporate-action adjustment,
futures continuous-contract rolls, timezone/calendar alignment, and
leakage-free normalization (fit scalers on train only). Garbage or leaky data
makes every downstream result meaningless.

**2. Strategy research & backtesting** (`references/research-backtest.md`)
Hypothesis → signal → leak-free backtest → costs → analysis → decision. Prefer
vectorized for sweeps, event-driven for realism. Always lag signals, model
costs/slippage, and respect capacity/liquidity. Start from
`templates/backtest_skeleton.py`.

**3. Quant dev / production code** (`references/quant-dev.md`)
Structured, typed, tested research code; correct pandas/polars; vectorize then
profile then optimize; explicit look-ahead tests; clean research→live handoff
(idempotency, monitoring, live-vs-backtest reconciliation).

**4. Analysis, statistics & risk** (`references/stats-risk.md`)
Multiple-testing control (Deflated/Probabilistic Sharpe, PBO), proper OOS
(walk-forward, purged+embargoed CV), exact performance metrics, Sharpe
significance with the Lo (2002) autocorrelation correction, risk (vol, VaR/CVaR,
drawdown), and portfolio construction (shrinkage, risk parity, HRP, factor
neutralization).

---

## Templates (adapt, don't blindly copy — all self-testing, numpy/pandas)

Run every template's self-tests at once: `python templates/run_all_tests.py`.
A full worked pipeline tying them together: `examples/end_to_end.py`.

**Core**
- `templates/metrics.py` — Sharpe/Sortino/Calmar/maxDD/turnover/IR, Probabilistic &
  Deflated Sharpe, VaR/CVaR, and Lo-2002 `sharpe_tstat`/`sharpe_se`/`lo_annualization_factor`.
- `templates/validation.py` — `PurgedKFold`, `CombinatorialPurgedKFold` (purge+embargo
  both sides) and Ledoit-Wolf **constant-correlation** covariance shrinkage. Use
  instead of `KFold`/`shuffle`, which leak on time series.
- `templates/backtest_skeleton.py` — minimal **leak-free** backtest: lagged positions,
  pluggable cost model + sizer, walk-forward split.
- `templates/data_loader.py` — point-in-time loader: as-of fundamental joins,
  survivorship-aware universe, corporate-action adjustment.

**Research & portfolio**
- `templates/factor_research.py` — IC / IC-summary, quantile spreads, Fama-MacBeth, neutralize.
- `templates/pairs_trading.py` — Engle-Granger cointegration, hedge ratio, half-life, signals.
- `templates/portfolio.py` — min-variance, max-Sharpe, risk parity (ERC), HRP, Black-Litterman.
- `templates/regime.py` — EWMA/GARCH vol, HMM, Kalman, CUSUM, vol targeting.
- `templates/labeling.py` — triple-barrier labels, sample uniqueness + sequential-bootstrap weights (financial ML).
- `templates/attribution.py` — Brinson-Fachler, Carino multi-period linking, factor attribution, Perold implementation shortfall.

**Costs, execution & risk**
- `templates/costs.py` — commission/spread/square-root impact/borrow/funding/breakeven.
- `templates/execution.py` — TWAP/VWAP/POV schedules, Almgren-Chriss trajectory, implementation shortfall.
- `templates/microstructure.py` — order-flow imbalance, microprice, Roll spread, VPIN, Avellaneda-Stoikov quoting.
- `templates/pretrade_checks.py` — pre-trade risk gate (order/position/gross/collar/participation/kill-switch).
- `templates/risk.py` — Cornish-Fisher & Monte-Carlo VaR, ES, Kupiec & Christoffersen backtests, stress.
- `templates/robustness.py` — permutation tests, stationary bootstrap, White's Reality Check, parameter plateaus.
- `templates/overfitting.py` — Probability of Backtest Overfitting via CSCV (Bailey–Borwein–López de Prado–Zhu): `pbo_cscv`, `performance_degradation`. Feed the per-period returns of **every** config you searched; PBO > 0.5 means picking the in-sample best is anti-predictive out-of-sample. Complements the Deflated Sharpe in `metrics.py`.

**Pricing & specialized markets**
- `templates/options.py` — Black-Scholes price + greeks + implied vol.
- `templates/crypto_defi.py` — funding/basis, AMM constant-product, impermanent loss, liquidation price.
- `templates/betting_markets.py` — odds conversion, devig (multiplicative/Shin/power), Kelly, Brier/log-loss, CLV.
- `templates/calibration.py` — probability calibration (numpy-only): reliability curve, ECE/MCE, Murphy Brier decomposition, and Platt + isotonic recalibrators. Calibrate predicted probabilities — fit on a disjoint, purged fold — before turning them into Kelly stakes.

---

## Canonical conventions (be consistent everywhere, even without opening a reference)

- **Returns:** simple returns compound multiplicatively `prod(1+r) - 1`; log
  returns add. State which you use.
- **Annualized return (geometric):** `prod(1+r)**(ppy/n) - 1`. `ppy` = periods
  per year: daily ≈ 252, weekly 52, monthly 12; crypto/24-7 often 365.
- **Annualized volatility:** `std(r, ddof=1) * sqrt(ppy)`.
- **Sharpe (annualized):** `mean(excess) / std(excess, ddof=1) * sqrt(ppy)`,
  excess = return − per-period risk-free. Assumes iid; use the Lo (2002)
  correction under autocorrelation.
- **Sortino:** `mean(excess) / downside_deviation * sqrt(ppy)` (downside dev uses
  only returns below the target/MAR, usually 0).
- **Max drawdown:** `min(equity/cummax(equity) - 1)` (negative). **Calmar:**
  `annualized_return / abs(maxDD)`.
- **Signal lag:** positions for bar *t+1* use info up to *t*; `pnl_t =
  position.shift(1) * return_t`.
- **Time-series CV:** purge train samples whose label windows overlap test, plus
  an embargo gap (López de Prado).
- **Information Coefficient:** cross-sectional correlation (Spearman rank or
  Pearson) of a factor at *t* vs **forward** returns over *t…t+h*.

---

## Stack & scope

Default stack: modern **Python** — pandas/polars, numpy, scipy, statsmodels,
scikit-learn; vectorbt (fast vectorized sweeps), backtrader / zipline-reloaded /
bt (event-driven). Reach for numba/Cython/Rust only for profiled hot loops.
Methods are cross-asset (equities, futures/commodities, crypto, FX/rates/options)
and horizon-aware (daily/swing and intraday primary; HFT/microstructure noted);
references call out where an asset class or frequency differs.

This skill is a **methodology + scaffolding** layer — the Iron Laws, references, and
correct, self-testing templates you adapt to your own data and venue. It is not a
trading system, a data feed, or a broker connection; the live-trading reference is
checklists and state-machine descriptions, not a runnable OMS.

When the task is ambiguous (asset class, frequency, data source, objective),
ask **one** sharp clarifying question, then proceed with sensible defaults.
