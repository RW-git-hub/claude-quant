---
name: vol-forecast
description: >-
  Use when asked to "forecast volatility", "predict next-period / h-ahead vol", "EWMA / GARCH(1,1)
  / HAR-RV vol forecast", "realized-vol forecast", or "the causal vol estimate that feeds vol
  targeting / inverse-vol scaling" — the quick single-job playbook that produces ONE causal
  h-ahead vol forecast and a clipped inverse-vol position scaler. For turning an existing vol
  estimate into Kelly/risk-parity/ERC weights or choosing among sizing regimes use position-
  sizing; for model selection across GARCH variants, HMM regimes, change-points, or formal
  QLIKE/DM/MCS evaluation use the regime-detector agent; for full backtest/risk suites use the
  broad claude-quant skill.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

One job: estimate vol on data up to t, forecast h-ahead, turn it into an inverse-vol position scaler. Causal throughout.

## Do this now
1. Open `skills/claude-quant/templates/regime.py`. Read the Conventions block (lines 8-18): functions return PER-PERIOD vol and are causal but do NOT lag for you.
2. Pick an estimator, fit on returns up to t ONLY:
   - EWMA (RiskMetrics): `ewma_vol(returns, lam=0.94)` (~0.97 monthly). Note its recursion uses `r[t-1]^2`, so the value at t already is the one-step-ahead forecast — no fit needed.
   - GARCH(1,1): `garch11_fit(returns)` then `garch11_filter(returns, **p)` for the conditional path. For h>1, iterate the variance recursion forward; it mean-reverts toward `omega/(1-alpha-beta)`. There is NO forecast helper — compute it yourself.
   - HAR-RV: `har_rv(rv, horizon=h)` in regime.py — strictly trailing, one-step-ahead (first 22 entries NaN). Build RV from intraday data with `realized_variance(intraday_returns)`.
3. Annualize: per-period vol × `sqrt(ppy)` (252/52/12). For an h-period horizon, variance adds, so vol scales by `sqrt(h)`.
4. Size: `vol_target_scale(forecast_vol, target_vol, max_leverage)` in regime.py — scale = target/forecast, clipped. Keep forecast and target on the SAME units/horizon.
5. Lag: `pnl_t = pos.shift(1) * ret_t` (Iron Law 1). Check cost-adjusted Sharpe per `references/stats-risk.md`.

## Reference
- `skills/claude-quant/references/time-series-regimes.md` — estimator choice, persistence, refit cadence.
- `skills/claude-quant/templates/risk.py` — `stress_grid` / `expected_shortfall` on the sized book.

## Gotchas
- Causal only: no full-sample GARCH fit, no centered window. Refit on an expanding/rolling window in backtests (Iron Laws 1, 4).
- Unit/horizon mismatch in `vol_target_scale` (daily forecast vs annual target) silently mis-sizes by ~sqrt(252).
- Vol clusters and mean-reverts — let GARCH revert toward its unconditional vol over h; don't hold today's spike flat.
- Persistence ≥ 1 from `garch11_fit` means a break/outliers — winsorize and refit.

## Expected output
A causal per-period vol Series, an h-ahead (annualized) forecast, a clipped `vol_scale` Series, and a lagged, cost-aware PnL/Sharpe check.
