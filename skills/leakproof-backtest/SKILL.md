---
name: leakproof-backtest
description: >-
  Use when asked to "backtest this signal/strategy", "set up / wire up a vectorized backtest",
  "lag positions / avoid same-bar fills", "pnl = position.shift(1)*ret", or "charge costs on
  turnover" — the quick-draw playbook to BUILD ONE leak-free, full-fill, single- or cross-
  sectional vectorized backtest fast. For auditing an EXISTING backtest for leaks use the
  backtest-auditor agent; for event-driven / partial-fill / limit-order realism use the broad
  claude-quant skill.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Stand up a leak-free vectorized backtest in minutes by reusing the plugin skeleton — never hand-roll `pnl = position * return`.

## Do this now
1. Open `skills/claude-quant/templates/backtest_skeleton.py`. Reuse `run_backtest(...)`; do NOT rewrite its lag/turnover wiring.
2. Write a point-in-time `signal_func(prices)` using only data up to bar t (Iron Law 1). For a quick start, `ma_crossover` is provided.
3. Size with `fixed_sizer` or `vol_target_sizer` — the vol sizer already `.shift(1)`s realized vol, so scaling is causal (kills pitfall #23). Do NOT swap in same-bar or full-sample vol.
4. Keep `lag=1` (default): `build_positions` shifts targets so `pnl_t = position_{t-1} * return_t`. `lag=0` is the same-bar bug (#9).
5. Charge costs on turnover (Iron Law 3). Pass a `cost_model`: `fixed_bps_cost(...)` for a quick pass, or wrap realistic frictions (`slippage_total` + `commission_return` from `skills/claude-quant/templates/costs.py`) into a turnover→cost callable. Turnover is `positions.diff().abs()` — already aligned to the execution bar, so do NOT lag it again.
6. Sanity-gate with `breakeven_cost_bps(gross_ann, annual_turnover)` (costs.py): if realistic per-trade cost exceeds it, the edge is a mirage — kill it.
7. OOS: split with `walk_forward_splits(n_obs, train_size, test_size, embargo)` (Iron Law 4); embargo >= label horizon.

## Gotchas that ruin this
- Same-bar fills (#9): run `lag=2` vs `lag=1`; if the edge evaporates, you had look-ahead.
- Contemporaneous/full-sample vol scaling (#23): only the lagged sizer is safe.
- Zero/flat costs (#11–#12): high-turnover "alpha" survives only at 0 bps.
- Multi-asset: with `fwd_ret` already next-bar (research-backtest.md §6), lag the weight OR use fwd returns — never both. Path-dependent sizing/limit fills → go event-driven.

## References
`skills/claude-quant/references/research-backtest.md` (§2 vectorized leakage trap; §6 cross-sectional) and `skills/claude-quant/references/pitfalls.md` (#9, #11, #23).

## Expected output
A runnable backtest returning `{equity, net_returns, positions, metrics{sharpe, ann_return, ann_vol, max_drawdown, avg_turnover, n_periods}}` — net-of-cost and lagged, plus a `lag=2` leakage check and a breakeven-cost verdict.
