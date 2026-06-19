---
name: factor-screen
description: 'Use when asked to "evaluate a factor", "screen a signal", "is this cross-sectional signal any good", "compute IC / rank-IC", "IC decay / half-life", "quantile / decile long-short spread", "factor monotonicity", or "factor t-stat" — the fast single-factor cross-sectional screen (the quick counterpart to the broad claude-quant skill).'
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Fast, leak-free verdict on ONE cross-sectional signal: does it rank the cross-section against FUTURE returns, net of churn? Build on the template; never re-derive the math.

## Do this now
1. **Shape the panel.** Two date x asset frames: `factor_df` (value known as of close of date t) and `fwd_ret_df` where `fwd_ret.loc[t]` is the t+1..t+h return — strictly bars AFTER the signal bar. Orient so higher factor = higher predicted return.
2. **Clean per date.** `winsorize()` then `cross_sectional_zscore()` or `cross_sectional_rank()`. Strip unintended bets: `neutralize(factor_df, sector_dummies, log_mktcap, beta)` (per-date OLS residual).
3. **IC + significance.** `information_coefficient(..., method="spearman")` then `ic_summary()` for `mean_ic`, `ic_ir`, `t_stat`, `hit_rate`. Rank-IC is the headline; add Pearson only if you'll size proportionally.
4. **IC decay.** Loop step 3 over h in {1,2,3,5,10,21,63} (reference §5 `ic_decay` helper) to find the half-life and set rebalance frequency near it.
5. **Quantile spread + churn.** `quantile_returns(q=5)` -> `quantile_spread_summary()` for `spread_sharpe` (gross) and `monotonic`. Compute `turnover()` (reference §6).

## References
- Template: `skills/claude-quant/templates/factor_research.py`
- `skills/claude-quant/references/factor-research.md` (decay §5, quantiles/turnover §6, pitfalls §10)
- Costing/deflation: `references/research-backtest.md`, `references/stats-risk.md`

## Gotchas (Iron Laws 1, 2, 4, 5)
- Exclude the signal bar: `ret.shift(-1)` THEN take the h-bar forward window — `rolling(h).prod().shift(-h)` anchored at t still includes bar t (leakage §10).
- Neutralize/standardize PER DATE only — pooled fits leak the future.
- `t_stat` is iid-naive; overlapping fwd returns (h>1) inflate it — discount, or use non-overlapping/Newey-West.
- One factor = one trial. Log every variant into `n_trials`; deflate. Mean rank-IC >= 0.10 on a raw factor -> audit for leakage.

## Expected output
mean_ic, ic_ir, t_stat, hit_rate; decay curve + half-life; spread_sharpe + monotonic flag; two-sided turnover; trial count. A pass earns a costed quantile backtest, not a position.
