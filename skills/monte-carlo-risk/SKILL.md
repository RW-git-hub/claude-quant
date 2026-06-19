---
name: monte-carlo-risk
description: 'Use when asked to "Monte-Carlo a strategy", "simulate equity paths", "bootstrap the returns", "distribution of drawdown / terminal wealth", "risk of ruin", "how bad could the drawdown get", or "stress-test sizing with simulation" — the quick resampling-stress playbook (the broad claude-quant skill is the full robustness lab).'
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Turn one realized backtest path into a distribution of outcomes via dependence-preserving resampling, then read off drawdown, terminal wealth, and ruin risk for a sizing rule.

## Do this now
1. Coerce the realized per-period returns to a finite 1-D NumPy array (`np.asarray(series)` first — pandas index alignment silently un-shuffles a resample).
2. Resample with the stationary bootstrap, never i.i.d.: `templates/robustness.py:stationary_bootstrap_indices(n, mean_block, n_samples, seed)`. Default `mean_block ~ n**(1/3)` (the template's rule of thumb); raise it toward the PnL autocorrelation horizon for trend/momentum. One index matrix resamples returns/signals/pnl consistently.
3. Build Monte-Carlo equity paths: index the returns by each row, then `metrics.py:equity_curve` per path. You now have `n_samples` alternate histories.
4. Per path compute max drawdown (`metrics.py:max_drawdown`, returned negative — take the 5th percentile for the worst case), terminal wealth, and Sharpe; report percentiles (5/50/95), not the mean. For a direct Sharpe CI: `robustness.py:bootstrap_sharpe_ci(returns, mean_block=...)`.
5. Risk-of-ruin: `templates/risk.py:risk_of_ruin(win_prob, win_loss_ratio, bet_fraction, ruin_threshold=...)` — fraction of MC paths breaching the floor. Size via `risk.py:kelly_fraction` (use a fraction of f*), sweep `bet_fraction` and confirm ruin is monotone increasing. Cross-stress with `risk.py:stress_grid`.

## Gotchas
- I.i.d. bootstrap destroys autocorrelation and vol clustering → drawdown CIs absurdly tight. Use stationary/block.
- Drawdown and ruin are path-dependent: resample order WITHIN each path; never shuffle one concatenated series.
- Bootstrap the OOS path, not the IS-optimized one — it quantifies sampling uncertainty, it cannot undo overfitting (Iron Laws 4-5).
- Fix and report every seed.

## Expected output
Percentile table (MDD / terminal wealth / Sharpe over `n_samples` paths), a ruin probability per `bet_fraction`, and the seed + `mean_block` used. See `references/robustness.md` for resampling and synthetic-stress detail.
