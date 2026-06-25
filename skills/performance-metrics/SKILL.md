---
name: performance-metrics
description: >-
  Use when asked to "compute/report performance metrics", "Sharpe ratio", "annualized return/vol",
  "Sortino", "Calmar", "max drawdown", "hit rate", "profit factor", "information ratio", or to add
  a "Probabilistic/Deflated Sharpe" line to a metrics block after a parameter search — the quick
  metrics-only playbook that turns a return series into a correct, honest metrics table. Boundary:
  for the ADVERSARIAL verdict on whether an edge is overfit (permutation tests, stationary
  bootstrap on the equity curve, parameter-plateau-vs-spike, multiple-testing investigation,
  honest Sharpe CIs) use the overfitting-detective agent; this skill just reports the numbers
  (including DSR) and does not run the investigation. The broad claude-quant skill covers the full
  lifecycle.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

One job: produce a correct, honest metrics block from a return series — and never trust a Sharpe until it is deflated for the trial count.

## Do this now
1. Reuse `skills/claude-quant/templates/metrics.py` — import it; do NOT re-derive. It enforces every convention below.
2. Inputs are per-period SIMPLE returns; NaNs are dropped (never filled to 0). Set `periods_per_year` (ppy) explicitly: 252 daily, 52 weekly, 12 monthly, 365 crypto/24-7. Pass an ANNUAL `risk_free`; the template converts it per-period.
3. Core block: `annualized_return` (geometric), `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `hit_rate`, `profit_factor`; add `information_ratio(returns, benchmark, ppy)` if benchmarked.
4. Significance (Lo 2002): report `sharpe_tstat` and `sharpe_se` — both use the `(1+0.5·SR²)` correction, NOT `SR·√n`. If returns are autocorrelated, replace the naive √ppy scaling with `lo_annualization_factor(returns, q=ppy)·per_period_SR` (positive autocorr ⇒ factor < √ppy ⇒ lower honest Sharpe).
5. Honesty gate: report `probabilistic_sharpe_ratio` (skew/kurtosis-adjusted, vs a PER-PERIOD benchmark SR). If ANY config/parameter search happened, report `deflated_sharpe_ratio(returns, n_trials, trial_sharpe_std)` where `trial_sharpe_std` = std of PER-PERIOD Sharpes across trials. DSR < 0.95 ⇒ the winner is plausibly luck.
6. Validate: `python skills/claude-quant/templates/metrics.py` — all self-tests must pass.

## Gotchas that ruin this
- User-facing vol/Sharpe use `ddof=1`; Sortino's downside denominator divides by full n (RMS of min(excess,0)) — by design, do not "fix" it.
- Mixing ppy 252 vs 365 silently mis-scales Sharpe ~20%.
- PSR/DSR `benchmark_sr` is PER-PERIOD; never feed it an annualized SR.
- Count `n_trials` HONESTLY — include abandoned variants, not just the final grid.

## Reference
Formulas, MinBTL, bootstrap CIs: `skills/claude-quant/references/stats-risk.md` (§1.3–1.6, §3, §4.1).

## Expected output
Metrics table + Lo t-stat/SE + PSR and (if searched) DSR, stating ppy, rf, and n_trials. A bare Sharpe is not acceptable.
