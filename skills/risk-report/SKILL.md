---
name: risk-report
description: >-
  Use when asked to "produce a risk report", "compute VaR / Expected Shortfall (ES)", "VaR
  backtest", "Kupiec / Christoffersen", "is my VaR calibrated", "component / marginal risk
  contributions", "risk attribution / budget", "stress test the book", "scenario / reverse
  stress", or "check risk limits" — the quick path to ONE defensible portfolio risk report right
  now (parametric + historical + Monte-Carlo VaR/ES, risk attribution, scenario stress, limit
  table, coverage backtest). For ongoing limit governance, breach handling, kill-switches and de-
  risking decisions use the risk-manager agent; for the full lifecycle use the broad claude-quant
  router.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

One job: produce a defensible portfolio risk report and prove the VaR is calibrated.

## Do this now
1. Read `skills/claude-quant/templates/risk.py`, `templates/metrics.py`, and `templates/portfolio.py`. Pin the sign convention: returns/VaR/ES are signed, loss is NEGATIVE; `level` = left-tail prob (0.01 → 99% VaR), not confidence. Assert it at every boundary.
2. VaR three ways at each level (0.05, 0.01): historical + Gaussian via `metrics.value_at_risk` / `conditional_value_at_risk` (`method=`); fat-tail via `risk.cornish_fisher_var`; Monte-Carlo via the `monte_carlo_var` recipe in `references/risk-management.md` (Student-t needs `df>2` and the `sqrt((df-2)/df)` covariance rescale). Report ES beside every VaR via `risk.expected_shortfall`.
3. Risk attribution: `portfolio.risk_contributions(w, cov)` returns NORMALIZED component contributions (sum to 1); flag the name/factor eating the budget.
4. Stress: `risk.stress_pnl` / `risk.stress_grid` with a named library (2008, 2020, 2022 rates, vol spike) plus the reverse-stress scan from the reference. Options books: full-revalue per scenario, never dot deltas.
5. Limit checks: `templates/pretrade_checks.py::check_order` gates ORDER-level notional/gross/collar/participation only. ES / sector / factor limits aren't templated — compute them and compare to your thresholds manually.
6. Backtest VaR: roll OUT-OF-SAMPLE LAGGED forecasts (var_t from data ≤ t−1), then `risk.count_exceptions` → `risk.kupiec_pof` (coverage) AND `risk.christoffersen_cc(exceptions, level)` (independence + conditional). Add the Basel traffic-light from the reference.

## Gotchas
- Gaussian VaR understates fat tails — never size capital on it; budget on ES.
- ES is always at least as severe as VaR (more negative): assert `es <= var`.
- In-sample VaR coverage is circular (Iron Law 4); lag forecasts (Iron Law 1). An unbacktested VaR is worthless.
- Kupiec passing does NOT clear clustering — run Christoffersen too.

## Output
VaR/ES × method × level table; component-risk table; stress + reverse-stress grid; limit breaches; backtest verdict (Kupiec/Christoffersen p-values + traffic-light). Cross-link `references/risk-management.md`.
