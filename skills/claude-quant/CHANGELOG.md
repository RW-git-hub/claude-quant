# Changelog

## 2.0.0 — 2026-06-19

Expanded `claude-quant` from a 5-reference / 4-template core into a comprehensive,
execution-verified quant skill via **17 research → implement → verify cycles**.
Each cycle ran an Opus author + adversarial-verify workflow to draft a reference
and a template, after which every template was verified by **actually running its
self-tests** (final state: 17/17 templates pass under `templates/run_all_tests.py`).

### Added — references
`factor-research`, `transaction-costs`, `ml-for-alpha`, `derivatives`,
`live-trading`, `prediction-sports-markets`, `stat-arb`, `portfolio-optimization`,
`microstructure`, `time-series-regimes`, `robustness`, `crypto-defi`,
`risk-management`, `playbooks`.

### Added — templates (all self-testing, numpy/pandas)
`factor_research`, `costs`, `labeling`, `options`, `pretrade_checks`,
`betting_markets`, `pairs_trading`, `portfolio`, `execution`, `regime`,
`robustness`, `crypto_defi`, `risk`, plus `run_all_tests.py` and
`examples/end_to_end.py`.

### Fixed — correctness issues caught by execution (not review)
- **metrics.py** — corrected the Sharpe t-stat vs standard error (Lo 2002; the
  `t = SR·√n` shortcut drops the `1 + ½SR²` term); guarded `lo_annualization_factor`
  against the `q≈n` NaN; collapsed `sortino` to a single hurdle; added
  `sharpe_tstat` / `sharpe_se`.
- **validation.py** — `PurgedKFold` purges `label_horizon` on **both** sides of each
  test block; ERC risk parity uses the **sqrt-damped** update (equalizes *total*,
  not marginal, risk contributions — the naive update returned inverse-variance).
- **regime.py** — CUSUM references the **in-control segment mean** (not the
  full-sample mean, which fires at t=0); `vol_target_scale` caps at zero vol.
- **stats-risk.md** — Ledoit-Wolf attributed by paper title (constant-correlation vs
  scaled-identity), Sharpe SE/t-stat clarified, `lo_annualization_factor` guarded.

### Meta
SKILL.md router now covers all 19 references and groups all 17 templates; stale
"companion template not yet shipped" notes removed from references.

## 1.0.0 — 2026-06-19

Initial release: `SKILL.md` (Iron Laws, job router, canonical conventions) +
references (`data`, `research-backtest`, `quant-dev`, `stats-risk`, `pitfalls`) +
templates (`metrics`, `validation`, `backtest_skeleton`, `data_loader`). All
reference content authored and adversarially fact-checked; templates execution-verified.
