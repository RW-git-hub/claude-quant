# Changelog

## 2.1.0 — 2026-06-25

Major capability + routing-validation release, from two more multi-agent workflows (a 13-feature
build pass and a routing-eval + body-consistency pass).

### Added — templates (numpy/pandas-only; all self-testing)
- **`attribution.py`** (new) — Brinson-Fachler allocation/selection/interaction, Carino (1999)
  multi-period linking (with the guarded 0/0 limit), factor attribution, Perold implementation
  shortfall. Backed by the new `references/attribution.md`.
- **`microstructure.py`** (new) — order-flow imbalance, microprice, Roll spread, VPIN,
  Avellaneda-Stoikov quoting.
- `robustness.py` — Hansen (2005) **SPA test** (`spa_pvalue`, studentized + recentered).
- `factor_research.py` — HAC/Newey-West IC t-stat (`ic_summary_hac`), `ic_decay`/`ic_half_life`,
  cap-weighted quantile spread, two-sided `turnover`.
- `regime.py` — multi-step `garch11_forecast`/`gjr_garch11_forecast` (correctly summed h-period
  variance, not √h) and `qlike_loss`/`mse_loss`/`diebold_mariano`.
- `risk.py` — Acerbi-Szekely (2014) ES backtest (`acerbi_szekely_z2`).
- `portfolio.py` — turnover-band / no-trade-region / shrink-to-prev cost-aware rebalancing.
- `labeling.py` — sample-weight primitives (`num_concurrent_events`, `time_decay_weights`,
  `sequential_bootstrap`, `trend_scanning_labels`).
- `options.py` — variance-swap fair strike (DDKZ log-contract strip) + capped var-swap P&L.
- `betting_markets.py` — correlated `joint_kelly`; `crypto_defi.py` — net cash-and-carry + LVR;
  `costs.py` — `cost_sweep`/`capacity_curve` + `day_count`; `pretrade_checks.py` —
  duplicate-clOrdID / self-cross rejection + per-symbol gross marks.

Template count 19 → **21**, references 19 → **20**. **21/21 self-tests pass.**

### Changed — routing (empirically validated)
- A 39-prompt blind routing-regression eval scored **39/39 (100%)**, including the adversarial
  collision cases the 2.0.3 rewrites targeted.
- A body-vs-description consistency pass fixed **9** agent/skill bodies that still claimed a
  capability their description had ceded — e.g. `backtest-auditor` now defers the deflated-Sharpe
  verdict to `overfitting-detective`; `volatility-strategist` consumes rather than produces the vol
  forecast; `vol-forecast`/`overfitting-detective` cite the new `garch11_forecast`/`spa_pvalue`;
  `options-quant`'s inventory was corrected to list the greeks/CRR/var-swap it actually ships.

## 2.0.3 — 2026-06-25

Routing/activation overhaul + correctness fixes, from a 48-agent audit-and-research workflow
(20 routing auditors + 20 domain researchers, adversarially synthesized).

### Changed — auto-routing (so a prompt reaches the right skill/agent)
- **The broad SKILL.md is now a router that defers, not an interceptor.** Its description reframes
  the contested triggers ("is this overfit", "vol targeting", "research a factor/signal", "pairs
  trading", "walk-forward") as *route-to-specialist* instead of claiming them, adds a "defer to the
  quick-draw skill / agent" layering line, and the Router gains a **"Hand off — the specialist for
  the job"** table mapping each job → fast skill + deep agent (plus a front-of-funnel row to
  `alpha-research-strategist`).
- **31 collision-checked description rewrites** across every agent and quick-draw skill, resolving
  real overlaps (e.g. the `backtest-auditor` ↔ `overfitting-detective` ↔ `quant-code-reviewer` ↔
  `data-integrity-sentinel` cluster) and gaps (`curve-fit` / `data-mined` / `PBO` / `CSCV` now route
  to `overfitting-detective`). Agent/skill NAMES are unchanged. `de-gross` was kept solely on
  `risk-manager` (removed from `portfolio-architect`) per adversarial verification.

### Fixed — correctness (Iron-Law bugs surfaced by the research pass)
- `data_loader.pit_join` exposes `allow_exact_matches` (was hardcoded `True` → same-bar look-ahead);
  default preserves behavior, `False` gives strict same-bar exclusion (+ self-test).
- `pairs_trading.kalman_hedge_ratio` docstring + the `pairs-cointegration` skill now require lagging
  the filtered beta (`.shift(1)`) before forming the tradable spread — the unlagged residual is
  look-ahead.
- `validation.summarize_search` passes the **float** effective-N to the Deflated Sharpe (was
  `int(round(...))`, anti-conservative; could zero out deflation near n_eff≈1.5).
- `backtest_skeleton.vol_target_sizer` gains a `vol_floor` bounding the target/realized ratio in a
  near-zero-vol window (+ self-test); default off.

### Fixed — documentation
- Corrected a dimensionally-wrong vol-swap convexity formula (`volatility-strategist`); fixed stale
  line refs + a false "no HAR-RV template" claim (`vol-forecast`); pointed recalibration at the
  shipped numpy `calibration.py` instead of sklearn (`devig-kelly-betting`); noted `risk_contributions`
  ships in `portfolio.py` (`risk-manager`); added a point-in-time ADV/liquidity note (`data.md`) and
  an intrabar stop/target-fill entry to the pitfalls catalog.

All 19 template self-tests pass.

## 2.0.2 — 2026-06-23

Polish pass on the flagship example and credibility signals.

### Changed
- **`examples/end_to_end.py` now prints realistic numbers.** The planted signal was far
  too strong (IC ≈ 0.42, gross Sharpe ≈ 40) — fantasy-tier output that undercut the
  skill's whole "real edge vs. backtest fantasy" thesis (a skeptical reader sees Sharpe 40
  and assumes leakage). The synthetic factor now plants a realistic IC (~0.03, t ≈ 3.8),
  yielding gross Sharpe ≈ 2.9 / net ≈ 1.7 after costs, with costs visibly biting. The run
  prints an explicit caveat that this is synthetic plumbing on a planted signal — **not a
  performance claim** — and that even that Sharpe is optimistic (iid cross-section, no
  return correlation / IC decay / capacity). A self-check now guards against re-inflating
  the demo Sharpe (`net Sharpe < 5`).

### Added
- CI status badge on the top-level README, linking to the GitHub Actions `tests` workflow.

## 2.0.1 — 2026-06-22

Documentation-accuracy and packaging pass. No template logic changed; every
self-test still runs green (**19/19** under `templates/run_all_tests.py`, verified
on numpy 1.26 / pandas 2.2 and numpy 2.x / pandas 3.0).

### Fixed — documentation accuracy
- **Template count corrected to 19.** The README and `marketplace.json` said
  "17 templates," but `templates/` ships 19, so both were corrected. (The 2.0.0 entry
  below is left unedited as a historical record — 2.0.0 genuinely shipped 17 templates;
  `calibration.py` and `overfitting.py` were added afterward, in the quick-draw-skills
  release.) Those two had shipped but were absent from the SKILL.md router, the READMEs,
  and the template list — so the skill never pointed at them. They are now documented
  and routed:
  - `overfitting.py` — Probability of Backtest Overfitting via CSCV (Bailey, Borwein,
    López de Prado & Zhu 2017): `pbo_cscv`, `performance_degradation`, `build_perf_matrix`.
  - `calibration.py` — probability calibration (numpy-only): reliability curve, ECE/MCE,
    Murphy Brier decomposition, Platt scaling, and isotonic (PAVA) recalibration.
- Removed the machine-specific absolute install path from `skills/claude-quant/README.md`.

### Added — packaging
- **`requirements.txt`** — the runtime dependency floor (numpy, pandas) needed to *run*
  the templates/example, plus the optional scientific stack the reference snippets adapt.
- **`.github/workflows/tests.yml`** — CI that runs `run_all_tests.py` and the end-to-end
  example on a Python 3.11/3.12/3.13 matrix, so the "execution-verified" promise is
  enforced on every push and catches numpy/pandas version drift.
- Clarified scope (methodology + scaffolding, not a trading system or data feed) and the
  "to *run* templates you need numpy + pandas" note in the READMEs; added a short
  "broad skill vs. quick-draw skill vs. subagent" guide to the top-level README.

### Added — tests
- `data_loader.py` self-tests gained adversarial fixtures for messy real data: a
  delisted name with no fundamental row (must stay NaN, never cross-symbol leak), an
  all-NaN fundamental value (NaN propagates, no fabricated number), a symbol missing
  from the corporate-action factor table (defaults to 1.0, no rows dropped), and a
  non-session/holiday row sneaking into the feed (dropped by `align_to_sessions`).
- `regime.py` self-tests gained adversarial fixtures: vol estimators stay finite on a
  flat/all-zero window and under a lone extreme bar, non-finite input is rejected loudly
  (not silently NaN-propagated), HAR-RV degrades gracefully on a too-short series, and a
  flat series triggers no spurious CUSUM change point.

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
