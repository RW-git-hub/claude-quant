# Agentic Quant Playbooks

This file is the skill's **"what do I actually do"** router. When a user makes a real
quant request, find the matching playbook below and follow its numbered checklist.
Each step names the **exact shipped file/function** to use and ends with the failure
modes to watch (always cross-referenced to `references/pitfalls.md`).

These playbooks are deliberately concrete. Do not improvise an alternative pipeline
when a playbook exists — the steps encode the Iron Laws so you cannot skip them by accident.

> **Function-name discipline.** Every function below is named exactly as it ships in
> `templates/`. Import and call the real symbol; do not invent shorter aliases (there is
> `metrics.sharpe_ratio`, not `metrics.sharpe`; `risk.expected_shortfall`, not `risk.es`).
> If you are unsure of a signature, open the template and read the docstring before calling.

---

## The Iron Laws (non-negotiable, apply to every playbook)

1. **No look-ahead.** A signal at time `t` may use only information available by `t`.
   PnL convention: `pnl_t = pos.shift(1) * ret_t` (positions lagged vs the returns they earn).
   In `backtest_skeleton.py` this is enforced by `build_positions(target, lag=1)` (lag=0 is the
   deliberately-kept look-ahead bug used only in the demo).
2. **No survivorship.** Universes must be point-in-time; include delisted/dead names.
   Build via `data_loader.universe_on(...)`, not "today's tickers backfilled."
3. **Mandatory costs.** No backtest is reportable without explicit commission + spread +
   impact (and borrow/funding where relevant) via `costs.py`. Gross-only numbers are drafts.
4. **Sacred OOS.** The out-of-sample / test period is touched **once**, at the end.
   Every tuning decision happens in-sample / via CV. If you peek, it's no longer OOS.
5. **Deflated stats.** A raw Sharpe from a search is inflated. The multiple-testing correction
   is the **Deflated Sharpe Ratio** (`metrics.deflated_sharpe_ratio`), which needs the trial
   count *and* the cross-trial std of per-period Sharpe estimates. The Probabilistic Sharpe
   (`metrics.probabilistic_sharpe_ratio`) tests one strategy against a benchmark Sharpe and is
   **not by itself** a search correction. Always report the number of trials.

If a user request conflicts with an Iron Law, say so explicitly and propose the compliant
version — do not silently produce a leaky/optimistic result.

---

## Playbook 1 — Idea → research plan

**Trigger:** "I have an idea for a strategy/factor", "research this signal", "where do I start".

1. **State the hypothesis as a falsifiable claim.** Write it down: what is the economic
   mechanism, the predicted sign, the horizon, the universe, and the rebalance frequency?
   ("Stocks with high X earn higher forward N-day returns because Y.") Read
   `references/factor-research.md` (hypothesis framing) and `references/research-backtest.md`
   (workflow) before touching data. No mechanism → no test.
2. **Pin the universe and data point-in-time.** Use `data_loader.universe_on(date, membership)`
   for a survivorship-free universe and `data_loader.pit_join(prices, fundamentals, ...)` for any
   fundamental/alt data so each row carries only what was knowable then (use
   `data_loader.make_available_date(...)` to derive the availability date from the report-period
   end). Adjust prices with `data_loader.adjust_prices(...)` for splits/dividends. See
   `references/data.md`.
3. **Build the factor / signal.** Compute the raw factor; cross-sectionally clean it with
   `factor_research.cross_sectional_zscore(...)` / `factor_research.cross_sectional_rank(...)`,
   `factor_research.winsorize(...)`, and `factor_research.neutralize(factor_df, *exposure_dfs)`
   (e.g. sector/beta/size neutral) so you test the *residual* edge, not a known exposure.
4. **Measure predictive content before any backtest.** Run
   `factor_research.information_coefficient(factor_df, fwd_ret_df, method="spearman")` (per-date
   rank corr of factor at `t` vs **forward** returns; `method="spearman"` is the default — do not
   pass `"pearson"` for IC) and `factor_research.ic_summary(...)`; inspect monotonicity with
   `factor_research.quantile_returns(...)` and `factor_research.quantile_spread_summary(...)`.
   If the IC is noise, stop here — backtesting noise wastes trials and inflates DSR penalties later.
5. **Leak-free backtest.** Translate the factor to positions with `backtest_skeleton.run_backtest`
   (it calls `build_positions(..., lag=1)`, so positions only earn future returns). Apply costs
   from `costs.py` (`commission_return` + `half_spread_cost` + `square_root_impact`, combined via
   `slippage_total`/`apply_costs`). Never write a bespoke loop that re-implements signal timing.
6. **Statistics.** Compute performance with `metrics.py`
   (`sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `turnover`,
   `information_ratio`) and add the honesty layer: `metrics.sharpe_se` /
   `metrics.sharpe_tstat` (Lo 2002) and, since you ran a search,
   `metrics.probabilistic_sharpe_ratio`. Read `references/stats-risk.md` for interpretation and
   the autocorrelation caveat on annualization (`metrics.lo_annualization_factor`).
7. **Decide and document trials.** Record how many variants you tried (it feeds Playbook 3/4 and
   the DSR `n_trials`). Promote to validation (Playbook 4) only if IC and net Sharpe survive.

**Failure modes (see `references/pitfalls.md`):** HARKing (hypothesis written after seeing
results); IC computed on Pearson or on same-day instead of forward returns; universe backfilled
from today's listings; costs added "later" (i.e. never); celebrating a gross Sharpe.

---

## Playbook 2 — Audit an existing backtest for look-ahead / leakage

**Trigger:** "is this backtest legit", "audit my code", "this Sharpe seems too good".

Treat a Sharpe that looks too good as guilty until proven innocent. Walk the checklist:

1. **Signal-lag check.** Confirm PnL uses `pos.shift(1) * ret` (or equivalent bar lag). Search
   the code for any same-bar use of close-to-decide-then-trade-at-same-close. Compare against
   `backtest_skeleton.build_positions` / `run_backtest` (note `lag=0` is the known bug). The
   single fastest leak test: shift the signal one extra bar — if the edge largely survives,
   suspect timing leakage already baked in.
2. **PIT feature check.** Every feature must be as-of, not restated. Verify fundamentals/alt-data
   came through `data_loader.pit_join(...)` with an availability date (`make_available_date`) and
   that there are no future-dated joins, no `fillna`/interpolation that reaches backward from the
   future, no `.rolling(..., center=True)` centered windows.
3. **Scaler/normalization fit scope.** Any standardization, winsorization, PCA, or model fit must
   be **fit on train only** and applied to test. A `StandardScaler().fit(all_data)` is leakage.
   (Note: `factor_research` standardizers like `cross_sectional_zscore` are *cross-sectional*
   per-date and therefore leak-free across time; the danger is any *time-series* fit on the full
   sample.) See `references/ml-for-alpha.md`.
4. **CV purge + embargo.** If there's cross-validation, it must purge overlapping-label samples
   and embargo around test folds. Replace plain KFold with `validation.PurgedKFold` (or
   `validation.CombinatorialPurgedKFold`). Confirm the `label_horizon` argument matches the actual
   label span so the purge gap covers it.
5. **Survivorship check.** Inspect the universe construction. If it's "current constituents"
   applied historically, it is biased — rebuild with `data_loader.universe_on(...)`.
6. **Costs realism.** Confirm commission + spread + impact via `costs.py`; check turnover with
   `metrics.turnover` and sanity-check that the edge per trade exceeds round-trip cost using
   `costs.breakeven_cost_bps(gross_ann_return, annual_turnover)`. A high-turnover strat with toy
   costs is the most common false positive.
7. **Re-run and diff.** Re-run with leaks fixed; report the gross→net and biased→PIT Sharpe
   deltas so the user sees the cost of each bias.

**Failure modes (see `references/pitfalls.md`):** "trade at the signal bar's close"; restated
fundamentals; scaler fit on full sample; KFold on overlapping labels; index reconstructed from
today's membership; spreads/impact omitted on a 300%-turnover book.

---

## Playbook 3 — Cross-sectional factor study

**Trigger:** "study this factor", "does this signal have alpha cross-sectionally".

1. **PIT universe + clean factor.** As in Playbook 1 steps 2–3: `data_loader.universe_on`,
   `data_loader.pit_join`, then `factor_research.winsorize` and
   `factor_research.neutralize(factor_df, *exposure_dfs)` (remove sector/size/beta so you isolate
   the factor's marginal information).
2. **IC and decay.** `factor_research.information_coefficient` + `factor_research.ic_summary`
   for mean IC, IC IR, t-stat, and hit rate; recompute against multiple forward horizons to study
   IC decay and pick the holding period. Spearman (rank), forward returns — always (it is the
   default; do not switch to Pearson).
3. **Quantile spread.** `factor_research.quantile_returns` (+ `quantile_spread_summary`) for
   monotonicity and the long-short top-minus-bottom spread. Non-monotone deciles with a good
   extreme spread = fragile.
4. **Cross-sectional regression.** `factor_research.fama_macbeth(fwd_ret_df, *exposure_dfs)` for
   the average factor premium and its t-stat. NOTE: this t-stat uses the **classic Fama-MacBeth
   time-series standard error** of the per-date coefficients (ddof=1), **not** Newey-West. If you
   need an autocorrelation-robust (Newey-West / HAC) adjustment on the premium series, apply it
   yourself downstream — the shipped function does not. This is your formal "is the premium real"
   test.
5. **Robustness.** Subject the long-short return series to
   `robustness.monte_carlo_permutation_test(...)` (or `robustness.mean_significance_permutation`)
   and `robustness.stationary_bootstrap_indices(...)` / `robustness.bootstrap_sharpe_ci(...)` for
   block-bootstrap CIs; if comparing many factor variants, use
   `robustness.reality_check_pvalue(...)` (White's Reality Check) and look for a broad
   `robustness.parameter_plateau_score(...)` rather than a lone spike. See
   `references/robustness.md`.
6. **Deflate.** You ran multiple variants — report
   `metrics.deflated_sharpe_ratio(returns, n_trials, trial_sharpe_std)` (you must supply the
   cross-trial std of per-period Sharpe estimates, not just the count) and
   `metrics.probabilistic_sharpe_ratio`. A factor that's significant before deflation but not
   after is **not** a finding. See `references/stats-risk.md`.

**Failure modes (see `references/pitfalls.md`):** un-neutralized factor that's really a sector
bet; IC inflated by a few mega-caps (check equal- vs value-weighted, winsorize); cherry-picked
quantile breakpoint; multiple-testing not deflated; bootstrap that ignores autocorrelation
(use the stationary/block bootstrap, not iid resampling).

---

## Playbook 4 — Validate a strategy before risking capital

**Trigger:** "is this ready to trade", "validate before going live", "is this overfit".

This is the gate between research and capital. Be adversarial.

1. **Walk-forward + CPCV.** Re-fit and test out-of-sample across rolling windows; for the
   robust view use `validation.CombinatorialPurgedKFold` (multiple train/test path combinations,
   purged + embargoed; `n_paths()` reports how many OOS paths you get). Plain walk-forward via
   `validation.PurgedKFold` at minimum. The OOS period from research stays sacred — do not re-tune
   on it here.
2. **Overfitting probability (PBO).** Assemble the per-period performance of every config you tried
   into a `(T × N)` matrix and call `overfitting.pbo_cscv(perf, n_blocks=16)` — the **shipped**
   CSCV implementation (Bailey-Borwein-LdP-Zhu). It returns `pbo` (is the in-sample-best config a
   coin flip OOS?), the per-split logit distribution, and `median_logit`. Build the matrix with
   `overfitting.build_perf_matrix({name: returns})`, or feed the multi-path OOS returns from
   `validation.CombinatorialPurgedKFold` as columns. Corroborate with
   `overfitting.performance_degradation(perf)` (OOS-on-IS slope and P[OOS loss | selected]). Feed
   the **full** search, not just survivors. High PBO (>0.5) → reject regardless of headline Sharpe.
   See `references/robustness.md` §6 and `references/stats-risk.md` §1.5.
3. **Deflated Sharpe.** `metrics.deflated_sharpe_ratio(returns, n_trials, trial_sharpe_std)` using
   the **true** number of configurations tried across the whole project (and the cross-trial std
   of per-period Sharpe), plus `metrics.probabilistic_sharpe_ratio` against a benchmark Sharpe.
   Both functions already apply the skew/kurtosis correction internally.
4. **MCPT + Reality Check.** `robustness.monte_carlo_permutation_test(...)` to test the realized
   equity curve against permuted-signal nulls; `robustness.reality_check_pvalue(...)` if selecting
   among strategies. Confirm a broad `robustness.parameter_plateau_score(...)` in the parameter
   surface.
5. **Cost & capacity stress.** Re-price under stressed `costs.py` assumptions (wider
   `half_spread_cost`, higher `square_root_impact`, added `borrow_cost`/`funding_cost`); find the
   cost level where the edge dies (compare net edge to `costs.breakeven_cost_bps`). Estimate
   capacity: at what AUM does impact eat the alpha?
6. **Risk / VaR backtest.** Validate the risk model itself: `risk.kupiec_pof` (unconditional
   coverage) and `risk.christoffersen` (independence; `risk.christoffersen_cc` for the joint
   conditional-coverage test) on VaR exceptions (use `risk.count_exceptions` to build the
   exception series); report `risk.gaussian_var` / `metrics.value_at_risk`,
   `risk.expected_shortfall`, and `risk.cornish_fisher_var`, plus `risk.stress_pnl` /
   `risk.stress_grid` scenarios. (There is no `risk.var`, `risk.es`, `risk.mc_var`, or
   `risk.stress` — use the exact names above.) See `references/risk-management.md`.
7. **Verdict.** Ship only if: positive **net** Sharpe OOS, low PBO, DSR significant after honest
   trial count, MCPT/Reality-Check significant, survives cost stress, and VaR backtest passes.
   Anything less is a "no" or "more research."

**Failure modes (see `references/pitfalls.md`):** tuning on the OOS set "just to check";
under-counting trials in DSR; KFold leakage from overlapping labels; ignoring capacity so the
paper edge is un-tradeable at size; passing VaR by luck (Kupiec/Christoffersen catch this).

---

## Playbook 5 — Build a point-in-time data pipeline

**Trigger:** "build a data pipeline", "avoid look-ahead in my data", "join fundamentals correctly".

1. **Read `references/data.md` first.** It defines as-of vs restated, effective vs announcement
   dates, and the corp-action conventions used below.
2. **PIT joins.** Derive an availability date with `data_loader.make_available_date(...)`, then
   use `data_loader.pit_join(prices, fundamentals, ...)` so each observation carries only data
   with an availability date `≤` the bar date. Never join on report-period date alone (that leaks
   because the report wasn't public yet).
3. **Survivorship-free universe.** `data_loader.universe_on(date, membership)` to materialize
   membership as it was on each date, including names later delisted/merged/renamed.
4. **Corporate actions.** `data_loader.adjust_prices(raw, factors, ...)` for split/dividend
   adjustment; be explicit whether returns are total-return or price-return and keep raw +
   adjusted both. (`data_loader.align_to_sessions` and `data_loader.cached_parquet` help with
   calendar alignment and caching.)
5. **Lag discipline downstream.** Document the availability lag of every field (e.g. fundamentals
   available T+N) so backtests can apply it. The pipeline's job is to make leakage *impossible*
   downstream, not merely discouraged.
6. **Validate.** Spot-check a few known events (a split, a delisting, a restatement) and confirm
   the as-of value matches what was knowable then, not the restated figure.

**Failure modes (see `references/pitfalls.md`):** joining on fiscal-period end instead of
availability date; using the latest restated fundamentals historically; today's index members
backfilled; price series silently total-return in one place and price-return in another.

---

## Playbook 6 — Size & risk-manage a book

**Trigger:** "how much should I size this", "vol target my strategy", "manage portfolio risk".

1. **Target volatility.** Scale exposure with `regime.vol_target_scale(forecast_vol, target_vol,
   ...)` driven by `regime.ewma_vol` or `regime.garch11_fit` / `regime.garch11_filter` (and
   optionally `regime.hmm_gaussian_2state` / `regime.kalman_local_level` /
   `regime.kalman_dynamic_beta` for regime/beta state, `regime.cusum_changepoints` for breaks).
   Lag the vol estimate — using same-bar realized vol to size the same bar is look-ahead. See
   `references/time-series-regimes.md`.
2. **Portfolio weights.** Combine sleeves/assets with `portfolio.py`: `min_variance_weights`,
   `max_sharpe_weights`, `mean_variance_weights`, `risk_parity_weights`,
   `inverse_variance_weights`, `hrp_weights` (robust to unstable covariances), or
   `black_litterman` to blend views with a prior. Shrink the covariance with
   `validation.constant_correlation_shrinkage` — raw sample covariance is dangerous at scale.
   See `references/portfolio-optimization.md`.
3. **Pre-trade gate.** Run every proposed order through `pretrade_checks.check_order(...)` against
   a `pretrade_checks.RiskLimits(...)` config (position/sector limits, gross/net leverage,
   concentration, liquidity/ADV caps). It is a hard gate, not advisory.
4. **VaR / ES + backtest.** Quantify tail risk with `risk.gaussian_var` /
   `metrics.value_at_risk`, `risk.expected_shortfall`, and `risk.cornish_fisher_var`; stress with
   `risk.stress_pnl` / `risk.stress_grid`; and continuously backtest coverage with
   `risk.kupiec_pof` + `risk.christoffersen` (`count_exceptions` to build the exception series).
   See `references/risk-management.md`.
5. **Live limits.** Wire the limits from `references/live-trading.md` (kill-switch thresholds,
   max daily loss, drawdown stops, exposure caps) to the running book.

**Failure modes (see `references/pitfalls.md`):** vol estimate not lagged (sizing look-ahead);
optimizer overfit to a noisy sample covariance (error maximization — prefer HRP/shrinkage);
pre-trade checks bypassed for "just this once"; reporting VaR without ever backtesting its
coverage; leverage that's fine in calm vol and lethal in a regime shift.

---

## Playbook 7 — Evaluate a prediction-market / sports bet

**Trigger:** "is this bet +EV", "size this wager", "evaluate these odds", "am I beating closing line".

1. **Read `references/prediction-sports-markets.md`** for conventions (American/decimal/implied
   odds, vig, CLV) before computing anything. Convert quotes with
   `betting_markets.american_to_decimal` / `decimal_to_implied` as needed.
2. **De-vig the market.** Convert quoted odds to implied probabilities and strip the
   overround with `betting_markets.devig_multiplicative(...)` (or `devig_power` / `devig_shin`
   for the more accurate margin models) to get the market's fair probability.
3. **Your model probability.** Produce an independent `p_model`. The edge only exists if your
   estimate is better-calibrated than the de-vigged market — otherwise you're paying the vig
   to bet the consensus.
4. **EV / edge.** Compute expected value with `betting_markets.expected_value(prob, decimal_odds)`
   from `p_model` vs the offered price. Bet only on positive post-vig EV.
5. **Kelly sizing.** Size with `betting_markets.kelly_fraction(prob, decimal_odds)`; use
   fractional Kelly (e.g. ¼–½) to account for estimation error in `p_model`. Never full-Kelly on a
   noisy edge.
6. **Calibration.** Track `betting_markets.brier_score` and `betting_markets.log_loss` over many
   bets — these tell you whether `p_model` is honest. A model that's confidently wrong shows up
   here before your bankroll does.
7. **CLV.** Measure `betting_markets.closing_line_value(entry_decimal, closing_decimal)`:
   consistently beating the closing line is the most reliable evidence of a real edge, more so
   than short-run P&L.

**Failure modes (see `references/pitfalls.md`):** betting on raw quoted odds without de-vigging;
over-betting via full Kelly on an overconfident model; mistaking variance for edge over a small
sample (calibration + CLV are the antidotes); ignoring market moves (line you can no longer get).

---

## Playbook 8 — Take a strategy live

**Trigger:** "deploy this", "go live", "ship to production trading", "paper-to-live".

Do not skip steps. Live failures are expensive and often silent.

1. **Pass Playbook 4 first.** A strategy that has not cleared validation does not go live. Confirm
   net OOS Sharpe, low PBO, significant DSR, cost/capacity survival, and a passing VaR backtest.
2. **Go-live checklist.** Walk `references/live-trading.md` end to end: data feed health,
   order-routing/connectivity, position reconciliation, kill-switch, max-daily-loss and
   drawdown limits, alerting, and a documented rollback.
3. **Pre-trade gate in the loop.** `pretrade_checks.check_order(...)` (against your
   `RiskLimits`) must run on every live order (limits, leverage, liquidity, fat-finger). Gate is
   hard-blocking.
4. **Reconciliation.** Reconcile expected vs actual fills, positions, and cash every cycle;
   alert on any break. Compare live fills against backtest cost assumptions (`costs.py`) — if real
   slippage exceeds modeled `square_root_impact` / `half_spread_cost`, the edge may not survive and
   you scale down or halt.
5. **Start small, ramp on evidence.** Paper → tiny capital → scale only as live results track the
   backtest within tolerance. Treat live as the final, ongoing OOS test.
6. **Monitor risk live.** Keep `risk.gaussian_var` / `risk.expected_shortfall` and the VaR
   backtest (`risk.kupiec_pof`, `risk.christoffersen`) running; honor the live limits and
   kill-switch from step 2.

**Failure modes (see `references/pitfalls.md`):** going live on a strategy that only passed gross
backtests; no reconciliation (silent position drift); live slippage far above modeled costs;
kill-switch/limits configured but never tested; scaling capital before live tracks paper.

---

## Asset-class & specialty cross-references

When a request is specific to an asset class or technique, layer the relevant reference onto the
playbook above:

- **Options / vol:** `references/derivatives.md` + `templates/options.py`
  (`bs_price`, greeks `bs_delta`/`bs_gamma`/`bs_vega`/`bs_theta`/`bs_rho`, `implied_vol`).
  Costs/greeks-aware sizing differs from linear assets.
- **Stat-arb / pairs:** `references/stat-arb.md` + `templates/pairs_trading.py`
  (`engle_granger`, `hedge_ratio`, `half_life`, `generate_signals`, `kalman_hedge_ratio`).
  Cointegration must be tested OOS.
- **Microstructure / execution:** `references/microstructure.md` + `templates/execution.py`
  (`twap_schedule`, `vwap_schedule`, `pov_schedule`, `almgren_chriss_trajectory`,
  `implementation_shortfall`). Intraday look-ahead is subtle — mind bar timestamps.
- **Crypto / DeFi:** `references/crypto-defi.md` + `templates/crypto_defi.py`
  (`funding_payment`, `perp_basis`, `amm_constant_product_out`, `impermanent_loss`,
  `liquidation_price`). Funding/borrow are first-class costs here.
- **Regimes / time series:** `references/time-series-regimes.md` + `templates/regime.py`
  (`ewma_vol`, `garch11_fit`, `hmm_gaussian_2state`, `kalman_local_level`, `cusum_changepoints`,
  `vol_target_scale`).
- **Labeling for ML:** `references/ml-for-alpha.md` + `templates/labeling.py`
  (`triple_barrier_labels`, `fixed_horizon_labels`, `average_uniqueness`, `meta_label`).
  Overlapping labels require purged CV (Playbook 4).

**End-to-end worked example:** `examples/end_to_end.py` runs a compliant research→validation flow.
**Self-test everything:** `templates/run_all_tests.py` validates the templates you rely on.

---

## Router quick-reference

| User asks… | Playbook | Lead files |
|---|---|---|
| "I have an idea / research this signal" | 1 | `factor_research.py`, `backtest_skeleton.py`, `metrics.py` |
| "Is this backtest legit / audit it" | 2 | `validation.py`, `data_loader.py`, `costs.py` |
| "Does this factor have alpha" | 3 | `factor_research.py`, `robustness.py`, `metrics.py` |
| "Is it ready / is it overfit" | 4 | `validation.py`, `robustness.py`, `overfitting.py`, `metrics.py`, `risk.py` |
| "Build a PIT data pipeline" | 5 | `data_loader.py` |
| "Size / risk-manage the book" | 6 | `regime.py`, `portfolio.py`, `pretrade_checks.py`, `risk.py` |
| "Evaluate this bet / odds" | 7 | `betting_markets.py` |
| "Take it live" | 8 | `live-trading.md`, `pretrade_checks.py`, `costs.py`, `risk.py` |

If no playbook fits exactly, compose the closest two — but the Iron Laws apply to every path.