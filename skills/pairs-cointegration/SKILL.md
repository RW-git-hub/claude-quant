---
name: pairs-cointegration
description: >-
  Use when asked to "build a pairs trade", "find a cointegrated pair", "trade a mean-reverting
  spread", "Engle-Granger / Johansen test", "estimate the hedge ratio / half-life", "z-score
  spread strategy", "test a spread or series for mean reversion (OU half-life / ADF)",
  "cointegrated basket", or "stat-arb pair" — the quick end-to-end playbook for ONE concrete
  cointegration pair or small basket you already have in hand. For designing, validating, or
  debugging a whole relative-value book — or a spread that worked in-sample but broke down out-of-
  sample — use the stat-arb-strategist agent; the broad claude-quant skill is the full-lifecycle
  router.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Build, signal, and cost a single mean-reverting pair end-to-end. Call the shipped, self-tested functions — never rewrite them.

## Do this now
1. **Open the toolkit + spec first**: `skills/claude-quant/templates/pairs_trading.py` and `skills/claude-quant/references/stat-arb.md` (esp. §2 cointegration, §3 hedge ratio, §4 half-life). Run `python pairs_trading.py` to confirm self-tests pass.
2. **Pick candidates by economics** (same sector/supply-chain/factor), NOT raw price correlation. Confirm each leg is I(1): `adf_tstat(level)` fails to reject (`> -2.86`), `adf_tstat(diff)` rejects.
3. **Test cointegration**: `engle_granger(y, x, lags)`. Compare `adf_tstat` to the EG residual critical value (~ **-3.34** at 5%), NOT -2.86 — the estimated beta biases the naive cutoff. Run BOTH orderings (y~x and x~y); OLS is asymmetric. For a basket (>2 legs) use Johansen — see ref §2 (the template ships residual ADF only).
4. **Half-life**: `half_life(eg["resid"])`. Reject if `+inf`/`nan` (no reversion) or implausible vs your holding period.
5. **Rolling hedge ratio (no look-ahead)**: use `kalman_hedge_ratio(y, x)` (point-in-time) or `hedge_ratio` over a trailing window — NEVER full-sample beta to trade OOS. The Kalman `beta_t` is the FILTERED posterior conditioned on the same-bar `y_t`/`x_t`, so **lag it** (`beta.shift(1)`, `intercept.shift(1)`) before building the tradable spread — the unlagged residual is near-zero by construction (look-ahead). Build the spread with `spread(y, x, beta, intercept)`.
6. **Signal**: `zscore(spread, window=<a few half-lives, int>)` (pass an int — `window=None` is full-sample look-ahead) → `generate_signals(z, entry, exit, stop)`. Keep `exit < entry < stop`; the stop is mandatory.
7. **Lag, cost, validate**: `pnl_t = position.shift(1) * dspread_t`. Subtract costs via `templates/costs.py` (`slippage_total`, `borrow_cost`, `funding_cost`). Tune entry/exit/window with `templates/validation.py` `PurgedKFold`; deflate with `metrics.py` `deflated_sharpe_ratio`.

## Gotchas that ruin this
- **Cointegration breaks OOS (#1 risk)**: confirm on held-out walk-forward, never full history; the stop guards a snapped spread.
- **Full-sample beta = look-ahead**: Kalman or trailing-window only.
- **Half-life ≪ holding period** = microstructure noise eaten by costs; **≫** = capital stranded.
- **Pair-mining N pairs** needs FDR/deflation — see `references/stats-risk.md`.

## Expected output
A costed, lagged spread-PnL series with walk-forward OOS cointegration confirmed, plus reported beta-path, half-life, and deflated Sharpe.
