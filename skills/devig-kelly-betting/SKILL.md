---
name: devig-kelly-betting
description: >-
  Use when turning betting/prediction-market odds into edges and stakes — "devig the
  vig/overround", "fair probability from odds", "Shin/power devig", "Kelly stake / fractional
  Kelly / bet sizing", "closing line value / CLV", "calibrate my model / Brier / log loss",
  Polymarket / sportsbook / Betfair / Pinnacle edge — the quick single-bet devig-to-stake
  playbook. For correlated multi-bet / joint-Kelly sizing, oracle-and-settlement-risk analysis, or
  a full calibration+CLV research brief, the prediction-market-analyst agent goes deeper; the
  broad claude-quant skill owns the full betting backtest lifecycle.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Turn market odds into fair probabilities, compare to YOUR calibrated model, and size stakes without going broke.

## Procedure (do this now)
1. **Convert to decimal.** Use `american_to_decimal` / `decimal_to_implied` in `skills/claude-quant/templates/betting_markets.py`. Implied probs sum to >1; that excess is the overround — never treat `1/d` as truth.
2. **Devig to fair probabilities.** `devig_multiplicative` (two-way), or `devig_shin` / `devig_power` for multi-way / longshot-heavy fields (they relax the proportional-vig assumption; Shin models an informed-bettor fraction). Build a devigged sharp-consensus fair value, not one soft book's line (ref §9, §11).
3. **Get YOUR probability and CALIBRATE it.** Score with `brier_score` / `log_loss`, then recalibrate on a SEPARATE held-out fold (purged for time series) using `platt_scale` / `isotonic_fit` in `skills/claude-quant/templates/calibration.py` (numpy-only — no sklearn needed). A sharp-but-miscalibrated `p` systematically mis-sizes every bet (ref §15).
4. **Edge + stake.** `expected_value(p, d)`, bet only if `EV>0`, then `kelly_fraction(p, d)` clamped to `[0,1]`, scaled by 0.25–0.5 (fractional Kelly). Simultaneous/correlated bets over-stake if you sum per-bet fractions (parlay trap): `joint_kelly` in `betting_markets.py` handles a few correlated legs, but a real correlated book / settlement-risk sizing is the **prediction-market-analyst** agent's job (ref §13).
5. **Track CLV + calibration live.** `closing_line_value(entry_d, close_d)`; positive mean CLV is your earliest, lowest-variance edge signal.

## Open these
- Template: `skills/claude-quant/templates/betting_markets.py` (`python betting_markets.py` self-tests).
- Reference: `skills/claude-quant/references/prediction-sports-markets.md` (§13 joint Kelly, §15 calibration, §16 leak-free, §17 ruin sim).
- `skills/claude-quant/templates/calibration.py` — `platt_scale` / `isotonic_fit` recalibrators + reliability/ECE (numpy-only).
- `skills/claude-quant/templates/validation.py` `PurgedKFold` for purge+embargo around events.

## Gotchas
- **Look-ahead (Law 1/4):** CLV is post-hoc ONLY — selecting bets with the closing line is a leak (ref §16). Decide on pre-event prices.
- **Thin-book slippage (Law 3):** walk the book; fill at executable taker price + fees + gas, never mid.
- **Resolution/settlement risk:** haircut UMA-dispute probability; USDC locked until resolution (ref §4–§5).
- **Tiny independent samples:** simulate risk-of-ruin (ref §17) before sizing; trust CLV over noisy realized PnL.

## Output
Calibrated `p`, devigged fair `p`, per-bet EV, clamped fractional-Kelly stake, and a running CLV/Brier tracker.
