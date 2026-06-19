---
name: data-pit-audit
description: 'Use when asked to "audit the data", "vet the data pipeline/feed", "is this point-in-time", "survivorship check", "corporate-action / futures-roll / calendar sanity check", or BEFORE trusting any backtest — the fast pre-research data-integrity gate (the broad claude-quant skill is the full lifecycle; this is just the dataset trust gate).'
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Run this gate BEFORE any signal/backtest work. The #1 source of fake alpha is the data, not the model.

## Do this now
1. **Open the canonical pattern** `skills/claude-quant/templates/data_loader.py` and run it (`python data_loader.py`) so the self-tests pass; mirror its primitives in the target pipeline instead of hand-rolling joins: `pit_join` (merge_asof `direction="backward"`), `universe_on` (membership with open-ended end), `adjust_prices`, `align_to_sessions`, `make_available_date`.
2. **PIT joins:** fundamentals/alt/macro must key on an *availability/announcement* date, never `period_end`. If absent, apply `make_available_date` lag (10-Q ~45d, 10-K ~60-90d; macro = release calendar). Flag any forward-filling/`interpolate`/non-backward `merge_asof` in features as a leak.
3. **Survivorship:** rebuild the universe per-date from historical membership; **include delisted names + delisting returns** (bankruptcy ≈ -100% or -30% for performance delistings). Key on permanent id (permno/figi), not ticker. A constant symbol list is bias by construction.
4. **Corporate actions:** keep raw price + adjustment factor; total-return series for returns, raw levels for price-level signals.
5. **Futures:** documented roll rule + stored roll schedule; compute returns from per-contract prices, NOT `pct_change` of a back-adjusted continuous series.
6. **Time/quality:** UTC timestamps on a real exchange calendar (half-days); bars stamped by close; dedupe `(symbol,ts)`; assert monotonic+unique index and `low<=open,close<=high`; `pct_change(fill_method=None)`.

## Read for the rules
- `skills/claude-quant/references/data.md` — authoritative PIT/survivorship/roll/calendar how-and-why.
- `skills/claude-quant/references/pitfalls.md` — failure signatures to grep for.

## Gotchas
- Today's index membership is historical fiction → survivorship.
- Back-adjusted futures can go negative/near-zero → `pct_change`/log returns are garbage.
- Dividend adjustment is not vintage-stable — it rewrites historical levels; snapshot it.

## Iron Laws
No look-ahead (#1), no survivorship (#2), correctness-before-cleverness (#6).

## Expected output
Pass/fail per checklist item with evidence (delisting count, IC pre- vs post-announcement-date, negative-price scan, duplicate/gap counts) and a go/no-go verdict on trusting the dataset.
