---
name: transaction-cost-model
description: >-
  Use when asked to "build a cost model", "add transaction costs to my backtest", "model slippage
  / market impact", "charge costs on turnover", "add borrow costs on shorts / funding costs on
  perps to my backtest", "compute break-even cost", "what's my capacity / max AUM", or "is my
  strategy fragile to costs" — the quick, specific playbook (not the broad claude-quant skill) for
  a size-aware commission+spread+sqrt-impact+financing model, charged on turnover, with break-even
  and capacity. For execution SCHEDULING (TWAP/VWAP/Almgren-Chriss, child-order plans) or deep
  capacity-vs-AUM decay analysis, defer to the execution-cost-analyst agent.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Build a size-aware cost model, charge it on turnover, and find where the edge dies.

## Do this now
1. **Open the template, don't rewrite it.** `skills/claude-quant/templates/costs.py` ships every component (pure, self-tested via `python costs.py`): `commission_return`, `half_spread_cost`, `square_root_impact` (~`coef*sigma*sqrt(Q/ADV)`), `slippage_total` (one half-spread + sqrt impact), `borrow_cost`/`funding_cost` (DOLLARS), `breakeven_cost_bps`, `apply_costs`.
2. **Per-name one-way rate (return units):** `commission_return + half_spread_cost + square_root_impact`. Convert weight-turnover to trade notional FIRST, then `participation = trade_notional / dollar_ADV` — Q and ADV in the SAME unit.
3. **Charge on turnover, lagged:** `turnover = weights.diff().abs()`; `net = apply_costs(gross, turnover, cost_per_turnover)`; P&L convention `pnl_t = pos.shift(1)*ret_t` (Iron Law 1).
4. **Financing** on HELD (lagged) notional: `borrow_cost` on shorts, `funding_cost` on perps — per-period rates, separate from trading cost.
5. **Break-even + capacity:** `breakeven_cost_bps(gross_ann_return, annual_turnover)` (turnover in book-turns/yr); compare to your realistic estimate. Compute ADV-cap `max_AUM ~= f*dollar_ADV/|Δw|` and plot the hump-shaped net-return-vs-AUM curve.

## Read for formulas/ranges
`skills/claude-quant/references/transaction-costs.md` (§5 impact, §9 capacity, §10 backtest wiring, §11 break-even/calibration, §12 cross-asset ranges). Cross-link `skills/claude-quant/references/microstructure.md` (§2 spread decomposition, §3 price impact, §8 TCA).

## Gotchas (Iron Laws 3, 5)
- Costs scale with **size and turnover** — flat-bps understates large orders and overstates capacity. Use sqrt impact, not linear, above ~1–2% ADV.
- Needs **sub-bp costs** to work = fragile: if break-even ≈ your estimate, call it uninvestable.
- Use one-way `Σ|Δw|` (don't net across names), keep one-way vs round-trip bases consistent, charge one-sided taxes (UK stamp) on buys only.
- Point-in-time ADV/sigma/spread — full-sample inputs leak the future (Iron Law 1).

## Expected output
Net return/Sharpe after costs, a cost-sweep (0/5/10/20/50 bps) Sharpe curve, break-even vs estimate, and a net-return-vs-AUM capacity curve.
