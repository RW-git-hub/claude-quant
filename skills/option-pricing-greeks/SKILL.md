---
name: option-pricing-greeks
description: >-
  Use when asked to "price an option/call/put", "what's the fair value of this option / how much
  is it worth", "compute Black-Scholes/BSM/Black-76", "get the greeks
  (delta/gamma/vega/theta/rho)", "solve/back out implied vol/IV from a price", "check put-call
  parity", or "sanity-check a single option quote against no-arb bounds and the smile" — the quick
  single-instrument pricing-and-risk playbook (not the broad claude-quant router). For fitting a
  whole vol surface (SVI/SABR/SSVI), exotic/path-dependent payoffs, book-level greeks, or
  hedge-P&L simulation, hand off to the options-quant agent.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Price a European option, return its full Greek set, invert IV, and prove the quote is arbitrage-free.

## Do this now
1. **Reuse the code — never rewrite BSM.** Open `skills/claude-quant/templates/options.py` (stdlib `NormalDist`, optional numpy; self-tests via `python options.py`).
2. **Pin conventions first.** Set `S, K, T` (years, Act/365 *or* Act/252 — match how you annualize `sigma`), `r` (cont-comp domestic), `q` (carry), `sigma`, `kind`. Encode carry in `q`: equity index = div yield; FX/Garman-Kohlhagen = `r_foreign` (`r`=domestic); futures/Black-76 = price the future with `q=r`; commodity = `r − convenience`. Confirm with `forward_price(S,r,q,T)`.
3. **Price + Greeks.** Call `bs_price`, then `bs_delta`, `bs_gamma`, `bs_vega`, `bs_theta`, `bs_rho`. American/discrete-dividend payers: BSM misprices the early-exercise premium and ex-div deltas — switch to a CRR tree or Longstaff-Schwartz MC (reference sec. 9).
4. **Solve IV** with `implied_vol(price,...)`: Newton (Brenner-Subrahmanyam ATM seed) with a bisection fallback, returning `nan` outside no-arb bounds. You MUST branch on `math.isnan`.
5. **Sanity-check.** `put_call_parity_gap(...)` ≈ 0; smile/surface no-arb (no butterfly/calendar arb, SVI/SSVI) per `references/derivatives.md` sec. 6.

## Read alongside
`skills/claude-quant/references/derivatives.md` — BSM/Black-76, the Greek table (vanna/volga/charm), parity (sec. 4), IV (sec. 5), the surface (sec. 6), American/trees (sec. 9).

## Gotchas (Iron Law 6: correctness first)
- **Carry sign.** Swapping `r`/`q` (esp. FX domestic vs foreign) silently mis-signs every Greek.
- **Greek scaling.** Template vega is per 1.00 vol (÷100 for a vol point); theta is per year (÷365 per day) and negative for a long ATM option.
- **IV edge cases.** Deep-ITM/short-dated quotes have near-zero vega → `nan`, not a number; handle it, don't print garbage.
- **No-arb surface.** A persistent parity gap means wrong `q`/`r`, stale spot, or American premium — not free money.

## Expected output
Price, five Greeks (units stated), solved IV (or explicit `nan` branch), and a parity residual ≈ 0.
