---
name: position-sizing
description: 'Use when asked to "size positions/bets", "how much to bet", set a "Kelly fraction / fractional Kelly", "vol target / volatility targeting", "inverse-vol / inverse-variance weights", "risk parity / equal risk contribution / ERC", or apply "leverage / gross-net / exposure caps" — the quick how-much-to-allocate playbook (the broad claude-quant skill covers full portfolio construction and optimization).'
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Decide how much to allocate per position/bet. Map the request to one sizing regime, lift the formula from the existing template, and enforce causal estimation + leverage caps.

## Procedure
1. Pick the regime: discrete edge with known p/payoff -> Kelly; single risky asset/strategy stream -> vol targeting; many assets, no return view -> risk-based weights (inverse-variance or ERC).
2. Kelly: call `kelly_fraction(win_prob, win_loss_ratio)` in `skills/claude-quant/templates/risk.py`, then deploy a FRACTION (0.25-0.5x). Stress the bet with `risk_of_ruin(...)` (same file) — ruin probability rises sharply with bet fraction.
3. Vol targeting: `scale_t = target_vol / sigma_hat_t`, where `sigma_hat_t` is rolling/EWMA realized vol on returns up to t-1 ONLY (`ret.rolling(w).std().shift(1)` or `ewm`). Position = `scale_t * signal_t`, then clip `scale_t` to a max-leverage cap.
4. Multi-asset weights: `inverse_variance_weights(cov)` (∝ 1/σ², ignores correlation) or `risk_parity_weights(cov)`, then verify with `risk_contributions(w, cov)` — all in `skills/claude-quant/templates/portfolio.py`. Feed a shrunk/causal cov.
5. Apply gross/net leverage and per-name caps LAST; rescale proportionally, don't clip silently.

## Open
- `skills/claude-quant/templates/risk.py`, `skills/claude-quant/templates/portfolio.py`
- `skills/claude-quant/references/risk-management.md` §5 (Kelly, risk of ruin, leverage), §6 (exposure limits)
- `skills/claude-quant/references/portfolio-optimization.md` §5 (ERC), §6 (HRP)

## Gotchas
- Full Kelly assumes a KNOWN edge and is brutally volatile — size at 1/4-1/2 Kelly.
- Vol estimate must be causal; unshifted/full-sample sigma is look-ahead (Iron Law 1).
- `inverse_variance_weights` is 1/σ² (not inverse-vol 1/σ) and still ignores correlation — simultaneous positions stack risk; size on the covariance (ERC) when correlated.
- Vol-scaled sizes inflate turnover — re-check costs after scaling (Iron Law 3).

## Expected output
Per-position sizes/weights, the sigma/cov window stated, Kelly multiplier used, realized gross/net leverage vs cap, and equal-risk-contribution verification where applicable.
