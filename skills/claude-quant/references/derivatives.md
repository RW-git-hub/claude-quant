# Derivatives, Options, and FX/Rates

Pricing, the Greeks, the volatility surface, vol trading, delta-hedging, and the FX/rates conventions that quants get wrong. Companion code: `templates/options.py`. Stats and risk live in `references/stats-risk.md`; costs in `references/transaction-costs.md`.

Conventions used throughout: continuously compounded rates `r` (domestic) and dividend yield / foreign rate `q`/`r_f`; time to expiry `T` in years; `N(.)` standard normal CDF, `n(.)` its PDF. Vol `sigma` is annualized. Forward `F = S*exp((r-q)*T)`.

---

## 1. Options basics

An option gives the right (not obligation) to transact the underlying at strike `K`.

| | Long call | Long put |
|---|---|---|
| Payoff at expiry | `max(S_T - K, 0)` | `max(K - S_T, 0)` |
| Right to | buy at K | sell at K |
| Bullish/bearish | bullish | bearish |
| Max loss | premium | premium |
| Max gain | unbounded | `K - premium` |

Short positions invert the payoff: a short call has unbounded loss. The buyer pays a premium up front; the writer collects it and takes the obligation.

**Moneyness.** For a call (reverse for puts):
- ITM (in the money): `S > K` — positive intrinsic value.
- ATM (at the money): `S ~ K`. Distinguish ATM-spot (`K = S`) from ATM-forward (`K = F`); the forward convention is standard in FX and for vol quoting.
- OTM (out of the money): `S < K` — zero intrinsic, all time value.

**Intrinsic vs time value.** Price = intrinsic + time value.
- Intrinsic (call) = `max(S - K, 0)`. Cannot be negative.
- Time value = price − intrinsic. For a European option time value can in principle be *slightly negative* for a deep-ITM option (the discounting on `K` can dominate), which is why discounted intrinsic is the correct lower bound (see below); for American options time value ≥ 0. It captures optionality (the chance of finishing further ITM) and decays to zero at expiry. Time value is maximized near ATM and for longer `T` and higher `sigma`.

> Detect: an option priced *below* discounted intrinsic value. Fix: that is an arbitrage (or a stale/wrong quote, or a deep-ITM American where you should exercise). For European options, the relevant lower bound is discounted intrinsic `max(S*exp(-qT) - K*exp(-rT), 0)`, not undiscounted intrinsic — a deep-ITM European call can legitimately trade below `S - K`.

---

## 2. Black-Scholes-Merton

With continuous dividend yield (or foreign rate) `q`:

```
d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)

Call = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
Put  = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)
```

`q = 0` recovers vanilla Black-Scholes (non-dividend equity index proxy). For FX, `q = r_f` (foreign rate) — this is the Garman-Kohlhagen model. For options on futures, use Black's model: replace `S*exp(-q*T)` with `F*exp(-r*T)` and drop the drift term (`d1 = (ln(F/K) + 0.5*sigma^2*T)/(sigma*sqrt(T))`).

**Assumptions and where they break:**

| Assumption | Reality | Consequence |
|---|---|---|
| Constant, known `sigma` | Vol is stochastic and clusters | The smile/skew (sec. 6); single-vol pricing mis-hedges wings |
| Lognormal returns (GBM) | Fat tails, negative skew | Underprices OTM puts; tail risk underestimated |
| Continuous paths, no jumps | Gaps (earnings, FX pegs, crypto liquidations) | Delta-hedging breaks across a gap; short gamma blows up |
| Frictionless, continuous hedging | Spreads, discrete rebalancing | Hedging error; replication is approximate (sec. 8) |
| Constant `r`, `q` | Curves, dividend timing/discreteness | Rho/dividend exposure; discrete divs need a tree or escrowed-dividend model |
| European exercise | Many listed options are American | Early-exercise premium ignored (sec. 9) |

> BSM is a *quoting and risk-translation device*, not a belief about the true distribution. The market quotes prices in implied-vol units because everyone agrees the model is wrong in the same way (the smile encodes the disagreement with lognormality).

---

## 3. The Greeks

Sensitivities of option value `V`. Analytic forms below assume BSM with yield `q`. (`n` = normal PDF.)

```
Delta  call:  exp(-q*T)*N(d1)            put:  exp(-q*T)*(N(d1) - 1)
Gamma  (both): exp(-q*T)*n(d1) / (S*sigma*sqrt(T))
Vega   (both): S*exp(-q*T)*n(d1)*sqrt(T)            # per 1.00 absolute change in sigma; divide by 100 for per vol-point (1%)
Theta  call:  -S*exp(-q*T)*n(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2)  + q*S*exp(-q*T)*N(d1)
       put:   -S*exp(-q*T)*n(d1)*sigma/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2) - q*S*exp(-q*T)*N(-d1)
Rho    call:   K*T*exp(-r*T)*N(d2)        put:  -K*T*exp(-r*T)*N(-d2)
```

(Theta as written is per year; divide by 365 for per-calendar-day, or by 252 for per-trading-day.)

| Greek | Measures (∂V/∂…) | Sign (long opt) | Hedging use |
|---|---|---|---|
| Delta | underlying `S` | call +, put − | Trade `−Delta` units of underlying to neutralize |
| Gamma | Delta itself (`∂²V/∂S²`) | + (both) | Long gamma ⇒ buy low/sell high when re-hedging; rebalance frequency |
| Vega | implied vol `sigma` | + (both) | Hedge with other options; size vol bets |
| Theta | time decay (per day if `/365`) | − (both, usually) | The carry cost of long gamma; theta ≈ −gamma rent |
| Rho | rate `r` | call +, put − | Usually small; matters for LEAPS, rates products |

**Key relationships:** Gamma and Vega are always positive for long vanilla options (same sign for calls and puts) and peak near ATM (strictly, Gamma and Vega peak slightly above ATM-spot, around the ATM-forward strike). Theta and gamma trade off: being long gamma costs theta. Deep-ITM options have delta → ±1 and gamma → 0.

**Second-order Greeks (vol risk management):** closed forms below (per 1.00 of vol; divide by 100 for per vol-point). They are call/put-symmetric except charm.
- **Vanna** = `∂Delta/∂sigma = ∂Vega/∂S = -exp(-q*T)*n(d1)*d2/sigma`. How delta moves as vol changes (or vega as spot moves). Sign follows `−d2`, so it flips across the money. Drives the cost of risk reversals; dominant in skewed surfaces.
- **Volga (vomma)** = `∂Vega/∂sigma = Vega*d1*d2/sigma`. Convexity of value in vol; long butterflies/strangles are long volga (positive where `d1*d2 > 0`). Matters when vol-of-vol is high.
- **Charm** = `∂Delta/∂T_calendar = −∂Delta/∂tau` (delta decay, per year). Closed form: `common = exp(-q*T)*n(d1)*[(r-q)/(sigma*sqrt(T)) - d2/(2T)]`, then `charm_call = q*exp(-q*T)*N(d1) - common`, `charm_put = -q*exp(-q*T)*N(-d1) - common`. Causes a static option's delta to drift over time even if spot is flat — the source of "hedge re-balancing at the open" flows and pin risk near expiry.

> Detect: a "delta-neutral" book that bleeds on quiet days. Fix: check charm (delta drifted as `T` shrank) and vanna (delta moved because the surface shifted), not just spot delta.

---

## 4. Put-call parity

For European options on the same `K`, `T`:

```
C - P = S*exp(-q*T) - K*exp(-r*T)
```

Equivalently `C - P = (F - K)*exp(-r*T)`. This is model-free (no vol assumption) — it follows from no-arbitrage alone.

**Synthetic positions:** long call + short put = synthetic long forward; combining options and the underlying replicates any leg. Conversions/reversals (and box and jelly-roll trades) lock these in.

> Detect: `C - P` deviates from `S*exp(-qT) - K*exp(-rT)` beyond bid-ask + borrow cost. Fix: it is an arbitrage *only* if you used the correct `q` (borrow fee / hard-to-borrow rate, discrete dividends) and `r`. Most apparent parity violations are wrong dividend/borrow assumptions, American exercise premium, or stale quotes — not free money. For American options, parity becomes an inequality.

---

## 5. Implied volatility

IV is the `sigma` that makes the BSM price equal the market price. There is no closed form — invert numerically (Newton on vega, or Brent for robustness across the whole surface).

```python
# Newton step; fall back to bisection/Brent when vega is tiny (deep ITM/OTM)
sigma -= (bs_price(sigma) - mkt_price) / vega(sigma)
```

**IV ≠ realized vol.** IV is the market's *risk-neutral* expectation of future vol plus a **variance risk premium (VRP)**: implied is systematically above subsequent realized (sellers demand compensation for crash risk). VRP is the structural reason short-vol/short-variance strategies earn positive carry — and why they crash. Compare IV to a forward-looking realized estimate over the *same horizon*, not trailing realized.

**ATM vs wings.** ATM IV is the cleanest vol read (highest vega, tightest spread, least model-dependence). Wing IVs (deep OTM) embed tail/skew premia and are far more sensitive to the smile model and to supply/demand. Do not average raw wing IVs as if they were the same quantity.

> Detect: comparing today's 30-day IV to trailing 20-day realized and calling it "cheap/rich." Fix: align horizons (30-day IV vs realized over the *next* 30 days, or use a same-window estimator), and account for VRP — IV > realized is the norm, not an edge by itself. (Note: comparing IV to *subsequent* realized is fine for research/attribution but is itself a look-ahead quantity — you cannot trade on next-30-day realized at trade time.)

---

## 6. The volatility surface

IV as a function of strike (or delta) and expiry. The surface is the object you trade; spot vol is a slice.

- **Skew/smile.** Equity index: persistent **downside skew** — OTM puts richer than OTM calls (crash protection demand, leverage effect). Single-name equity is flatter/smiley. **FX: roughly symmetric smile** for major pairs (no natural "down" direction), tilting to risk-reversal skew when one currency has crash risk. Commodities often show forward/upside skew. Crypto: pronounced and regime-switching, often call skew in bull phases.
- **Term structure.** ATM IV vs expiry: usually upward-sloping in calm regimes (contango), inverting in stress (front-month spikes above back-month). Earnings/events create local humps.
- **Sticky-strike vs sticky-delta (sticky-moneyness).** A dynamics assumption: under sticky-strike the IV at each fixed `K` stays put as spot moves (so ATM IV changes); under sticky-delta the smile moves with spot (IV at fixed delta/moneyness is constant). This changes your *effective* delta because of vanna — sticky-delta adds a skew-driven delta adjustment. Equity index intraday is often closer to sticky-strike; trending/FX markets closer to sticky-delta. Pick the regime; it is a P&L attribution choice, not cosmetic.
- **Parametrizations.** **SVI** (stochastic vol inspired) fits a single-expiry smile in total-variance space; enforce no-arbitrage (no butterfly/calendar arbitrage) via the SSVI / "no-butterfly" constraints. **SABR** is a stochastic-vol model with a near-closed-form IV approximation, the rates-desk standard for swaptions and caps/floors.

> Detect: pricing or risking every strike at the ATM vol ("one-vol-fits-all"). Fix: interpolate on the surface; a single vol misprices wings and zeroes out skew Greeks (vanna/volga), so your hedge ratios are wrong exactly where tail losses live.

---

## 7. Vol trading structures

| Structure | Construction | Primary exposure |
|---|---|---|
| Straddle | long call + put, same K (ATM) | long vega + gamma, pays theta |
| Strangle | long OTM call + OTM put | cheaper vega, more convex (volga), wider breakeven |
| Risk reversal | long OTM call − short OTM put (or reverse) | **skew** (vanna); near vega-neutral |
| Butterfly | long wings − short body (e.g. +1/−2/+1) | **convexity of the smile** (volga); curvature |
| Calendar spread | short near-expiry − long far-expiry, same K | term structure; long vega, plays decay differential |
| Variance swap | swap realized variance vs strike `K_var` | pure variance, constant cash gamma via static option strip |
| Vol swap | swap realized vol vs strike | pure vol; needs convexity adj. (variance ≠ vol²) |
| Dispersion | short index vol vs long single-name vol | implied correlation |

**Long gamma vs short gamma P&L.** Long gamma = long convexity: you profit from large moves (realized > implied) and *pay theta* for the privilege (you rebalance against the move, buying low/selling high). Short gamma = short convexity: you collect theta and lose on large moves — the classic "picking up pennies in front of a steamroller." The breakeven move per period is set by the theta/gamma ratio (≈ implied daily move).

Variance swaps are preferred over straddles for clean vol exposure because a delta-hedged straddle's exposure drifts with spot, while a variance swap is engineered to have *constant cash (dollar) gamma* — its exposure to *variance* is constant, delivered via a `1/K²`-weighted strip of options. (Its vega *in vol terms* is not constant: it scales with the level of vol, roughly `2*sigma*notional_var`.) **Variance ≠ vol²:** a vol swap requires a convexity correction. Because variance is convex in vol (Jensen), `E[sqrt(var)] ≤ sqrt(E[var])`, so the fair vol-swap strike sits *below* the square root of the variance-swap strike; quoting a vol swap at `sqrt(K_var)` and ignoring the (negative) convexity adjustment overpays for the vol swap.

---

## 8. Delta-hedging

The BSM derivation assumes continuous, costless re-hedging that perfectly replicates the option. In practice you hedge discretely.

**The core identity (delta-hedged P&L over `dt`):**

```
P&L ≈ 0.5 * Gamma * (dS)^2  +  Theta * dt        (delta-hedged, vol/rates fixed)
```

For a long option (Gamma > 0, Theta < 0): you gain `0.5*Gamma*(dS)²` from realized moves and lose `|Theta|*dt` from decay. Aggregated, your P&L is proportional to **realized variance minus implied variance**:

```
P&L ≈ 0.5 * Gamma * S^2 * (sigma_realized^2 - sigma_implied^2) * dt
```

So delta-hedging a long option is a bet that realized vol exceeds the implied vol you paid. This is *why* an option's value reduces to a vol view once delta is neutralized.

**Discrete-hedging error.** With finite rebalancing, replication is imperfect; the hedging error has approximately zero mean but nonzero variance. The standard result (Boyle-Emanuel / Bertsimas-Kogan-Lo) is that the *standard deviation* of the hedging error scales like `~1/sqrt(N)` in the number of rehedges — equivalently its *variance* scales like `~1/N` (more frequent ⇒ lower variance but higher transaction cost). The gamma-theta tradeoff, balanced against costs, sets the optimal frequency. The companion `delta_hedge_pnl` simulator makes this concrete: the gamma-theta attribution reconciles to the realized hedged P&L only in the `dt → 0` limit, and the gap *is* the discrete-hedging error — it is unbiased across paths and shrinks under refinement.

> Detect: a backtest that delta-hedges continuously / costlessly and shows a smooth Sharpe. Fix: hedge on a realistic schedule (fixed-time or delta-band), charge spread + impact on every rebalance (see `references/transaction-costs.md`), and report the *distribution* of P&L — discrete hedging adds fat-tailed slippage, especially short gamma into a gap.

---

## 9. American options & early exercise

American options allow exercise any time before `T`. Price them on a lattice or by PDE/Monte-Carlo (Longstaff-Schwartz).

**CRR binomial tree.** Over `N` steps of `dt = T/N`:

```
u = exp(sigma*sqrt(dt));  d = 1/u
p = (exp((r - q)*dt) - d) / (u - d)        # risk-neutral up-prob
```

Backward-induct: at each node `value = max(intrinsic, exp(-r*dt)*(p*V_up + (1-p)*V_down))`. The `max` with intrinsic is the early-exercise check — that is the only structural difference from a European tree. Guard `0 ≤ p ≤ 1`: `p` stays in `[0,1]` exactly when `sigma*sqrt(dt) ≥ |(r-q)*dt|`, so with large `|(r-q)|` (huge carry) and too few steps `p` leaves `[0,1]` and the tree silently produces an arbitrageable price (the companion `crr_price` raises rather than return it; the fix is more steps, which shrinks `dt` faster than `sqrt(dt)`).

**When early exercise matters:**
- **American calls on a dividend-paying stock:** optimal to exercise only just *before* an ex-dividend date if the dividend exceeds the remaining time value. On a *non*-dividend stock, never exercise a call early (so it equals its European value).
- **American (deep-ITM) puts:** can be optimal to exercise early to capture interest on `K` now (the put's early-exercise premium grows with `r` and moneyness). This holds even without dividends.
- The early-exercise premium widens with high rates, high dividends, and deep moneyness; near zero for OTM and low-rate cases.

> Detect: pricing listed single-stock American options (or any dividend payer) with European BSM. Fix: use a tree/PDE with the discrete dividend schedule; BSM will misprice the early-exercise premium and produce wrong deltas around ex-div dates.

---

## 10. FX specifics

**Spot, forward, covered interest parity (CIP).** With domestic rate `r_d` and foreign `r_f` (continuous):

```
F = S * exp((r_d - r_f) * T)
```

Quote convention matters: `S` is units of domestic per one unit of foreign (e.g. USD per EUR for EURUSD). The forward points (`F − S`) are pure interest differential; deviations from CIP reflect cross-currency basis (funding/balance-sheet costs), not arbitrage in normal markets.

**Carry trade.** Borrow low-yield, invest high-yield; earns the rate differential. CIP says the forward already prices this out, but the *carry trade bets uncovered interest parity (UIP) fails* — historically it does (carry earns a premium that periodically crashes when the funding currency rallies). Carry P&L ≈ rate differential − spot depreciation of the high-yield currency.

**FX option quoting in delta terms.** FX desks quote vol by **delta**, not strike, and define structures off the smile:
- **ATM** = the delta-neutral straddle (DNS) vol. For the (non-premium-adjusted) forward-delta convention this corresponds to `K = F*exp(0.5*sigma²T)` (the strike where the forward-delta of the call and put are equal and opposite, i.e. `d1 = 0`); the exact ATM strike depends on the delta convention. This is *not* ATM-spot.
- **25-delta risk reversal (RR)** = `IV(25Δ call) − IV(25Δ put)` — the **skew** of the smile (sign and size of the tilt).
- **25-delta butterfly (BF)** = `0.5*(IV(25Δ call) + IV(25Δ put)) − IV(ATM)` — the **convexity** (smile curvature). (Note: market-quoted "broker butterfly" / smile-strangle conventions can differ from this simple vol-of-strangle definition; reconcile against your data source.)

From `{ATM, RR, BF}` you reconstruct the 10/25-delta wing vols and build the surface. Be careful with delta conventions: spot vs forward delta, and **premium-adjusted** delta (used for premium-in-foreign-currency pairs) — getting this wrong shifts every strike.

> Detect: treating EURUSD "ATM vol" as the vol at `K = S`, or using spot delta where the broker quotes premium-adjusted forward delta. Fix: match the exact convention (ATM-forward / delta-neutral straddle, correct delta type) before inverting strikes — otherwise your strikes and hedge ratios are systematically off.

---

## 11. Rates specifics

**Yield curve.** Discount factors `D(t) = exp(-y(t)*t)` (continuous) define the term structure of zero rates `y(t)`. Bootstrap from deposits, futures/FRAs, and swaps. Build on the correct curve (post-2008 OIS discounting; post-LIBOR-transition, SOFR/RFR curves — discounting curve separate from the forecasting/projection curve).

**Duration and convexity** (price sensitivity to yield `y`):

```
Modified duration D_mod = -(1/P) * dP/dy
Convexity        C      =  (1/P) * d²P/dy²
dP/P ≈ -D_mod * dy + 0.5 * C * (dy)^2
```

Duration is the first-order (linear) rate risk; convexity is the second-order curvature (always positive for vanilla / option-free bonds — bonds gain more from a yield drop than they lose from an equal rise; callable/MBS can be negatively convex). Long convexity is the rates analogue of long gamma.

**DV01 (dollar value of a basis point)** = `−dP/dy * 0.0001` ≈ `D_mod * P * 0.0001`. The desk's working unit of rate risk; size and hedge trades to net DV01.

**Curve trades.**
- **Steepener:** profits when the curve steepens (long the short end / short the long end, DV01-neutral). 
- **Flattener:** the reverse. 
- Construct DV01-neutral so you isolate the *slope* move, not the level.

**Swaps.** A vanilla interest-rate swap exchanges fixed for floating coupons on a notional. The par swap rate is the fixed rate making PV = 0:

```
swap_rate = (1 - D(T_n)) / sum_i( tau_i * D(T_i) )      # standard single-curve form
```

(the denominator is the annuity / PV01 of the fixed leg; this single-curve identity assumes the floating leg discounts to `1 - D(T_n)`, which holds when projection and discount curves coincide — under dual-curve/OIS discounting compute the floating leg PV from projected forwards explicitly). Swaptions (options on swaps) are quoted in vol and modeled with SABR/Black. Note the day-count and frequency conventions (`tau_i`) — they materially move the rate.

> Detect: hedging a curve trade to equal *notional* instead of equal *DV01*. Fix: DV01-match the legs, or you're net long/short duration and your "pure slope" bet is contaminated by a level move.

---

## Pitfalls

| Pitfall | Detect | Fix |
|---|---|---|
| BSM on American / dividend payers | Pricing listed single-name options with European BSM; wrong values near ex-div | Use CRR tree / PDE with discrete dividend schedule; check early-exercise nodes |
| One-vol-fits-all (ignoring the smile) | Every strike risked at ATM vol; vanna/volga show as zero | Interpolate on the surface (SVI/SABR); compute skew Greeks |
| Confusing IV and HV (realized) | Comparing 30-day IV to trailing 20-day realized | Align horizons; account for the variance risk premium (IV > realized is normal) |
| Continuous/costless delta-hedge assumption | Smooth hedged-P&L curve, no slippage | Discrete rehedge schedule + costs; report P&L distribution, model gap risk |
| Spot vs forward (and ATM-spot vs ATM-forward) | Using `K=S` for "ATM"; spot delta where forward/premium-adj is quoted | Match exact convention before inverting strikes; use `F = S*exp((r-q)T)` |
| Sign errors in put Greeks | Put delta entered positive; put theta sign flipped | Put delta = `exp(-qT)*(N(d1)-1) < 0`; rho_put < 0; re-derive from the table |
| Wrong `r`/`q` in parity ⇒ phantom arb | `C-P` "violation" without checking borrow/dividends | Use borrow fee + discrete dividends for `q`, OIS/RFR for `r`; American ⇒ inequality |
| Variance ≠ vol² | Pricing a vol swap as if linear in variance | Apply the (negative) vol-swap convexity correction; fair vol-swap strike < sqrt(K_var) |
| Curve trade hedged by notional not DV01 | Net duration leaks into a "slope" bet | DV01-neutralize the legs |
| Hedging-error variance scaling | Claiming variance ∝ 1/sqrt(N) | Std-dev ∝ 1/sqrt(N), variance ∝ 1/N (Boyle-Emanuel) |

`templates/options.py` implements all of the above: `bs_price` and the full Greek set — `bs_delta`/`bs_gamma`/`bs_vega`/`bs_theta`/`bs_rho` plus the second-order `bs_vanna` (`-exp(-qT)n(d1)d2/sigma`), `bs_volga` (`Vega·d1·d2/sigma`) and `bs_charm` (the `∂Delta/∂T_calendar` closed form) bundled in `greeks()`; `implied_vol` (Newton + bisection inversion with no-arb guards); `crr_price`/`crr_american` (the CRR tree with the early-exercise overlay and a `0 ≤ p ≤ 1` guard); and `delta_hedge_pnl`, a discrete, cost-charged, no-look-ahead delta-hedging simulator that returns realized P&L decomposed into the `0.5*Gamma*dS² + Theta*dt` gamma-theta attribution plus the cost drag. Every formula is checked against finite differences and analytic limits in the module's self-tests.
