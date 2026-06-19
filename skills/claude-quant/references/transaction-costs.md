# Transaction Costs & Frictions

Costs are where backtested edge goes to die. A gross Sharpe of 2.0 on a 300%/month turnover strategy can be net-negative once spread, impact, and fees are charged. This reference gives the formulas, realistic ranges, and detect/fix framing to model costs honestly and size strategies to capacity.

**Scope note:** This covers *cost models, frictions, and capacity*. Execution *algorithms* (TWAP/VWAP, Almgren-Chriss optimal scheduling) are a separate cycle — here we use the square-root impact law only as a *cost estimator*, not as a scheduler.

**Conventions used throughout:**
- Returns are simple and compound multiplicatively.
- Costs are charged on **traded notional** (turnover), applied per rebalance, on positions lagged vs the returns they earn (`pnl_t = pos.shift(1)*ret_t`).
- 1 bp = 0.0001 = 0.01%. A "round trip" = buy then sell = 2 × one-way cost.
- `Q` = order size (shares or notional), `ADV` = average daily volume (same units as `Q`), `sigma` = daily return volatility unless stated.
- All `sqrt(Q/ADV)` and `Q/ADV` terms require `Q` and `ADV` in the **same units** (both shares, or both currency/notional). Do not mix weight-space turnover with share-space ADV — convert to a common unit first (see §10).

---

## 1. Taxonomy: what you actually pay

| Layer | Components | Predictable? | Charged on |
|---|---|---|---|
| **Explicit** | Commissions, exchange/clearing fees, regulatory fees, taxes, rebates | Yes (known ex-ante) | Per trade / notional / shares |
| **Spread** | Half-spread to cross the book | Mostly (observable) | Each crossing |
| **Slippage** | Price drift between decision and fill | Partly | Each fill |
| **Impact** | Your own trading moves the price | No (model + calibrate) | Nonlinear in size |
| **Financing** | Borrow fee, margin interest, dividends on shorts, funding, swap/roll | Yes (rate known) | Per holding period |

Total cost per trade (one-way), as a fraction of notional:

```
cost = explicit_bps + half_spread_bps + slippage_bps + impact_bps
```

Note: spread, slippage, and impact overlap conceptually — effective-spread/IS calibration (§3, §11) measures their *sum*, so when you calibrate from fills do not also add separate parametric terms for the same component or you double-count. The decomposition above is for a *bottom-up parametric* model when fills are unavailable.

Financing is a **holding** cost (per unit time), not a **trading** cost — keep the two separate in accounting (see §10–11).

---

## 2. Explicit costs: commissions, fees, taxes, rebates

These are deterministic and the easiest to get exactly right. Getting them *wrong* is usually a sign-error (charging a rebate as a cost) or unit confusion (per-share vs bps).

### 2.1 Commission structures

- **Per-share** (US equities, many prime brokers): e.g. \$0.0005–\$0.005 /share. As bps: `commission_bps = per_share / price * 1e4`. A \$0.001/share fee on a \$10 stock = 1 bp; on a \$200 stock = 0.05 bp. **Per-share fees punish low-priced names** — do not model them as a flat bps.
- **Bps / value-based** (crypto, FX, futures often per-contract): e.g. 1–10 bps of notional.
- **Tiered**: rate falls with monthly volume. Backtests usually assume the *marginal* tier you'll actually trade at, not tier 1.

### 2.2 Maker vs taker (rebates)

On crypto and equity ECNs, fees depend on whether you *add* or *remove* liquidity:

- **Taker** (marketable order, crosses spread): pays a fee, e.g. +2 to +7 bps (crypto), or +0.30 \$/100 sh (US equity).
- **Maker** (resting limit order, adds liquidity): pays less, sometimes a **rebate** (negative cost), e.g. −1 bp (crypto) or −0.20 to −0.30 \$/100 sh (US equity).

Detect/fix: A common backtest error is assuming maker rebates while modeling fills as if every order executes immediately. **You cannot earn the maker rebate AND assume a guaranteed fill** — resting orders face adverse selection and non-fills. If you assume maker pricing, you must also model queue/fill uncertainty. Default to **taker** pricing unless you have a fill model.

### 2.3 Exchange / clearing fees

Per-contract for futures (e.g. \$0.50–\$2.00 round-turn per contract incl. clearing + NFA), per-trade clearing for options. Small but real; convert to bps on notional for portfolio-level accounting.

### 2.4 Taxes & regulatory fees

- **UK stamp duty (SDRT)**: 0.5% (50 bps) on *purchases* of UK shares — one-sided, buys only, and large. ETFs and some instruments are exempt. This alone kills most high-turnover UK equity strategies.
- **France/Italy FTT**: ~20–30 bps on purchases of large-cap domestic equities; design-dependent exemptions (intraday netting, market making). Rates and thresholds change — verify against current statute before charging.
- **US SEC fee (Section 31)**: charged on the *sell* side, tiny and the rate changes periodically (recently order of ~\$8–\$30 per \$1M sold, i.e. <0.3 bp; the rate is reset by the SEC and has varied widely historically). Plus FINRA TAF (per-share, sells).
- **Hong Kong, Singapore, etc.**: stamp duties of ~10–13 bps, often two-sided.

Detect/fix: FTTs and stamp duties are **one-sided** (usually buys). Charging them round-trip double-counts; charging them two-sided in a market with one-sided tax overstates cost. Encode side-awareness (charge on the signed buy leg only, not on `|Δw|`).

---

## 3. Spread cost: the cost of crossing

The **quoted spread** is `ask - bid`. Crossing once (a marketable order at the touch) costs roughly the **half-spread**:

```
half_spread_bps = (ask - bid) / (2 * mid) * 1e4
```

But the **quoted** spread overstates realized cost for passive/midpoint fills and understates it for large orders that walk the book. Use the **effective spread**, measured against the mid at order arrival:

```
effective_spread_bps = 2 * D * (fill_price - mid_arrival) / mid_arrival * 1e4
```

where `D = +1` for buys, `−1` for sells. The factor of 2 puts the effective spread on the same (full-spread) basis as the quoted spread; the **one-way** cost you actually pay is half of this, `D * (fill_price - mid_arrival) / mid_arrival * 1e4`. Effective/quoted < 1 means you captured price improvement (resting inside the spread); > 1 means you walked the book.

The **realized spread** measures what the liquidity *provider* keeps after the price moves against them (uses mid at `t + Δ`); the difference between effective and realized spread is the **price impact** component (§4). This decomposition is the bridge from spread to impact.

Detect/fix:
- **Using quoted instead of effective spread.** Quoted spread can be 1.5–2× the effective spread for liquid names with midpoint fills, and far *too small* for size. Calibrate effective spread from your own fills (§11).
- **Charging full spread when you only cross once.** A round trip crosses twice → 2 × half-spread = ~1 full spread. One-way = half-spread. Don't double.
- **Double-counting the factor of 2.** `effective_spread_bps` above already includes the ×2 to match quoted-spread basis; the per-order cost is half of it. Mixing the two conventions is a frequent 2× error.

---

## 4. Slippage models

Slippage = realized fill price worse than the reference (decision/arrival) price, net of explicit fees. Three standard parametrizations, increasing in fidelity:

### 4.1 Fixed bps
```
slippage = k_bps        # e.g. 5 bps one-way, constant
```
Crude but transparent; fine for first-pass sensitivity. Wrong whenever liquidity varies (it always does).

### 4.2 Spread-proportional
```
slippage = c * half_spread        # c ~ 1.0 for marketable, <1 for passive
```
Ties cost to the security's own liquidity. Good default for small orders (size << ADV).

### 4.3 Volatility-scaled
```
slippage_bps = a * sigma_bps      # sigma = per-bar or daily vol in bps
```
Captures that fills in volatile regimes are worse. Often combined with spread:
`slippage = half_spread + a * sigma`. This is the small-order limit of the impact model below (the `sqrt(Q/ADV)` term flattens to a size-independent floor when participation is negligible).

---

## 5. Market impact: temporary vs permanent

Impact is the part of cost caused by *your own* trading. Two components:

- **Temporary impact**: price concession to attract liquidity *now*; reverts after you stop. You pay it; it does not move the "fair" price. This is what you control via scheduling.
- **Permanent impact**: information your trade reveals; the mid shifts and stays. Roughly half of total impact in many calibrations (calibration-dependent — do not treat 50% as universal).

```
total_impact ≈ temporary_impact + permanent_impact
```

### 5.1 Linear (small-size) model
For small participation, impact is approximately linear in order size:
```
impact (return units) ≈ lambda * (Q / ADV)
impact_bps           ≈ lambda * (Q / ADV) * 1e4
```
`lambda` is a security-specific Kyle-lambda-like coefficient (here expressed so that `lambda*(Q/ADV)` is in *return* units). Valid only at low participation; **it underestimates badly as Q/ADV grows.**

### 5.2 Square-root law (Almgren et al.)
The widely-used empirical law for the **cost of executing a parent order** of size `Q`:

```
impact (in return units) ≈ Y * sigma * sqrt(Q / ADV)
```

Definitions:
- `sigma` = **daily** return volatility of the asset (decimal, e.g. 0.02 = 2%/day).
- `Q` = order size in **shares** (or notional), `ADV` = **average daily volume** in the same units. `Q/ADV` is the fraction of a day's volume you trade.
- `Y` = dimensionless coefficient of order **O(1)** (commonly ~0.3–1.0 depending on market and calibration) — **must be calibrated to your own fills**, not assumed.

In bps: `impact_bps = Y * sigma * sqrt(Q/ADV) * 1e4`.

Worked example: `Y=0.5`, `sigma=0.02` (2%/day), `Q/ADV=0.10` (trade 10% of a day's volume):
```
impact = 0.5 * 0.02 * sqrt(0.10) = 0.5 * 0.02 * 0.3162 = 0.003162 = 31.6 bps
```
The sqrt is why impact is brutal: trading 10% of ADV costs ~32 bps; trading 1% costs ~10 bps (i.e. 1/10th the size costs ~1/3.16th the bps, because sqrt(10) ≈ 3.16) — but trading 40% costs ~63 bps. Concavity also means *splitting a fixed parent order across N days* lowers total cost: per-day cost scales as sqrt of per-day size, so N equal slices cost `sqrt(N)` times the single-day-of-1/N-size cost, i.e. total cost ∝ `1/sqrt(N)` for a fixed parent — halving daily size by spreading over 2 days cuts total impact cost by a factor of ~`sqrt(2)`, not in half. (Beware: spreading over more days incurs more spread/commission per slice and adds timing risk — that trade-off is the scheduler's job, the next cycle.)

### 5.3 Participation rate (POV) framing
If you trade at participation rate `p = (your volume)/(market volume)` over the execution window, temporary impact rises with `p`. A common form:
```
temporary_impact_bps ≈ eta * sigma * (p)^beta * 1e4,   beta ~ 0.5
```
The practical takeaway for *cost modeling*: cost rises with how aggressively you trade relative to available volume. (Choosing the schedule that minimizes this is the next cycle's topic.)

### 5.4 Impact decay
Temporary impact reverts over a horizon (minutes to a day); permanent impact persists. For multi-day execution and for back-to-back rebalances in the same name, model that today's impact partially decays before tomorrow — otherwise you may either double-charge or ignore residual price displacement. For daily-rebalance backtests, charging total impact per rebalance on the *traded* delta is the standard approximation.

Detect/fix:
- **Assuming linear impact at high participation.** Linear `lambda*(Q/ADV)` understates cost above ~1–2% of ADV. Use square-root.
- **Ignoring impact entirely.** Any strategy whose per-name trade is a non-trivial fraction of ADV (smaller/mid caps, concentrated bets, illiquid futures/crypto pairs) is dominated by impact, not spread/fees.

---

## 6. Shorting & financing costs

Holding costs accrue per unit time on the *position*, independent of trading.

### 6.1 Borrow fee (stock loan)
Quoted as an **annualized** rate on the short market value:
```
borrow_cost_per_day = borrow_rate_annual * |short_mv| / 360   # or /365 by convention
```
Use the day-count your broker uses (US equity stock loan is typically ACT/360); be consistent across the book.
- **General collateral (GC)**: easy-to-borrow, ~0.3–1% annual.
- **Hard-to-borrow (HTB)** / "special": 5% to >50% annual; can spike intraday and the rate is *not* locked — you can be recalled.

Detect/fix: **Underestimating borrow.** Short-side alpha in small/crowded names is frequently entirely consumed by borrow. Use point-in-time borrow rates if available; never assume GC for the whole book. **Look-ahead trap:** do not apply *today's* observed borrow rate to historical shorts unless it is genuinely point-in-time — borrow rates are not in most price feeds and using a current/most-recent rate is leakage. A recalled borrow forces a buy-in at the worst time — model HTB names as higher-cost or excludable.

### 6.2 Margin interest
Long positions bought on margin and the financing leg of leverage cost interest, typically a spread over a reference rate (e.g. SOFR + spread). Charge on the financed notional per day.

### 6.3 Dividends owed on shorts
A short seller **pays** the dividend to the lender (a cost); a long **receives** it. On ex-date, debit `dividend_per_share * short_shares`. For total-return series this is embedded; for price series you must add it explicitly. In some jurisdictions short dividend payments are taxed less favorably ("payment in lieu"), worsening the cost. **Look-ahead trap:** dividend amounts and ex-dates must be applied on the correct ex-date with point-in-time data — using a later-announced/adjusted dividend or a back-adjusted total-return series alongside an explicit dividend debit double-counts.

---

## 7. Perpetual funding (crypto)

Crypto perpetual swaps have no expiry; the **funding rate** tethers the perp to spot. It is exchanged **between longs and shorts** every interval (commonly every 8h; some venues 1h or 4h):

```
funding_payment = funding_rate * position_notional      # per interval, signed
```

- `funding_rate > 0` (perp above spot, typical in bull markets): **longs pay shorts**.
- `funding_rate < 0`: **shorts pay longs**.

This is a *carry*, not a trading cost: a market-neutral basis trade (long spot / short perp) *earns* funding when it's positive. Annualize to compare: `funding_annual ≈ funding_rate * intervals_per_day * 365`. A 0.01%/8h rate ≈ 0.0001 × 3 × 365 ≈ 0.1095 ≈ 10.95%/yr.

Detect/fix: **Forgetting funding** turns a directional perp strategy's P&L meaningfully wrong over weeks; for carry strategies, funding *is* the entire P&L. Charge/credit it every interval the position is held, using the funding rate **published for that interval applied to the position held going into the interval** — using the next interval's (not-yet-known) rate, or the period-average over the holding window, is look-ahead.

---

## 8. FX rollover & futures roll

### 8.1 FX swap points / rollover (tom-next)
Holding a spot FX position past settlement incurs the **interest rate differential** between the two currencies, realized as swap points / rollover credited or debited daily (commonly triple on the day covering the weekend, e.g. Wednesday for many spot conventions). Long the higher-yielding currency → you typically *receive*; long the lower-yielder → you *pay*. This is the FX carry.

### 8.2 Futures roll
Futures expire; to maintain exposure you **roll** to the next contract. The roll incurs:
- **Two spreads/commissions** (close near, open far) — a trading cost.
- The **calendar spread** level (contango/backwardation) — an embedded carry, not a transaction cost per se, but it shows up as a drag/boost in a continuous (back-adjusted) series.

Detect/fix: With back-adjusted continuous futures, the roll's calendar-spread effect is already in the price series — do **not** separately charge it as a cost, or you double-count. Do still charge the roll's *spread + commission* (the actual transaction). Conversely, with a raw front-month splice that ignores roll gaps, you *must* add the roll cost manually. **Look-ahead trap:** choose the roll date from a fixed ex-ante rule (e.g. N days before expiry, or first day open-interest in the back month exceeds the front) using only data available on the roll date — do not pick the roll date that minimizes the realized gap in hindsight.

---

## 9. Capacity: how cost caps AUM

Capacity is the AUM at which marginal alpha = marginal cost. Because impact ∝ sqrt(size), per-trade cost grows with AUM while alpha per dollar is roughly fixed → net return falls and eventually goes negative.

Two practical bounds:

1. **ADV participation cap.** Limit each name's trade to a max fraction `f` of its ADV (commonly 1–10%). For a name traded with turnover `|Δw|` per rebalance and dollar-ADV `A`:
   ```
   max_AUM_name = f * A / |Δw|
   ```
   (i.e. the per-rebalance dollar trade `|Δw| * AUM` must not exceed `f * A`.) Portfolio capacity is bounded by the most-constrained names (often the small/illiquid alpha-rich ones).

2. **Cost drag = turnover × cost.** Annual cost drag on returns:
   ```
   annual_cost_drag = annual_one_way_turnover * one_way_cost_fraction
   ```
   e.g. 1200% annual one-way turnover (12× the book/yr, one-way) × 15 bps one-way = 12 × 0.0015 ≈ 180 bps/yr drag. Keep turnover and cost on the **same basis** (both one-way, or both round-trip) — mixing one-way turnover with a round-trip cost double-counts. Since one-way cost itself rises with size (sqrt law), drag accelerates with AUM.

Break-even AUM: find AUM where net Sharpe (or net return) crosses your hurdle. Plot net return vs AUM — it's typically hump-shaped; the peak is the optimal book size, and you typically run *below* it for safety margin.

Detect/fix: **Reporting capacity at backtest AUM only.** Always show the net-return-vs-AUM curve. A strategy that's great at \$10M and dead at \$500M is a different product than the backtest implies.

---

## 10. Modeling costs in a backtest

The core identity: **cost is charged on what you trade, not what you hold.**

```python
import numpy as np

# weights: target portfolio weights, DataFrame indexed by date, columns=assets.
#          Already point-in-time / lagged (decided using info available at t-1 or earlier).
# All per-bar quantities (sigma, ADV, half_spread_bps, explicit_bps) must themselves
# be point-in-time — i.e. known as of the rebalance date, never forward-filled from the future.

# --- turnover (one-way traded fraction per name per date) ---
prev = weights.shift(1).fillna(0.0)      # row 0: trade from 0 -> initial weights
trades = (weights - prev).abs()          # |Δw|, one-way; row 0 charges the initial build

# --- impact needs Q/ADV in a COMMON unit. Convert weight-space turnover to traded
#     notional, then to a fraction of dollar-ADV (dollar_adv aligned to weights' shape) ---
trade_notional = trades * AUM                 # $ traded per name per date
participation  = trade_notional / dollar_adv  # Q/ADV, dimensionless (same units top & bottom)

# --- one-way cost rate per name in RETURN units (sum the relevant components) ---
cost_rate = (
    explicit_bps                                  # per-name, side-aware where needed
    + half_spread_bps                             # one-way = half spread
    + Y * sigma * np.sqrt(participation) * 1e4    # square-root impact, bps
) / 1e4                                            # bps -> return units

cost = (trades * cost_rate).sum(axis=1)            # portfolio trading cost per date, return units

# --- P&L: positions lagged vs the returns they earn; subtract trading cost on the trade date ---
gross_pnl = (weights.shift(1) * returns).sum(axis=1)
net_pnl   = gross_pnl - cost

# --- holding costs (borrow/funding/margin) on positions HELD over the period, per period ---
# Pass rates already on a PER-PERIOD basis matching the bar (e.g. borrow_rate_daily = annual/360).
held = weights.shift(1)                            # position carried into the period
borrow_leg  = held.clip(upper=0).abs() * borrow_rate_period   # shorts pay borrow (>=0 cost)
funding_leg = held * funding_rate_period                       # signed: long pays if rate>0
holding_cost = (borrow_leg + funding_leg).sum(axis=1)
net_pnl = net_pnl - holding_cost
```

Rules:
- **Charge on turnover (`|Δw|`), once per rebalance.** Holding a position incurs *no* trading cost — only the *change* does.
- **One-way vs round-trip.** Each `|Δw|` is a one-way trade; a position opened and later closed is charged on both legs as they occur. Don't pre-multiply by 2 *and* also charge both legs.
- **Do not net across names.** Buying \$1M of A and selling \$1M of B is \$2M of trading and two sets of costs — not zero. Sum `|Δw|` across names; never net signed trades into a smaller number before costing. (Within the *same* name across a rebalance, the net change *is* the trade — that netting is correct.)
- **Impact uses per-name trade size vs that name's ADV**, not portfolio-level size, and `Q` and `ADV` must be in the same unit. A small portfolio trade concentrated in one illiquid name has large impact.
- **Lag correctly.** Costs hit on the date you trade; the traded position earns returns from the *next* period (`weights.shift(1)` against `returns`), consistent with the house P&L convention. Holding costs also accrue on the lagged (held) position, not the target.
- **Side-aware explicit costs.** Stamp duty / FTT / SEC fees are one-sided — charge them on the signed buy (or sell) leg, not on `|Δw|`.

Detect/fix:
- **Same-bar fills (look-ahead).** Computing the signal from the close and filling at that same close (or worse, using a close the signal peeked at) is look-ahead disguised as zero slippage. Fill at next bar's open/VWAP, or charge realistic slippage for a same-bar marketable fill — and always lag the position.
- **Forward-filled cost inputs (look-ahead).** Using a security's full-sample average ADV/sigma/spread (which embeds the future) instead of a trailing, as-of estimate inflates capacity and understates cost. Use only data available at the decision time.
- **Costing held notional instead of traded notional** turns a buy-and-hold into a bleeding mess and a high-turnover strategy into something too cheap. Cost the *diff*, not the level.

---

## 11. Calibration from fills & sensitivity analysis

### 11.1 Implementation shortfall (the ground truth)
Calibrate models against **realized** cost, measured as implementation shortfall vs the **arrival (decision) price**:

```
IS_bps = D * (avg_fill_price - arrival_mid) / arrival_mid * 1e4 + explicit_bps
```
`D = +1` buy / `−1` sell. IS captures spread + impact + timing slippage in one number, benchmarked to the price *when you decided to trade* (not VWAP, which flatters you, and not the close, which can leak look-ahead). Note IS already bundles spread + impact + timing — when you fit it, fit *those components against it*; do not then add an independent parametric spread/impact term on top of a model already calibrated to IS.

Calibrate `Y` (square-root) and spread terms by regressing realized IS on `sigma * sqrt(Q/ADV)` and on `half_spread`:
```
IS_bps ≈ b0 + b1 * half_spread_bps + Y_hat * sigma * sqrt(Q/ADV) * 1e4
```
Use a robust fit (outliers from news/halts are common); segment by asset class, liquidity bucket, and venue. Re-calibrate periodically — costs drift with regime. Calibrating on the same period you backtest is a mild in-sample optimism; prefer out-of-sample or rolling calibration.

### 11.2 Sensitivity & break-even
Even without fills, *bound* the answer:

- **Sweep cost.** Re-run the backtest at cost = 0, 5, 10, 20, 50 bps round-trip. Report net Sharpe vs cost. A strategy alive only at <5 bps is fragile.
- **Break-even cost.** Solve for the cost at which net return hits zero / your hurdle. Keep turnover and cost on the **same basis**:
  ```
  breakeven_one_way_cost ≈ gross_return_per_period / one_way_turnover_per_period
  ```
  (If you quote a round-trip break-even, divide one-way turnover accordingly so bases match.) Compare to your realistic cost estimate. If break-even cost ≈ your estimated cost, you have no margin — treat as unviable.
- **Turnover attribution.** Decompose drag into spread, impact, fees, financing. If impact dominates, the strategy is capacity-bound (trade slower / smaller / fewer names); if fees dominate, renegotiate or reduce trade count; if financing dominates, the short book or leverage is the problem.

---

## 12. Rough cross-asset cost ranges (round-trip, liquid names, modest size)

Indicative only — calibrate to your own fills. Wider for small/illiquid names and large size. Rates change; verify current schedules.

| Asset class | Commission/fees | Spread (round trip) | Notes |
|---|---|---|---|
| US large-cap equity | ~0.1–1 bp (per-share) | ~1–5 bps | + SEC/FINRA (sells), borrow on shorts; stamp duty in UK/EU |
| US small-cap equity | ~1–3 bps | ~10–50+ bps | Impact dominates; HTB borrow common |
| Equity index futures (ES, etc.) | <0.5 bp | ~0.5–2 bps | Very liquid; roll 4×/yr |
| Major FX spot (EURUSD) | ~0–0.2 bp | ~0.1–1 bp | Rollover/swap = carry; venue-dependent |
| EM / minor FX | ~1–3 bps | ~2–20 bps | Wider, jumpier |
| Crypto spot (BTC/ETH, top venue) | ~1–7 bps taker / rebate maker | ~1–5 bps | 24/7; venue/tier dependent |
| Crypto perps | ~2–6 bps taker | ~1–10 bps | + funding every 1–8h (can dominate) |
| Liquid options | per-contract + exch fees | spread can be 1–10%+ of premium | Wide spreads; price in vega/gamma, not just notional |

UK equity buys carry **+50 bps stamp duty** (one-sided, buys only) on top of the above — by far the largest single line for UK strategies.

---

## 13. Pitfalls (detect / fix)

| Pitfall | Symptom | Detect | Fix |
|---|---|---|---|
| Ignoring spread/impact | Backtest Sharpe collapses live | Cost-sweep shows Sharpe cliff near 0 bps | Charge half-spread + square-root impact on turnover |
| Quoted instead of effective spread | Costs off by ~1.5–2× (either way) | Compare quoted vs effective from fills | Calibrate effective spread; for size, model book-walking |
| Effective-spread 2× confusion | Spread cost off by exactly 2× | Check whether the ×2 (full-spread basis) is applied twice | One-way = half-spread; apply ×2 only to match quoted basis |
| Underestimating / look-ahead borrow | Short-side P&L good in backtest, bad live | Compare assumed vs PIT borrow rates; check rate is as-of | Use PIT borrow; flag/exclude HTB; model recalls |
| Same-bar fills (look-ahead) | Suspiciously smooth, high Sharpe | Check fill timestamp vs signal timestamp | Fill next bar (open/VWAP); lag positions; add slippage |
| Forward-filled ADV/sigma/spread (look-ahead) | Capacity too high, cost too low | Compare full-sample vs trailing inputs | Use trailing, as-of cost inputs only |
| Forgetting funding (perps) | Multi-week perp P&L drifts wrong | Reconcile vs exchange funding history | Credit/debit funding every interval on held notional, this-interval rate |
| Linear impact at high participation | Capacity overstated; large fills bleed | Q/ADV > ~2% with linear model | Use square-root law; calibrate Y from fills |
| Mixed units in Q/ADV | Impact wildly off | Check Q and ADV same unit; weight-space vs share-space | Convert turnover to notional before Q/ADV |
| Netting costs across names | Total cost too low | Sum of \|Δw\| vs costed amount mismatch | Cost on Σ\|Δw\|, never on signed net |
| Costing held vs traded notional | Buy-and-hold looks terrible / churner looks cheap | Cost scales with positions not trades | Charge on `weights.diff().abs()` |
| Mixed turnover/cost basis | Drag or break-even off by 2× | One-way turnover × round-trip cost | Keep both one-way or both round-trip |
| Double-counting one-sided tax | UK/FTT charged round-trip or on \|Δw\| | Tax on both legs | Charge stamp/FTT on the buy leg only |
| Roll cost double-count (futures) | Continuous series + manual roll carry | Both back-adjustment and roll carry applied | Charge only roll spread+commission; let back-adjust handle calendar |
| Maker rebate without fill model | Free money in backtest | Rebate assumed + guaranteed fills | Use taker pricing unless modeling queue/fills |

---

## 14. Templates

- **`templates/costs.py`** — cost-model functions: explicit/spread/slippage/square-root impact, borrow & funding accrual, effective-spread and implementation-shortfall calibration, break-even and cost-sweep helpers.
- **`templates/backtest_skeleton.py`** — applies costs on turnover per rebalance with correctly-lagged positions (`pnl_t = pos.shift(1)*ret_t`), separating trading cost from holding (financing/funding) cost, with point-in-time cost inputs.

See also `metrics.py` (net Sharpe), `validation.py` (purge/embargo so cost-laden P&L isn't also look-ahead-contaminated), and `factor_research.py` (turnover/IC trade-off when ranking factors).