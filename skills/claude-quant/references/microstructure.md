# Market Microstructure & Optimal Execution

How orders interact with the order book, how prices form from order flow, and how to schedule a large trade. This file is about **mechanics and trajectories**. For modeling cost *magnitudes* (spread, commission, borrow, slippage models you plug into a backtest) see `references/transaction-costs.md`; the runnable cost/impact functions live in `templates/costs.py`. The runnable, self-tested TWAP/VWAP/POV/IS schedulers and the Almgren-Chriss trajectory live in `templates/execution.py`.

Conventions used here: prices in absolute terms unless noted; `S` = mid, `s` = half-spread, `X` = total shares to execute (signed; buy `>0`), `T` = horizon, `sigma` = per-unit-time volatility of price (price units, not return units), `Q` = inventory/size. Implementation shortfall is signed so that **positive = you paid more than the benchmark** (a cost). Note: A-C's risk-aversion `lambda`, Kyle's impact `lambda`, and Avellaneda-Stoikov's `gamma` are three different parameters — they are flagged at each use.

---

## 1. Order book mechanics

A central limit order book (CLOB) is a sorted, two-sided queue of resting **limit orders**:

- **Bid side**: buyers willing to pay up to a price, sorted descending. Top = **best bid** (highest bid).
- **Ask/offer side**: sellers, sorted ascending. Top = **best ask** (lowest ask).
- **Mid** `S = (bid + ask)/2`; **quoted spread** `= ask - bid`; **microprice** (size-weighted) `= (bid*ask_size + ask*bid_size)/(bid_size + ask_size)`. Note the cross-weighting: the bid is weighted by *ask* size and vice versa, so a heavy bid (large `bid_size`) pulls the microprice **toward the ask** — i.e. toward the side likely to be hit next. It is a better short-horizon fair-value estimate than the mid.

```
       price   bid_size | ask_size
       100.03                  900
       100.02                  400   <- best ask (lowest ask = top of ask side)
       ---- spread = 0.02, mid = 100.01 ----
       100.00     600                <- best bid (highest bid = top of bid side)
        99.99     250
        99.98    1200
```

(Best ask is the *lowest* ask price, 100.02, not 100.03. Asks sort ascending; the cheapest offer is at the top of the book.)

**Limit vs market orders.**
- A **limit order** posts liquidity ("maker"): names a price, joins the queue, fills only if the market comes to it. You may capture the spread / a rebate but bear *non-execution risk* and *adverse selection* (you tend to get filled exactly when the price is about to move against you — see §2).
- A **market order** takes liquidity ("taker"): executes immediately against resting orders, walking the book until filled. You pay the spread (and impact for size) but get fill certainty.

**Price-time priority (FIFO).** Most equity/futures venues match by (1) best price, then (2) time of arrival at that price. Your fill probability for a posted order depends on **queue position**: how much size sits ahead of you at your price level. Some venues (esp. options, some futures, and certain rate products) use **pro-rata** or **size-time-pro-rata** matching, where large orders get a proportional share of incoming flow regardless of arrival time — this changes optimal order-sizing behavior (under pro-rata you may *oversize* to win a larger allocation, the opposite of FIFO incentives).

**Why queue position matters.** Two traders posting at the same price are not equal. The one at the front fills first and, crucially, can **cancel and step away** when the queue ahead of them evaporates (often a sign the price is about to trade through). Models of fill probability (e.g. Cont-Stoikov-Talreja) treat the book as a queueing system: your fill probability falls as queue-ahead size rises and as the cancellation rate ahead of you rises. Queue value is why latency and early posting matter even for non-HFT participants.

**Order types.**

| Type | Behavior | Use when |
|---|---|---|
| Limit | Rest at a price; maker | You want price control / to capture spread |
| Market | Take immediately; walks book | You need certainty now |
| IOC (immediate-or-cancel) | Fill what you can now, cancel rest | Sweep available liquidity without leaving a resting footprint |
| FOK (fill-or-kill) | All-or-nothing, immediately | You need the full size or none (e.g. legging risk) |
| Iceberg / reserve | Shows small "tip", hides the bulk; replenishes on fill | Hide size; reduce signaling (note: detectable via replenishment timing) |
| Pegged (mid/primary/market) | Auto-reprices relative to a reference (mid, same-side BBO) | Stay passive at a relative price without manual re-quoting |
| Stop / stop-limit | Dormant until trigger price; then becomes market/limit | Risk exit / breakout entry (warning: stop-markets can cascade; stop-limits carry gap/non-fill risk) |

Asset-class notes: crypto books are fragmented across venues, run 24/7, and post-only / maker-rebate semantics vary; FX spot is largely bilateral/RFQ with "last look"; listed options books are thin and often pro-rata.

---

## 2. The bid-ask spread, decomposed

The quoted spread compensates the liquidity provider for three distinct costs:

1. **Order-processing** — fixed costs of quoting/clearing; roughly constant per trade.
2. **Inventory** — the maker accumulates a position away from target and bears price risk while unwinding; widens with volatility and with the size of the maker's current inventory (see Avellaneda-Stoikov, §5).
3. **Adverse selection** — counterparties may be informed. In **Glosten-Milgrom (1985)**, the maker faces a mix of informed and noise traders; each trade is a Bayesian signal, so the maker must quote a spread even with zero processing/inventory cost just to break even against informed flow. In **Kyle (1985)**, a single informed trader hides among noise traders; the market maker sets a linear price schedule and market depth is `1/lambda` (see §3). Both deliver the same intuition: **spread exists because trades carry information.**

**Three empirical spread measures** (don't conflate them — `D = +1` for a buy, `-1` for a sell):

- **Quoted spread** `= ask - bid`. What you see; an upper bound on the *spread* cost for a marketable order no larger than top-of-book size (you'd cross at most a half-spread). For larger orders you walk the book and pay more than the quoted spread.
- **Effective spread** `= 2 * D * (P_exec - S_mid)` where `S_mid` is the mid *at order arrival*. Measures what you **actually paid** relative to mid (captures price improvement and book-walking). Half of it, `D*(P_exec - S_mid)`, is the per-share effective cost.
- **Realized spread** `= 2 * D * (P_exec - S_mid(t+Δ))` using the mid a short interval Δ later (commonly 5 min). Measures the maker's **realized revenue** after the price moves following the trade.

The decomposition closes as:

```
effective spread  =  realized spread  +  price impact
(what taker pays)    (maker keeps)       (information, = 2*D*(S_mid(t+Δ) - S_mid(t)))
```

Check: `realized + impact = 2D(P_exec - S_mid(t+Δ)) + 2D(S_mid(t+Δ) - S_mid(t)) = 2D(P_exec - S_mid(t)) = effective`. A large impact share relative to realized spread = adverse-selection-heavy ("toxic") flow.

**Detect/fix:** if your TCA uses the quoted spread as the cost of a marketable order, you will mis-cost any order larger than top-of-book depth (you walk the book) and ignore price improvement on small orders. Use the effective spread against the **arrival mid**, not the touch you traded at.

---

## 3. Price impact

The price you move by trading. Two components with different lifetimes:

- **Temporary impact** — the transient cost of demanding immediacy (walking the book, paying the spread). Decays as the book refills ("resilience"). Depends primarily on your *rate* of trading.
- **Permanent impact** — the lasting shift in fair value because the market infers information from your flow. Depends on *cumulative size* traded. Does not decay.

**Kyle's lambda (linear impact).** In Kyle's model the equilibrium price schedule is linear in net order flow:

```
ΔS = lambda * Q          (lambda = price move per unit signed volume)
```

`lambda` is the **inverse of market depth**: `1/lambda` shares move the price one unit. Estimate it by regressing signed mid-price changes on signed order flow (order-flow imbalance) over short windows. High `lambda` = illiquid / informative flow. This linear law underlies Almgren-Chriss's permanent-impact term `g(v)=γv`. (Distinct from the empirical square-root law below, which describes *total* metaorder shortfall, not the instantaneous schedule.)

**The square-root law (metaorders).** Empirically, the *total* implementation shortfall of executing a large "metaorder" of size `Q` over roughly a day scales as:

```
impact  ≈  Y * sigma_daily * sqrt(Q / V_daily)
```

where `sigma_daily` is daily volatility (return units), `V_daily` is daily volume (same units as `Q`), and `Y` is an O(1) prefactor (~0.3–1 across markets, roughly asset-class-stable). Key consequences:
- Impact grows **sub-linearly** (sqrt of size), so doubling size raises *total* cost by only ~41% (`sqrt(2)≈1.41`), not 100% — but per-share cost still rises with size.
- It depends on size **relative to volume** (participation `Q/V`), not absolute shares. The same 100k shares is cheap in a liquid name and ruinous in a thin one.

This is exactly the form implemented in `square_root_impact(order_size, adv, daily_vol, coef)` in `templates/costs.py` (`impact = coef * daily_vol * sqrt(order_size/adv)`); `coef` plays the role of `Y`. **Units must match**: `order_size` and `adv` in the same units (both shares or both notional), and `daily_vol` is a return fraction so the output is a return fraction of notional.

**Impact decay / resilience.** After a metaorder finishes, the price partially **reverts** as the temporary component decays, settling at a permanent level. Empirically that permanent level is often around two-thirds of the *peak* impact (i.e. ~1/3 reverts), though this varies by market and execution speed. Resilience = the speed at which the book replenishes. Trading faster than the book can refill pushes you up a depleted book and raises temporary cost super-linearly in the trade rate.

**Detect/fix:** a backtest that fills at mid (or a fixed bps) regardless of size has **no size penalty** — it will love large, concentrated trades that are uneconomic live. Add at minimum a sqrt-law shortfall term scaled by participation (`square_root_impact` in `templates/costs.py`), and stress-test capacity by scaling AUM until the sqrt cost eats the edge (`apply_costs` / `breakeven_cost_bps` in the same file).

---

## 4. Price formation & information

Prices form from the interaction of two stylized populations:

- **Informed traders** trade in the direction of future value; their flow is autocorrelated and predictive.
- **Noise / liquidity traders** trade for exogenous reasons (rebalancing, hedging, cash needs); their flow is roughly unpredictable.

The market maker can't tell them apart trade-by-trade, so every fill carries **adverse selection**: conditional on being hit, the counterparty is slightly more likely to be informed, so the maker loses in expectation on that print and recoups via the spread on the noise traders. This is the engine of price discovery — quotes ratchet toward fair value as the book absorbs informed flow.

**Toxic flow & VPIN (intuition).** "Toxicity" = the fraction of flow that is informed / adverse to liquidity providers. **VPIN** (Volume-Synchronized Probability of Informed Trading) estimates it without needing exact trade-direction labels:
1. Bucket trades into equal-**volume** bars (not equal-time — this adapts to activity).
2. Within each bar, split volume into buy vs sell pressure (e.g. bulk-volume classification: `V_buy = V * Z(ΔP/σ_ΔP)` with `Z` a standardized CDF, the rest is sell).
3. `VPIN ≈ mean( |V_buy - V_sell| / (V_buy + V_sell) )` over a rolling window of buckets.

High VPIN = order flow is one-sided and likely informed; makers widen or pull quotes, raising the cost of taking. Useful as an **execution gate**: pause or slow a passive execution when toxicity spikes (you're more likely to be picked off). Caveats: VPIN is sensitive to bucket size and the classification rule; its forecasting claims (e.g. around the 2010 Flash Crash) are disputed in the literature — treat it as a regime/toxicity indicator, not a calibrated probability.

---

## 5. Market making

A market maker quotes a bid and an ask, earns the spread on round trips, and manages **inventory risk**: an accumulated position exposes them to price moves before they can flatten.

**Avellaneda-Stoikov (2008) optimal quotes.** Maximize exponential utility (risk aversion `gamma`) of terminal wealth over `[0,T]` with mid `S`, volatility `sigma`, inventory `q`, and order-arrival intensity `lambda(δ) = A * exp(-k*δ)` (fill rate falls with distance `δ` from the mid; `k` is the depth-sensitivity of arrivals). Two results:

**Reservation (indifference) price** — the maker's inventory-adjusted fair value:

```
r(S, q, t) = S - q * gamma * sigma^2 * (T - t)
```

If long (`q>0`), `r < S`: skew quotes **down** to encourage sells and discourage buys, shedding inventory. The skew grows with inventory, risk aversion, variance, and time remaining.

**Optimal total spread** around the reservation price:

```
δ_total = gamma * sigma^2 * (T - t)  +  (2/gamma) * ln(1 + gamma/k)
```

Quote bid `= r - δ_total/2`, ask `= r + δ_total/2`. First term = inventory/risk component (grows with `gamma`, `sigma`, time left); second = the fill-intensity / market-structure component. The quotes are **symmetric about the reservation price `r` but asymmetric about the mid `S`** (because `r ≠ S` whenever `q ≠ 0`) — that asymmetry is the inventory hedge.

Behavior: as `t -> T`, the inventory term vanishes and the maker quotes tightly to flatten; high `gamma` widens spreads and skews aggressively to stay near flat. (Note: this is the classic A-S approximation; the exact solution and common variants, e.g. Guéant-Lehalle-Fernandez-Tapia, refine the spread term, but the intuition is identical.)

**Make vs take.** Posting (make) earns the spread/rebate but risks non-execution and adverse selection; crossing (take) guarantees execution but pays the spread. Post when: your edge horizon is long relative to expected fill time, queue position is good, toxicity (VPIN) is low, and the price is not moving away from you fast. Take when: you have short-lived alpha, you're behind in a trend, or non-execution risk dominates (see §6 pitfall on naive TWAP). This mirrors the maker-rebate caveat in `references/transaction-costs.md` §2.2: you cannot assume the maker rebate *and* a guaranteed fill.

---

## 6. Execution algorithms

Schedule a parent order into child orders over time. The four workhorses (all are straightforward to implement from the formulas here; there is no prebuilt scheduler module in this repo):

**TWAP — Time-Weighted Average Price.** Slice evenly across time: `x_i = X / N` per interval. Simple, predictable, benchmark-agnostic. Pro: trivial, low signaling if intervals/sizes are randomized. Con: ignores intraday volume seasonality (over-participates in quiet periods); a **clockwork schedule is gameable**; bad in trending markets (uniform pace = high timing risk if the price runs away).

**VWAP — Volume-Weighted Average Price.** Trade in proportion to the (forecast) intraday volume curve so your average price tracks the day's VWAP: `x_i ∝ Ŷ_i` (expected volume share of bucket `i`, typically the U-shaped open/close-heavy profile). Pro: minimizes tracking error to the most common institutional benchmark; blends in with natural volume so lower relative impact. Con: depends on a volume **forecast** (errors → tracking error); chases volume even when the price is unfavorable; gameable (see pitfalls).

**POV / Participation.** Trade a fixed fraction `rho` of *realized* volume in real time: `x_i = rho * V_i`. Pro: self-adjusts to actual liquidity, caps your footprint at a known participation rate. Con: completion time is **uncertain** (depends on market volume); the constant participation rate can be detected and gamed; may not finish if volume dries up.

**Implementation Shortfall (IS) algos.** Optimize against the **arrival price** (decision price), explicitly trading impact vs timing risk — i.e. they solve an Almgren-Chriss-style problem (§7) and typically **front-load** (trade faster early) because the unexecuted remainder carries volatility risk. Pro: directly minimizes the cost the PM actually cares about (slippage vs decision). Con: more aggressive early (more impact); needs good impact/vol estimates; noisier vs simple VWAP on any single order.

**Which when:**
- Benchmarked to VWAP / want to "blend in": **VWAP**.
- Capacity-constrained / want a liquidity cap and don't care exactly when it finishes: **POV**.
- You have a price view / short-lived alpha / care about slippage-vs-decision: **IS (front-loaded)**.
- No view, want simplicity and you're a small fraction of volume: **TWAP** (randomize intervals).

---

## 7. Almgren-Chriss optimal execution

The canonical framework for the **impact-vs-risk trade-off**. Liquidate `X` over `[0,T]` in `N` steps of length `tau = T/N`. Two costs pull in opposite directions:

- Trade **fast** → high **market-impact cost** (you eat the book) but low **timing risk** (little time exposed to price moves).
- Trade **slow** → low impact but high timing risk (volatility of the unexecuted inventory).

**Cost model.** With permanent impact `g(v)=γ*v` and temporary impact `h(v)= ε*sgn(v) + η*v` (per-share; `v` = trade rate; `ε` = fixed per-share crossing cost that carries the sign of the trade, `η` = linear temporary coefficient), for a holdings path `x(t)` (shares remaining):

```
E[C]   = permanent + temporary impact integrated over the schedule
Var[C] = sigma^2 * ∫ x(t)^2 dt        (price-risk of the remaining inventory)
```

Minimize the mean-variance objective `E[C] + lambda_AC * Var[C]`, where **`lambda_AC` is the A-C risk-aversion** (units: 1/$ — the trader's penalty on execution-cost variance; this is *not* Sharpe's lambda and *not* Kyle's impact lambda).

**Closed-form optimal trajectory** (continuous limit, holdings remaining at time `t`):

```
x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)

with    kappa^2 = lambda_AC * sigma^2 / eta_tilde
        eta_tilde = eta - gamma*tau/2        (≈ eta for small step tau)
```

`kappa` (1/time) sets the **decay rate** of the holdings curve — the "urgency". The trade rate is the (negative) time-derivative of `x(t)`; the holdings path is **convex and front-loaded** for `lambda_AC > 0`.

**Risk-aversion limits:**
- `lambda_AC -> 0` (risk-neutral): `kappa -> 0`, and `sinh(kappa(T-t))/sinh(kappa T) -> (T-t)/T`. Holdings decline **linearly** → constant trade rate = **TWAP**. Pure impact minimization, no urgency.
- **large `lambda_AC`** (risk-averse): large `kappa`, `x(t)` decays quickly → **front-load**, shed most of the position early to cut timing risk, accepting higher impact.

**Efficient frontier of execution.** Sweeping `lambda_AC` traces a frontier in `(E[C], Var[C])` space: each point is the minimum expected cost for a given variance tolerance (or vice versa). You pick the point matching your risk appetite — exactly analogous to the Markowitz frontier, but for a single trade's execution rather than a portfolio. Fast strategies sit at high-cost/low-variance; slow ones at low-cost/high-variance.

Extensions worth knowing: nonlinear (sqrt-law, §3) temporary impact instead of linear `η*v` (no longer closed-form — solve numerically); transient/decaying impact (Obizhaeva-Wang) which favors smoother schedules; and an **alpha-drift** term, which tilts the schedule to trade faster when your signal is decaying. Implement the trajectory directly from the closed form above; the sqrt cost term for capacity work is in `templates/costs.py`.

---

## 8. Transaction cost analysis (TCA)

Measure realized execution quality against a benchmark and attribute the cost to sources you can act on.

**Benchmarks (pick deliberately):**
- **Arrival price** (decision/arrival mid): the mid when you decided to trade. The honest benchmark for IS — it can't be gamed by your own trading.
- **Interval VWAP** (over the trading window): easy to "beat" by trading passively, and partly *self-referential* (your own fills help set it). Good for "did I trade in line with the market", weak as a pure quality measure.
- **TWAP / close / open**: situational.

**Implementation Shortfall** = the all-in gap between the paper portfolio (filled instantly at the decision price, frictionless) and the real portfolio. Signed so positive = cost. A standard (Perold) decomposition for a buy of `X` shares — decision price `S_decision`, arrival price `S_arrival` (when the order reaches the market), average exec `S_exec`, final/cancel price `S_end`, with `x_done` of `X` filled:

```
IS  =  delay cost        (S_arrival - S_decision) on the full order X   [latency before you started]
    +  execution/impact   (S_exec   - S_arrival)  on x_done             [what trading cost vs arrival]
    +  opportunity/timing  (S_end    - S_arrival)  on (X - x_done)       [unfilled shares that ran away]
    +  fees & commissions  (explicit)
```

(For a sell, flip the sign of each price-difference term, or multiply the whole thing by `D = -1`.) Each term tells you a different fix: large **delay** → routing / decision-to-order latency; large **impact** → you traded too aggressively / too large a share; large **timing/opportunity** → too passive, missed fills; large **fees** → venue/routing economics. Report IS in **bps of notional**, signed, with slippage-vs-arrival as the headline; show VWAP slippage as secondary.

**Detect/fix:** benchmarking only against interval VWAP hides the cost of *not trading* (an order you canceled when the price ran looks great vs VWAP but cost you the alpha). Always compute arrival-price IS including the opportunity term on unfilled shares.

---

## 9. Smart order routing, venues, and latency (brief)

**Smart order routing (SOR).** When liquidity is fragmented across venues (lit exchanges, dark pools, ATSs; many crypto exchanges), an SOR splits/sequences child orders to capture the best aggregate price net of fees/rebates, posts to maximize fill probability, and probes dark venues to source hidden size without signaling. Key trade-offs: **fee/rebate optimization** (maker rebates vs taker fees — "maker-taker" can bias routing toward rebate-capture at the cost of fill quality), **adverse selection in dark pools** (cheap mid-fills but possibly toxic counterparties), and **information leakage** from spraying many venues (footprint detectable). Routing belongs in TCA: attribute fills by venue and compare effective + realized spread per venue.

**HFT / latency / co-location (brief).** At short horizons speed is a real edge: co-located servers, kernel-bypass networking, and direct (not consolidated) feeds shave microseconds. This matters even if you're not an HFT because (a) your resting orders can be **picked off** on stale quotes (latency adverse selection), and (b) **queue position** (§1) is won by being early. Practical implications for a systematic trader: don't assume your posted order fills the instant the print touches your price; model fill probability via queue position; and treat "last look" (FX) and quote fade as real non-execution risk.

---

## Pitfalls (detect / fix)

| Pitfall | Detect | Fix |
|---|---|---|
| **Ignoring queue position** | Backtest assumes every posted order at the touch fills; live fill rates far lower | Model fill prob as a function of queue-ahead size and cancellation rate; only count posted fills when price trades *through* your level or queue-ahead clears |
| **Assuming mid-price fills** | PnL uses mid for entries/exits; live slippage consistently negative vs backtest | Fill marketable orders at the far touch + book-walk; passive fills at your posted price only when crossed; never at mid |
| **Crossing the spread when you could post** | Every child order is a market order; effective-spread cost ≈ full quoted spread every time | Use passive/IOC posting when alpha horizon > expected fill time and toxicity is low; reserve taking for short-lived alpha or trend-chasing |
| **Naive TWAP in a trending market** | Uniform schedule; large negative timing cost when price trends against you | Front-load (IS / Almgren-Chriss with `lambda_AC>0`) or add an alpha-drift term; randomize intervals to avoid being gamed |
| **Underestimating impact for large size** | Cost model is flat bps or fills at mid; strategy capacity looks unlimited; loves concentrated trades | Add sqrt-law shortfall scaled by participation `Q/V` (`square_root_impact`, `templates/costs.py`); stress capacity by scaling AUM until impact eats the edge |
| **VWAP gaming / self-referential benchmark** | Algo "beats VWAP" but you are a large share of that VWAP; arrival-price IS is poor | Benchmark against **arrival price**, not interval VWAP; cap participation; include opportunity cost on unfilled shares in TCA |
| **Predictable / detectable schedule** | Constant POV or clockwork TWAP; adverse price drift right before each child | Randomize slice sizes and timing; use icebergs/dark venues; vary participation |
| **Forecast-error blind spot (VWAP/POV)** | Tracking error blamed on "the market"; really the volume forecast was wrong | Backtest the **volume forecast** separately; use adaptive POV against realized volume; widen completion-time tolerance |
| **Maker rebate + guaranteed fill** | Backtest books the rebate while assuming every resting order fills | A resting order faces non-fill and adverse selection — model fill uncertainty or default to taker pricing (`references/transaction-costs.md` §2.2) |

---

## See also

- `templates/costs.py` — cost/impact functions used here: `square_root_impact`, `linear_impact`, `slippage_total`, `apply_costs`, `breakeven_cost_bps`.
- `references/transaction-costs.md` — cost *magnitude* models (spread, commission, borrow, slippage) for backtests; complements the *mechanics* here. The sqrt-law there is used only as a cost estimator, not a scheduler.
- `templates/backtest_skeleton.py` — where execution fills and costs are applied (remember the no-look-ahead convention: `pnl_t = pos.shift(1) * ret_t`, i.e. `position_{t-1} * return_t`).

(The TWAP/VWAP/POV/IS schedulers and the Almgren-Chriss trajectory are implemented and self-tested in `templates/execution.py`, following the closed forms in §6–§7.)