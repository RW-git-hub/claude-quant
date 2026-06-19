# Crypto / DeFi Quant

Practical quant for 24/7 crypto and DeFi: market structure, perpetual funding, basis, on-chain data, AMMs, impermanent loss, liquidations, MEV, and the risks that wreck naive backtests. Companion code: `templates/crypto_defi.py`. For execution cost modeling see `references/transaction-costs.md`. For a related "pricing without an official close / fragmented venues / settlement edge cases" treatment see `references/prediction-sports-markets.md`.

Conventions used here match the rest of the skill: simple returns compound multiplicatively; positions are lagged versus the returns they earn (`pnl_t = pos.shift(1) * ret_t`); annualized Sharpe = `mean(excess)/std(excess, ddof=1)*sqrt(ppy)`. Crypto twist: there is no weekend gap, so the periods-per-year (ppy) you choose must match your bar (see Section 1).

---

## 1. Market structure

Crypto trades continuously, on many uncoordinated venues, with no authoritative closing print. Every downstream choice (returns, vol, Sharpe annualization, "gap" handling) inherits this.

- **24/7, no official close.** Markets never halt — no overnight gap, no weekend, no settlement auction. "Daily close" is a convention you impose, not a fact the market provides.
- **Fragmented venues, no NBBO.** No consolidated tape. The "price" of BTC differs across Binance, Coinbase, Kraken, OKX, plus on-chain DEX pools. Spreads and depth differ; cross-venue basis is real and tradable but carries transfer/withdrawal latency and counterparty risk.
- **CEX vs DEX.**
  - *CEX* (centralized exchange): off-chain matching engine, custodial, deep liquidity, low explicit fees (~1–10 bps maker/taker), but custody risk (FTX) and withdrawal gates.
  - *DEX* (on-chain, e.g. Uniswap, Curve): non-custodial, transparent, but every trade costs **gas + LP fee + price impact + MEV leakage** (Section 8) and is publicly visible in the mempool before it confirms.
- **Choosing a daily cutoff.** Pick one UTC instant and apply it everywhere. **00:00 UTC** is the most common convention (CoinGecko/CoinMarketCap daily candles, most data vendors). Document it; mixing 00:00 UTC closes with 16:00 ET / 21:00 UTC (US equity) closes silently corrupts cross-asset studies. Note that some venues historically used non-UTC daily boundaries (e.g. Binance's UI defaulting to local/8:00 UTC for some products) — always verify the vendor's stated cutoff rather than assuming UTC midnight.
- **Annualization.** Match ppy to your sampling because there are no non-trading days:
  - daily bars: **ppy = 365** (crypto), not 252 — there are no weekends/holidays off. (Use 365.25 if you want leap-year precision; the difference is immaterial for Sharpe.)
  - hourly bars: ppy = 365*24 = 8760.
  - If you deliberately resample to a business-day TradFi calendar to compare with equities, then 252 is appropriate — but state it.
  - **Caveat on sub-hourly annualization.** Annualizing a Sharpe from 1-minute (or finer) bars by `sqrt(525600)` is usually meaningless: high-frequency returns have strong microstructure autocorrelation and fat tails, so the IID-scaling assumption behind `sqrt(ppy)` breaks. Report the raw per-bar statistics, or use a Newey-West / block estimator, before quoting an annualized number.
- **Wash trading in reported volume.** A large fraction of self-reported CEX volume (especially on smaller/unregulated venues) is fabricated to inflate rankings. Do **not** use raw reported volume for liquidity sizing, ADV-based cost models, or volume-weighted signals without filtering.

```python
# Detect: cutoff convention is implicit / inconsistent
# Fix: resample explicitly to a named UTC convention before computing returns
import numpy as np
import pandas as pd

def to_daily_utc(bars: pd.DataFrame, price_col="close") -> pd.Series:
    """Last print within each [00:00, next 00:00) UTC day, stamped at the day's start.

    Input index must be a DatetimeIndex. If tz-naive it is assumed to be UTC and
    localized; if tz-aware it is converted to UTC. With closed='left'/label='left'
    the bar covering a calendar day is stamped at that day's 00:00 UTC and contains
    every print from 00:00 up to (but not including) the next 00:00 — i.e. the
    standard 00:00-UTC daily candle. (Do NOT use closed='right'/label='right': that
    pulls the next day's opening print into the prior day's bar.)
    """
    idx = bars.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    px = bars[price_col].copy()
    px.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    daily = px.resample("1D", label="left", closed="left").last()
    return daily.dropna()

# Annualization helper — DO NOT hardcode 252 for native crypto bars.
# WARNING: sub-hourly keys are provided for completeness but sqrt-scaling a Sharpe
# from them is statistically unsound (see caveat above).
PPY = {"1D": 365, "1H": 365 * 24, "1min": 365 * 24 * 60}
def ann_sharpe(excess: pd.Series, bar="1D") -> float:
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return excess.mean() / sd * np.sqrt(PPY[bar])
```

---

## 2. Perpetual futures & funding

The perpetual swap ("perp") is the dominant crypto instrument. It has no expiry; instead a periodic **funding payment** between longs and shorts tethers the perp's `mark` price to the underlying `index` (spot).

**Funding mechanics.**
- Funding is exchanged directly between traders (longs <-> shorts), **not** paid to the exchange.
- Paid every funding interval — commonly **every 8h** (00:00 / 08:00 / 16:00 UTC on Binance/Bybit), but **venue-specific** (dYdX/Hyperliquid pay hourly; some venues 4h or 1h; Binance may switch a symbol to 4h dynamically when funding hits the cap). Always read the venue spec.
- The rate is built from a premium component plus a fixed interest component, then **clamped** to a cap. Caps are venue- and tier-specific (commonly ±0.5% to ±2% per 8h interval for the base tier; do not hardcode a single number):

```text
premium_t      = (mark_t - index_t) / index_t            # simplified; real venues use a premium index
funding_rate_t = clamp( premium_ewma + interest_component, -cap, +cap )
payment        = position_notional * funding_rate_t        # sign: + means longs pay shorts
```

- **Sign convention:** positive funding => perp trades *above* spot (crowded longs) => **longs pay shorts**. Negative => shorts pay longs. Funding flips sign and magnitude frequently. (Note: the interest component creates a small positive funding bias even at zero premium on most venues, so funding is mildly positive "on average.")

**Funding as carry.** A held perp position accrues funding P&L on top of price P&L. Annualizing a typical 8h rate: `funding_annual ≈ funding_8h * 3 * 365` (3 intervals/day × 365 days). A persistent +0.01% / 8h is ≈ +10.95%/yr paid *by* longs — material, and routinely ignored in naive perp backtests.

**The funding / cash-and-carry basis trade.** Build a **delta-neutral** book to harvest funding:
- Funding persistently **positive** (longs pay): **short the perp + long spot** (or long a dated future). Net price exposure ≈ 0; you collect funding from longs.
- Funding persistently **negative**: **long the perp + short spot**; you collect from shorts.
- P&L ≈ Σ funding − (spot/perp drift on imperfect hedge) − fees − borrow/financing − liquidation buffer cost. The carry is real but bounded by margin, exchange counterparty risk, and the risk that funding flips before you can unwind.

```python
def perp_pnl(pos, ret_perp, funding_rate, funding_is_due):
    """
    pos:            target position (units of notional), set at t, lagged below.
    ret_perp:       simple return of the perp mark over each bar.
    funding_rate:   per-interval rate aligned to the bar where the stamp lands;
                    sign + => longs pay shorts.
    funding_is_due: bool mask, True on bars where a funding stamp occurs.
    Funding is charged on the position HELD into the stamp (pos.shift(1)), matching
    the price-return lag convention.
    """
    held = pos.shift(1)                     # positions lagged vs returns they earn
    price_pnl   = held * ret_perp
    funding_pnl = -held * funding_rate.where(funding_is_due, 0.0)  # long pays when rate>0
    return price_pnl + funding_pnl
```

Detect a perp backtest that ignores funding: P&L for a flat-price, persistently-long position is ~0. With funding modeled, the same position bleeds (or earns) carry. If your funding column is all zeros or absent, the backtest is wrong.

---

## 3. Futures basis & term structure

Dated futures (e.g. CME BTC/ETH, exchange quarterly contracts) carry a **basis** versus spot. Annualize it to compare across maturities:

```text
basis_annual = (futures_price / spot_price - 1) * (365 / days_to_expiry)
```

This is a simple (linearized) annualization. If you want a continuously-compounded basis use `ln(futures/spot) * 365/days_to_expiry`; for the typical small near-dated basis the two agree closely.

- **Contango:** futures > spot (positive basis). Normal in bull regimes; longs implicitly pay to hold leverage. A long-spot/short-future carry trade earns the basis as it converges to 0 at expiry.
- **Backwardation:** futures < spot (negative basis). Signals spot demand / short-perp crowding / stress.
- Use **365** (calendar) days for crypto, consistent with Section 1. The basis term structure (3M, 6M, perp-implied) is itself a regime signal — steep contango often precedes leverage flushes.

---

## 4. On-chain data

DeFi exposes a parallel, fully auditable data source. Latency, reorg risk, and decoding effort differ from CEX feeds.

- **Blocks & transactions.** State advances per block (~12s on Ethereum mainnet post-Merge, sub-second on some L2s). A block can be **reorged** (replaced). Post-Merge, Ethereum mainnet provides deterministic *finality* roughly every two epochs (~12–13 min, ~64–95 slots); before finality the tip can still reorg, so treat the most recent blocks as non-final ("confirmations"). Using unconfirmed/reorg-prone tip data introduces look-ahead-like corruption.
- **Mempool.** Pending, not-yet-included transactions. Public => your DEX order is visible to searchers *before* it confirms (the root of front-running/MEV, Section 8). Caveat: a large and growing share of orderflow now goes through *private* mempools / builder relays, so the public mempool is an incomplete view.
- **Addresses / UTXOs.** Ethereum-style: account balances. Bitcoin-style: UTXO set. On-chain analytics (active addresses, exchange in/outflows, whale moves) are feature sources — but address clustering and labels are heuristic and noisy.
- **Events / logs.** Contracts emit structured `event` logs (e.g. Uniswap `Swap`, `Mint`, `Burn`; ERC-20 `Transfer`). These are the cleanest primitive for reconstructing pool state, LP positions, and volumes. Decode with the contract ABI.
- **Subgraphs (The Graph).** Indexed, queryable views over events via GraphQL — far faster than raw `eth_getLogs` scans. Beware: subgraphs can lag the chain head and may have indexing bugs; reconcile against raw logs for research-grade data.
- **Oracles (Chainlink).** On-chain price feeds many protocols rely on for liquidations/pricing. **Oracle-manipulation risk:** if a protocol uses a spot DEX pool (or a thin feed) as its oracle, an attacker flash-loans size to move the pool, triggers a mispriced liquidation/mint, and reverses — a recurring exploit class. Robust designs use time-weighted average prices (TWAPs) or multi-source medianized feeds precisely to raise the cost of this attack. When modeling a protocol, identify *which* oracle it trusts and how manipulable it is.

```python
# Treat the chain tip as non-final: require confirmations before using a block's state
def is_final(block_number, head, min_conf=12):
    return (head - block_number) >= min_conf
# Detect: backtest reads pool reserves at the tip block -> exposed to reorgs / MEV ordering
# Fix: snapshot reserves at block <= head - min_conf
```

---

## 5. AMMs (automated market makers)

DEX liquidity is priced by a curve, not an order book.

**Constant-product (Uniswap v2): `x * y = k`.**
- Reserves `x` (token0), `y` (token1); invariant `k = x*y` held constant by swaps (it actually grows slightly as fees accrue to reserves).
- **Spot price** of token0 in token1: `P = y / x`.
- **Swap output with fee** (fee `f`, e.g. 0.30% => `f = 0.003`): selling `dx` of token0 for token1,

```text
dx_eff = dx * (1 - f)
dy     = y * dx_eff / (x + dx_eff)        # amount of token1 received
```

- **Price impact / slippage** grows with trade size relative to reserves. Execution price `dy/dx` is strictly worse than spot `y/x`; the gap is your slippage and it is *deterministic from the reserves at execution* — model it exactly, do not assume mid. (It is only deterministic given the reserve state when *your* trade lands; other txs ordered ahead of you in the same block change the reserves — see MEV, Section 8.)

```python
def v2_amount_out(dx, x, y, fee=0.003):
    dx_eff = dx * (1 - fee)
    dy = y * dx_eff / (x + dx_eff)
    exec_price = dy / dx                  # token1 per token0 actually received
    spot       = y / x
    slippage   = exec_price / spot - 1    # negative: you got less than spot
    return dy, slippage
```

**Concentrated liquidity (Uniswap v3) intuition.** LPs allocate liquidity to a chosen price **range** `[p_a, p_b]` instead of `(0, ∞)`. Within the range the position behaves like a v2 pool with amplified depth (higher capital efficiency, more fees per dollar). Outside the range the position is 100% in one token and earns **no** fees. This concentrates both fee income *and* impermanent loss (Section 6): for the same price move, a tighter range produces proportionally larger IL on the deployed capital, and a position that drifts out of range stops earning while fully exposed to the move. Active range management (and its gas cost) is the real job.

---

## 6. Impermanent loss (IL)

IL is the opportunity cost of providing liquidity to an AMM versus simply **HODLing** the two tokens. It arises because the AMM mechanically sells the appreciating asset and buys the depreciating one.

**Formula.** Let `r` = price ratio = (new price / initial price) of the volatile asset (vs the other). For a constant-product 50/50 pool:

```text
IL(r) = 2 * sqrt(r) / (1 + r) - 1        # <= 0 always; 0 only at r = 1
```

The IL curve is symmetric in log-price and always non-positive:

| price move `r` | IL |
|---|---|
| 1.00 (no change) | 0.0% |
| 1.25 | -0.62% |
| 1.50 | -2.02% |
| 2.00 | -5.72% |
| 4.00 | -20.0% |
| 0.50 | -5.72% |
| 5.00 | -25.46% |

IL is symmetric in `r` vs `1/r`: a 2x and a 0.5x both cost 5.72%. It is "impermanent" only in that it reverses if price returns to entry — if you withdraw after a permanent move, the loss is realized. (Note the formula above is the pure value-divergence term; it ignores fees, which is exactly why "fees vs IL" below is the real question.)

**Fees vs IL breakeven for LPs.** LP net P&L ≈ **fees earned − IL − gas**. An LP is profitable only when accumulated fee income exceeds IL over the holding period:

```text
fee_income_period  ≈ pool_fee_rate * volume_through_position / position_value
LP_profitable  <=>  fee_income_period > |IL(r)| + gas_and_rebalance_costs
```

High-volume/low-volatility pairs (stable-stable, correlated assets) earn fees with little IL — the structurally attractive LP zone. Volatile/low-volume pairs lose. **Concentrated (v3) liquidity amplifies both terms**, raising the breakeven bar.

```python
import numpy as np
def impermanent_loss(r):
    return 2 * np.sqrt(r) / (1 + r) - 1     # r = price_new / price_init

def lp_net_return(r, fee_rate, volume_to_liquidity_ratio, gas_frac=0.0):
    """Crude LP P&L vs HODL, as a fraction of position value.
    fee_rate * volume_to_liquidity_ratio approximates fees earned per unit of
    position value over the period; il is already negative; gas_frac is positive.
    This is a v2 / full-range approximation only.
    """
    il   = impermanent_loss(r)
    fees = fee_rate * volume_to_liquidity_ratio
    return fees + il - gas_frac              # il is already negative
# Detect: LP "yield" quoted as APR with zero IL term -> overstated
# Fix: subtract IL over the realized price path before claiming the position was profitable
```

---

## 7. Liquidations

Leverage in crypto is forcibly unwound when collateral can no longer cover a position. Liquidations are both a risk to your own book and a tradable/dangerous market dynamic.

- **Perp liquidation.** Each position has a **maintenance margin** (MM) requirement. When equity / notional falls below MM, the engine force-closes the position (often at a worse-than-mark price, plus a liquidation fee). For an isolated long the liquidation price ≈ `entry * (1 - 1/leverage + mm_rate)` (and for an isolated short ≈ `entry * (1 + 1/leverage - mm_rate)`). This is an approximation that ignores fees, accrued funding, and the entry fee already debited from margin — real engines fold those in, so true liquidation triggers slightly earlier. Higher leverage => liquidation price sits closer to entry.
- **Lending liquidation (Aave/Compound/Maker).** Borrow against collateral; a **health factor** = `collateral_value * liquidation_threshold / debt_value`. When it drops below 1, liquidators repay part of the debt and seize collateral at a **liquidation bonus** (e.g. 5–10%) — the incentive that gets liquidators to act.
- **Cascade / contagion risk.** Liquidations sell collateral into the market, pushing price further against remaining leveraged positions, triggering *more* liquidations — a reflexive cascade (the classic crypto "long squeeze" / "liquidation wick"). Funding, basis, and on-chain health-factor distributions are leading indicators.
- **Liquidation bots.** Permissionless on DeFi: bots monitor health factors and race (via MEV/priority gas, Section 8) to capture the liquidation bonus. On CEX, the exchange's liquidation engine (and its insurance fund) plays this role.

```python
# Detect: leveraged backtest never checks margin -> infinite survival, fantasy returns
# Fix: enforce a maintenance-margin stop each bar
def liquidated(equity, notional, mm_rate):
    return equity <= mm_rate * abs(notional)   # force-close + fee when True
```

---

## 8. MEV (maximal extractable value)

MEV is value extracted by **reordering, inserting, or censoring** transactions within a block. On a public mempool, your on-chain trade is an open target. MEV is a **real, recurring execution cost** for DEX strategies, not a tail event.

- **Arbitrage (benign-ish).** Searchers equalize prices across pools/venues. Tightens prices but competes for the same cross-pool edges you might target.
- **Front-running.** A searcher sees your pending profitable tx and submits the same action first with higher priority.
- **Sandwich attack (the big LP/taker tax).** Around your DEX swap, a searcher places a buy **before** (pushing price up) and a sell **after** (capturing the inflated price), pocketing the difference — you get filled at the worst price inside your slippage tolerance. Loose slippage limits = larger sandwich.
- **Priority-gas auctions (PGA) / priority fees.** Searchers bid gas/priority to win ordering. Post-Flashbots and post-PBS (proposer-builder separation), much of this moved to private orderflow / builder auctions, but the cost is still borne by users via worse fills or paid as priority tips to builders/proposers.

**Quant implication.** For any on-chain (DEX) execution, total cost ≈ `gas + LP_fee + price_impact + MEV_leakage`. Mitigations to model: tight slippage tolerance, private RPC / orderflow (e.g. Flashbots Protect), batch auctions (e.g. CoW Swap), splitting size, and avoiding obviously profitable public txs.

```python
# Detect: DEX backtest fills at mid/spot with only the LP fee
# Fix: charge gas + deterministic curve impact (Section 5) + an MEV/sandwich haircut
def dex_fill_cost(dx, x, y, fee=0.003, gas_usd=0.0, mev_bps=5.0, notional_usd=None):
    """Returns total adverse cost as a positive fraction of notional.
    slip from v2_amount_out is negative (you got less than spot), so -slip is the
    positive curve-impact cost; fee and the MEV haircut are added on top.
    """
    _, slip = v2_amount_out(dx, x, y, fee)
    cost = -slip + fee + mev_bps / 1e4         # fractional, all adverse
    if notional_usd:
        cost += gas_usd / notional_usd          # gas amortized over trade size
    return cost
```

---

## 9. Stablecoins, staking/yield, bridges

- **Stablecoins & depeg risk.** A "$1" peg is a claim, not a guarantee. Fiat-backed (USDC, USDT) depend on reserve quality and redemption (USDC briefly traded near ~$0.88 during the March 2023 SVB scare before recovering). Crypto-collateralized (DAI) depend on collateral and oracles. Algorithmic (UST) can **reflexively collapse to ~0** (Terra/LUNA, May 2022). Never treat a stablecoin as a literal risk-free unit in P&L; model peg deviation and redemption gates as a fat-tailed risk factor.
- **Staking / yield.** Native staking (ETH PoS), liquid staking tokens (e.g. stETH), and DeFi lending yields are *not* risk-free carry: they carry slashing, smart-contract, de-peg (stETH/ETH discount during stress, e.g. mid-2022), and lockup/unbonding-queue risk. Quote yields net of these, and treat "APY" with skepticism (often includes inflationary token emissions that dilute, not real cash yield).
- **Bridges.** Moving assets across chains routes through bridge contracts that have been **among the single largest sources of exploit losses** (Ronin ~$625M, Wormhole ~$320M, Nomad ~$190M). A cross-chain or cross-venue arb's carry must price bridge risk + latency. Wrapped assets (e.g. wBTC) inherit the custodian/bridge's solvency.

---

## 10. Risks & survivorship in backtests

Crypto-specific risks that must be priced or screened, not assumed away:

- **Smart-contract / audit risk.** Code is the counterparty. Audits reduce but don't eliminate exploit risk; an audited protocol can still be drained (logic bugs, economic exploits).
- **Bridge risk** (Section 9) — outsized historical losses.
- **Oracle risk** (Section 4) — manipulable feeds drive bad liquidations/mints.
- **Custody risk.** CEX insolvency/withdrawal freeze (Mt. Gox, FTX, Celsius). "Not your keys, not your coins."
- **Rug-pull / exit scam.** Team mints/dumps tokens or pulls liquidity. Endemic in long-tail tokens.
- **Regulatory risk.** Token can be delisted, geofenced, or reclassified; stablecoin issuers face redemption/regulatory shocks.
- **Survivorship bias (dead tokens).** This is the dominant backtest killer in crypto. Thousands of tokens have gone to ~0 or been delisted. A universe built from *currently listed* tokens silently excludes every failure — inflating returns of any "buy small-cap alts" strategy enormously. You **must** include delisted/dead tokens in the historical universe at their last traded (often ~0) price. (Even "carry to ~0" is optimistic: a dead token is typically *illiquid* well before it prints 0, so model an exit-cost / un-sellable haircut, not a clean mark.)

```python
# Detect: universe == today's listed tokens, queried historically
# Fix: build a point-in-time universe that includes tokens later delisted/dead,
#      carrying them to a realistic terminal value (often ~0) rather than dropping them.
def point_in_time_universe(listings_history, as_of):
    """listings_history needs listed_at, delisted_at (NaT if still live), symbol.
    Returns symbols live as of `as_of`, excluding future-listed tokens and including
    tokens that were live then but have since delisted/died (no survivorship leak).
    """
    live = listings_history[(listings_history.listed_at <= as_of) &
                            (listings_history.delisted_at.isna() |
                             (listings_history.delisted_at > as_of))]
    return set(live.symbol)   # excludes future knowledge, includes soon-to-die tokens
```

---

## Pitfalls (detect / fix)

| Pitfall | Detect | Fix |
|---|---|---|
| Ignoring funding in perp backtests | Flat-price held perp shows ~0 P&L; funding column absent/all-zero | Model funding stamps every interval; `funding_pnl = -pos.shift(1)*rate` on due bars (Sec 2) |
| Treating 24/7 as if it has a daily close/gap | Uses 252 ppy; "overnight gap" logic; weekend handling | ppy=365 daily (8760 hourly); single named UTC cutoff (00:00 UTC); no gap modeling (Sec 1) |
| sqrt-scaling a Sharpe from sub-hourly bars | Annualized Sharpe quoted from 1-min returns via sqrt(525600) | Report per-bar stats; use Newey-West/block estimator before annualizing (Sec 1) |
| Ignoring gas + MEV + slippage on DEX trades | DEX fills at mid/spot, only LP fee charged | Charge `gas + curve impact + MEV/sandwich haircut`; tight slippage; private orderflow (Sec 5, 8) |
| Underestimating impermanent loss | LP "yield" quoted as fee APR with no IL term | Subtract `IL(r)` over realized price path; LP profit only if fees > \|IL\| + gas (Sec 6) |
| Survivorship (delisted / dead tokens) | Universe = currently-listed tokens, applied historically | Point-in-time universe including dead tokens at terminal (~0) price + illiquidity haircut (Sec 10) |
| Trusting wash-traded volume | Liquidity/ADV/VWAP sizing off raw reported CEX volume | Filter venues; use trade-print/on-chain or regulated-venue volume; haircut suspicious feeds (Sec 1) |
| Oracle / tip-read assumptions | Backtest prices DeFi actions at manipulable spot oracle; reads chain tip | Identify the protocol's actual oracle + manipulability; require block confirmations (Sec 4) |
| Leveraged book never liquidated | Positions survive arbitrary drawdowns; no margin check | Enforce maintenance-margin / health-factor stop each bar (Sec 7) |
| Stablecoin treated as exact $1 | P&L uses peg = 1.0 with no deviation/redemption risk | Model peg deviation + redemption gates as fat-tailed factor (Sec 9) |

---

## See also
- `templates/crypto_defi.py` — runnable implementations: funding P&L, v2 swap/slippage, IL, basis, liquidation checks, point-in-time universe.
- `references/transaction-costs.md` — general cost modeling (extend with gas + MEV + curve impact for DEX).
- `references/prediction-sports-markets.md` — kindred no-official-close / fragmented-venue / settlement-edge pricing problems.
- `references/pitfalls.md`, `references/data.md` — survivorship, point-in-time, and look-ahead foundations these crypto cases specialize.