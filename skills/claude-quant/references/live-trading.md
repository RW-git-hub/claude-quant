# Live Trading / Production Operations

Taking a backtested strategy to live capital. This is about the **plumbing**, not new alpha. The dominant failure mode at go-live is not "the signal was wrong" — it is operational: a duplicate order on reconnect, a timezone bug that places trades into a closed market, a reconciliation break that hides a phantom position, or a missing kill switch when the strategy runs away. The strategy can be fine and the plumbing still kills you.

The pre-trade risk gate and reconciliation logic described here (Sections 6, 8) are operational scaffolding you build per venue; `references/transaction-costs.md` and `templates/costs.py` cover the cost modeling that live execution is constantly validating against.

---

## 1. The research-to-production gap: why live underperforms backtest

Live almost always underperforms the backtest. The gap is *expected* — the question is whether you can attribute and bound it before risking capital. Decompose it.

| Source | Detect | Fix |
|---|---|---|
| **Unmodeled costs/slippage** | Live realized cost per share/contract systematically exceeds the backtest's modeled cost; the gap grows with order size. | Re-fit the cost model on live fills (impact + spread + fees + borrow). Build orders against realistic, size-aware costs, not a flat bps. See `templates/costs.py` and `references/transaction-costs.md`. |
| **Latency** | Time between signal timestamp and order ack is non-trivial vs. the bar you traded on; price has moved by ack time. | Measure decision→ack→fill latency. Model "arrival price" decay. Don't assume you trade at the close that produced the signal. |
| **Fill assumptions** | Backtest assumed fills at mid/close/VWAP that you don't actually get; limit orders that "filled" in backtest don't fill live. | Replace fill model with a conservative one (cross the spread for marketables; model non-fill probability for passive). Reconcile modeled vs. actual fills (Section 5). |
| **Subtle look-ahead that did not exist live** | Live signal differs from what the backtest computed for the same timestamp. The backtest used data not available at decision time (revised fundamentals, restated index membership, a feature using the bar's own close to decide trades *in* that bar). | Run the live signal engine over historical point-in-time data and diff against backtest signals bar-by-bar. Any non-zero diff is a leak. See `references/pitfalls.md`. |
| **Regime change** | Live Sharpe drops outside the backtest's confidence band; correlation to a known factor spikes. | Pre-commit a drawdown/decay threshold for halting (Section 7). Don't rationalize. Track rolling live-vs-expected. |
| **Capacity / market impact** | Performance degrades as AUM grows; your own orders move the price; participation rate climbs. | Estimate capacity *before* scaling. Cap participation (Section 6). Ramp AUM gradually and watch impact per unit traded. |

**Rule:** before go-live, write down the *expected* slippage and cost in bps, and the *expected* live Sharpe haircut. Then monitor against those numbers. A gap you predicted is operations; a gap you didn't is a bug. (Note: a single live Sharpe estimate over a short window has a huge standard error — judge against a *band*, not a point, and don't over-react to a few weeks of data. See `references/stats-risk.md`.)

---

## 2. System architecture

The canonical pipeline, left to right, each stage with a clean interface:

```
Market-data feed ─► Signal/Alpha ─► Portfolio/Target ─► OMS/EMS ─► Broker/Exchange API
   (ticks/bars,      (factor/      (target weights/    (order      (FIX / REST /
    book, ref data)   model out)    positions)          mgmt,       WebSocket)
                                                        risk gate)
```

- **Market-data feed:** normalize vendor formats to one internal schema; timestamp on receipt *and* carry the exchange timestamp. Detect staleness (Section 9).
- **Signal/Alpha:** consumes point-in-time data only. Emits a signal with a timestamp and the data version it was computed from.
- **Portfolio/Target:** converts signals to **target positions/weights** (not orders). Applies vol targeting, position limits, netting. Output is desired end-state, which makes the system idempotent and recoverable.
- **OMS/EMS:** diffs target vs. current position → child orders; runs pre-trade risk (Section 6); manages order lifecycle (Section 3); schedules execution (TWAP/VWAP/POV). The **OMS owns truth about orders**; the **EMS owns truth about how they're worked**.
- **Broker/Exchange API:** the only component that talks to the venue. Wrap it so the rest of the system is venue-agnostic.

**Separation of research vs execution code.** This is non-negotiable.

- Research code optimizes for iteration speed and expressiveness; execution code optimizes for correctness, determinism, and fail-safety.
- **Share the signal/feature library** so the live signal is *literally the same code path* as the backtest signal — this is your best defense against look-ahead drift (Section 1). Do **not** share backtest fill/cost simulators into production.
- A backtest is allowed to crash and be re-run. Live code must never crash silently mid-session, must persist state, and must fail to a safe (flat / halted) state.

**Config and secrets.**

- Config is versioned, environment-scoped (`paper` / `staging` / `prod`), and reviewed. The same binary reads different config per environment — never branch on `if prod:` in code.
- Secrets (API keys, FIX credentials) live in a secrets manager or env injected at runtime, **never in the repo, logs, or error messages**. Separate keys per environment with least privilege (e.g., a paper key that *cannot* place real orders).
- Pin the prod config to a specific strategy parameter set with a hash; log the hash at startup so you know exactly what was running.

---

## 3. Order lifecycle

Every order moves through a state machine. You must handle every transition, including the unhappy ones.

```
            ┌─► ack ─┬─► partial-fill ─► fill
new ───────►│        │
            │        ├─► cancel
            └─► reject└─► expire
```

| State | Meaning | What you must do |
|---|---|---|
| **new** | sent to venue, not yet acknowledged | start an ack timeout; treat as *in-flight* (may or may not exist at venue) |
| **ack** | venue accepted the order | record venue order ID ↔ your client order ID |
| **partial-fill** | some quantity filled | update position by the *filled delta only*; leaves remain working |
| **fill** | fully filled | finalize; reconcile quantity and price |
| **cancel** | you cancelled, venue confirmed | release reserved exposure; verify *cancelled qty + filled qty = ordered qty* |
| **reject** | venue refused | log reason code; do **not** retry blindly (Section below) |
| **expire** | time-in-force reached its limit (DAY at session close; IOC/FOK cancel the unfilled remainder *immediately*) | treat unfilled remainder as not done; re-evaluate target |

Note on TIF: DAY orders expire at the session close; IOC and FOK are not "time elapsed" — IOC fills what it can *instantly* and cancels the rest, FOK requires the full quantity instantly or cancels everything. In all cases the unfilled remainder is gone and must be re-evaluated next cycle, but the mechanism and timing differ — don't model IOC/FOK as if they sit working until a timer expires.

**Client order IDs (clOrdID).** Generate a **unique, deterministic** client order ID for every order *before* sending. This is the backbone of idempotency and reconciliation. Make it traceable: `{strategy}-{date}-{seq}` or a UUID persisted before send. Never reuse an ID within the venue's dedup window. (In FIX, a *replace/cancel* assigns a **new** clOrdID that references the prior one via `OrigClOrdID` — so "never reuse" applies to genuinely new orders, not to the chained IDs of an amend.)

**Idempotency — never double-send on reconnect.** This is the single most dangerous bug in live trading.

- Persist the order intent (with its clOrdID) to durable storage **before** sending to the venue. Mark it `pending`, then `sent`, then `acked`.
- On reconnect or restart, **query open orders and recent fills from the venue** and match by clOrdID before sending anything new. If the venue already has your clOrdID, do not resend.
- Treat send as *at-least-once* and rely on the venue's clOrdID dedup + your own bookkeeping to make the *effect* exactly-once. (Caveat: not every venue/REST API dedups on a client ID — confirm yours does. If it doesn't, your persisted-state check is the *only* thing preventing a double-send, so it must be durable and queried before every send.)

```python
# Idempotent send (pseudocode)
order = build_order(target_diff)                 # has a deterministic clOrdID
store.persist(order, status="pending")           # durable BEFORE network I/O
if venue.has_open_or_recent(order.clord_id):     # survived a prior send?
    reconcile(order, venue.lookup(order.clord_id))
else:
    ack = venue.new_order(order)                 # may time out; that's fine
    store.update(order.clord_id, status="sent")
```

Subtlety: a send can *time out* without you knowing whether the venue received it. Do **not** treat a timeout as a reject — mark the order `unknown` and resolve it by querying the venue (by clOrdID) before sending anything for that intent. Resending on timeout without a clOrdID dedup guarantee is exactly how you get a doubled position.

**Handling rejects.** Categorize the reject reason:
- *Risk/limit reject* (too big, restricted symbol): do not retry — the order should never have been sent; this is a pre-trade-check failure. Alert.
- *Transient* (rate limit, throttle): back off and retry with the **same** clOrdID (so a late ack of the first attempt doesn't duplicate). Only do this if the venue dedups on clOrdID; otherwise a "rejected" first attempt that actually landed will double. Cap retries.
- *Market-state* (market closed, halted): stop trying; respect the calendar (Section 10).

**Handling partials.** Update position by filled deltas as they arrive — never recompute position from "assumed full fill." At end of TIF, the working remainder may be unfilled; recompute the target diff next cycle rather than chasing the same order forever.

---

## 4. State management & recovery

The live system *will* crash, lose its connection, or be restarted mid-session. Design for it.

- **Persistent positions and orders.** Every position, working order, and fill is written to durable storage (DB / append-only log) synchronously enough that a crash loses nothing material. Memory is a cache, not the source of truth.
- **Crash recovery / replay.** On startup, rebuild state by: (a) loading last persisted snapshot, then (b) **querying the venue** for current positions, open orders, and fills since the snapshot, then (c) reconciling the two. The venue is authoritative for what actually happened; your store is authoritative for *intent*.
- **Effective exactly-once.** True end-to-end exactly-once delivery is not achievable over an unreliable network; you approximate it with *at-least-once delivery* (idempotent sends, Section 3) + *deduplication* (clOrdID + persisted state). Don't chase true exactly-once at the network layer; make the *effect* idempotent.
- **Reconnection.** On a dropped market-data or order feed: flag data as stale immediately (do not trade on stale data), attempt reconnect with backoff, and on reconnect **resync** open orders/positions before resuming. For sequenced feeds (e.g., FIX), use the protocol's resend / gap-fill mechanism rather than assuming you missed nothing.
- **Idempotent target model.** Because Portfolio/Target emits desired end-state (Section 2), recovery is "compute diff between actual current position and target," which is naturally self-correcting after any gap.

**Test recovery explicitly:** kill the process mid-session in paper, restart, and assert that recovered state == venue state and no duplicate orders were sent. Include the nastier case: kill the process *during* a send (after persist, before ack) and assert it resolves to a single order.

---

## 5. Live-vs-backtest reconciliation

You cannot trust a backtest you have not validated against live behavior.

- **Shadow / paper trading.** Run the *production* code against live market data, sending orders to a paper/simulated venue (or logging "would-have-sent"). This validates the plumbing and the signal path without capital risk. Run it for a meaningful window (Section 11). Caveat: a paper/sim venue gives you *optimistic* fills (no real queue position, no adverse selection, no impact), so paper trading validates plumbing and signal correctness but does **not** validate realized execution cost — only live capital does.
- **Compare live fills to modeled fills.** For each fill, record: decision price (arrival), modeled fill price, actual fill price, size, venue, timestamp. Compute realized slippage relative to arrival (signed so positive = worse, per the convention below) and compare to the model's prediction.
- **Track realized vs. expected slippage** over a rolling window, per symbol and per size bucket. Persistent one-sided slippage means your cost model is wrong (re-fit) or your execution is being adversely selected (change tactic).
- **Signal reconciliation.** Re-run the live signal over point-in-time history and **diff against the backtest signal** at each timestamp. Any systematic difference is look-ahead, a data-versioning bug, or a code divergence (Section 1).
- **Drift alarms.** Alert when rolling live Sharpe / hit-rate / slippage drifts outside a pre-set band vs. the backtest expectation. Pre-commit the bands so you don't move the goalposts when live disappoints. Size the band for sampling error (a short live window has wide Sharpe error bars — see `references/stats-risk.md`).

```python
# Per-fill slippage vs arrival, in bps (positive = worse than arrival).
# arrival_px is the decision/arrival reference price; both prices must be
# the same price type (e.g., both trade prices) for the comparison to be valid.
def realized_slippage_bps(side, arrival_px, fill_px):
    sign = 1.0 if side == "BUY" else -1.0
    return sign * (fill_px - arrival_px) / arrival_px * 1e4

# Aggregate per symbol / size bucket and compare to the model's prediction;
# alert on persistent (one-sided) bias, not on single-fill noise.
```

This matches the implementation-shortfall sign convention in `references/transaction-costs.md` (`D=+1` buy / `−1` sell, benchmarked to the decision price). Note this measures slippage *per fill against arrival* — to compare to a model calibrated on implementation shortfall, aggregate over the parent order, not per child fill.

---

## 6. Pre-trade risk checks

Every order passes through a hard gate *before* leaving the OMS. These checks are cheap and catch catastrophic, irreversible mistakes.

- **Fat-finger / price collars.** Reject orders priced too far from a reference (last/mid/prev close), e.g. > X% away. Catches a misplaced decimal or a stale-price-driven order. (Beware: a legitimate overnight gap or a corporate action — Section 10 — can trip this; widen or refresh the reference around known events rather than blocking real trades.)
- **Max order size.** Reject child orders above an absolute per-order quantity/notional cap.
- **Max position per name.** Reject if the order would push the position past a per-symbol limit (long or short).
- **Gross / net exposure limits.** Reject if the order breaches portfolio gross or net notional / leverage limits.
- **Max participation.** Cap the order (or its schedule) to ≤ X% of expected/realized volume (ADV or interval volume) to bound impact and avoid being the market.
- **Restricted lists.** Block symbols on a do-not-trade list (compliance, hard-to-borrow, halted, single-name risk).
- **Self-trade prevention.** Don't cross your own resting orders. Use the venue's STP flags where available and check your own working orders before sending the opposite side.

**Design rules:** checks are **default-deny** (fail closed — if a limit or reference price is unavailable or stale, reject, don't pass), **deterministic**, **logged with the reason on every rejection**, and applied to *both* automated and any manual/override orders. A check you can silently bypass is not a check. Build this gate per venue; it is the single highest-leverage piece of operational code you will write.

---

## 7. Kill switches & circuit breakers

You need a way to stop, instantly, at multiple granularities — and automatic triggers so a human doesn't have to be watching.

- **Global flatten.** One command/button that cancels all working orders and (optionally) flattens all positions to cash. This is the "pull the plug" control. Test it in paper; it must work when everything else is on fire. (Caveat: a forced flatten *into a stressed market* can itself realize large impact — for an illiquid book, "cancel all working orders and stop" may be safer than "market-sell everything now." Decide per strategy in advance.)
- **Per-strategy disable.** Halt a single strategy (stop new orders, optionally cancel its working orders) without touching others. Granular blast-radius control.
- **Auto-halt triggers** (circuit breakers), each with a pre-committed threshold:
  - **Drawdown** beyond intraday or rolling limit.
  - **Loss limit** (daily max loss in $ / bps).
  - **Error rate** (rejects, exceptions, reconnects per minute over a threshold).
  - **Order rate / runaway** (more orders or notional per interval than the strategy should ever produce — the classic runaway-loop guard).
  - **Stale data** (no fresh ticks for N seconds → halt, don't trade blind).
  - **Reconciliation break** (Section 8) → halt until resolved.

```python
def check_circuit_breakers(state, limits):
    # daily_pnl is negative when losing; max_daily_loss is a positive magnitude.
    if state.daily_pnl <= -limits.max_daily_loss:      return halt("daily loss limit")
    if state.drawdown >= limits.max_drawdown:          return halt("drawdown limit")
    if state.errors_per_min > limits.max_error_rate:   return halt("error rate")
    if state.orders_per_min > limits.max_order_rate:   return halt("runaway order rate")
    if state.data_age_s > limits.max_data_age:         return halt("stale market data")
```

PnL- and drawdown-based breakers depend on a *correct, reconciled* mark — a stale or wrong price can make PnL look fine while you bleed (or trip a false halt). Gate the PnL breakers on data freshness, and prefer the order-rate / stale-data breakers as your first line of defense against a runaway loop, since they don't depend on valuation.

**A halt should default to safe:** stop sending new orders, cancel discretionary working orders, and require explicit human re-arm to resume. Halting must never *itself* require the very component that's failing (e.g., the kill switch must not depend on the order gateway that just died).

---

## 8. Position & PnL reconciliation with the broker

Your internal book and the broker's book **will** diverge. Find the breaks before they find you.

- **Start-of-day (SOD) reconciliation.** Before trading, pull positions, cash, and (for derivatives) margin from the broker and assert they match your persisted EOD state, adjusted for **corporate actions** and overnight settlement (Section 10). Do not start trading on an unreconciled book.
- **Intraday reconciliation.** Periodically (and after any disconnect) re-pull positions/orders and compare to internal state. Reconcile fills: every venue fill should map to one of your orders by clOrdID/venue ID. (Be aware broker position snapshots can lag your latest fills by seconds — allow for in-flight fills before declaring a break, or you'll generate false alarms.)
- **PnL reconciliation.** Compare your computed PnL to broker statements (realized + unrealized, including fees, financing, dividends). Differences usually trace to fees, FX, corporate actions, or marking convention (e.g., last vs. mid vs. settlement price).
- **Breaks investigation.** A *break* is any mismatch in quantity, price, or cash. Triage:
  1. **Quantity break** → missed/duplicated fill, or a fill you applied as full when it was partial. Most dangerous: a phantom position. **Halt** if material.
  2. **Price/PnL break** → fees, financing, corporate action, FX conversion, or marking convention not applied.
  3. **Order break** → an order acked at the venue that your system thinks failed (or vice versa) — directly tied to idempotency/recovery (Sections 3–4).

Log every break with enough context to replay it. Treat an unexplained *quantity* break as a trading-stop condition (a price/PnL break is usually a marking or fee discrepancy, not a position error — investigate but don't necessarily halt).

---

## 9. Monitoring & alerting

If it isn't monitored, it's already broken and you don't know.

- **Heartbeats.** Every component emits a heartbeat; absence of a heartbeat is an alert. The monitoring system must itself be monitored (dead-man's switch).
- **Staleness detection.** Age of last tick per feed, age of last fill, age of last reconciliation. Stale market data → halt (Section 7), don't trade on it. (Calibrate "stale" per market — a thin name or an overnight session can legitimately go quiet for minutes; a liquid future going silent for seconds is alarming.)
- **Fill quality / slippage-vs-model.** Live dashboards of realized vs. expected slippage and cost (Section 5); alert on persistent bias.
- **Latency.** Decision→ack→fill latencies, p50/p99; alert on degradation (often the first sign of a venue or network problem).
- **Error rates.** Rejects, exceptions, reconnects, retries per interval.
- **Exposure & PnL.** Live gross/net, per-name, drawdown, daily PnL against limits.
- **Dashboards.** One screen that answers "is the system healthy and within limits right now?" — green/red, not a wall of numbers.
- **Pager policy.** Tiered: *page now* (kill-switch-worthy: runaway orders, reconciliation break, feed down during market hours), *alert* (degraded but contained: elevated slippage, single reject), *log only* (informational). Every page must be actionable and map to a runbook. Alert fatigue is an operational risk — tune thresholds so a page means something.

---

## 10. Time/clock sync, timezones, exchange calendars, corporate actions

Time bugs are silent and brutal. They place orders into closed markets, mislabel bars, and corrupt reconciliation.

- **Clock sync (NTP/PTP).** Sync host clocks (NTP at minimum; PTP for latency-sensitive). Monitor clock drift and alert. A skewed clock corrupts every timestamp, latency measurement, and TIF.
- **Timezones.** Store and compute in **UTC** internally; convert to exchange-local only at the boundary (for calendars, session times, display). Be explicit about DST: exchange open/close in local time stays fixed, but its *UTC offset* shifts when that region's DST changes — and different regions change DST on different dates (and the Southern Hemisphere is offset by six months), so a single global "spring forward" assumption is wrong. Never do naive datetime math across a DST boundary; use a real tz database (e.g., IANA `zoneinfo`), not fixed offsets. Equities, futures, crypto (24/7), and FX (roughly 24/5 with regional sessions) all differ — encode each market's sessions, not a global assumption.
- **Exchange calendars live.** Use a maintained, point-in-time calendar of trading days, holidays, half-days, and session times. Before trading, assert the market is open *now* for *this* product. Half-days and ad-hoc holidays cause early closes — respect them or your end-of-day orders land after the close.
- **Corporate actions in production.** Splits, dividends, symbol changes, mergers, spin-offs change position quantity, cost basis, and reference prices **overnight**. Apply them to your book and your reference data *before* SOD reconciliation, or you'll see a (false) quantity/PnL break and a (real) collared/fat-finger reject on the next morning's order (e.g., a 2:1 split doubles your share count and halves the price, so a collar referencing yesterday's unadjusted price rejects everything). For futures: handle expiry and roll explicitly, and make sure the contract you trade is the active/front one, not an expired symbol. See `references/data.md` and `references/derivatives.md`.

---

## 11. Deployment

Promote through environments; never push research-grade code straight to prod capital.

1. **Staging.** Production code + config, against live (or replayed) market data, sending to a simulated venue. Validates the full pipeline and recovery (Section 4) with zero capital risk.
2. **Paper-trading period.** Run the production system in paper for a pre-committed window long enough to observe real market conditions, signal reconciliation (Section 5), and at least one of every lifecycle event (partials, rejects, cancels, reconnects). Skipping paper trading is a top cause of go-live disasters. Remember paper fills are optimistic (Section 5) — paper validates correctness, not realized cost.
3. **Canary / ramp.** Go live with **minimal capital** (a canary allocation). Confirm fills, reconciliation, and slippage match expectations. Then **ramp AUM gradually**, watching impact/capacity (Section 1) at each step. Do not jump to full size.
4. **Rollback.** Every deploy is reversible: pin the previous known-good binary + config hash, and have a tested procedure to revert *and* reconcile state. A deploy that changes order behavior should go out when you can watch it, not minutes before the close.

Deploys touching live trading: change-controlled, reviewed, logged (who/what/when/config hash), and ideally not during volatile sessions.

---

## 12. Operational risk: the strategy is fine, the plumbing kills you

Most live blowups are operational, not alpha. Budget engineering accordingly.

- **Logging / audit trail.** Append-only, timestamped (UTC), immutable log of every signal, target, order, ack, fill, reject, cancel, risk-check decision, halt, and config change. You must be able to **fully reconstruct** any trading day. This is both a debugging tool and, in many jurisdictions, a regulatory requirement. Never log secrets.
- **Failover / HA.** No single point of failure for the components that can lose money or visibility. Redundant feeds; a standby that can take over with the persisted/venue-reconciled state; the kill switch reachable even if the primary is down. Test failover — an untested failover is a liability, not a safeguard. (Guard against split-brain: two instances both believing they're primary will double every order. Use a single source of leadership, and make the gateway reject orders from a non-leader.)
- **Runbooks.** Every pageable alert maps to a written, tested response: how to halt, how to flatten, how to reconcile a break, how to fail over, who to call.
- **Least privilege & blast radius.** Prod keys do only what they must; paper keys can't trade real; one strategy's failure can't take down others.

---

## Go-live checklist

**Code & environment**
- [ ] Production and research code separated; live signal uses the *same* feature/signal library as the backtest
- [ ] Config is environment-scoped, versioned, and reviewed; strategy param set pinned to a hash logged at startup
- [ ] Secrets in a manager / env injection; none in repo, logs, or errors; per-environment least-privilege keys
- [ ] Paper key provably cannot place real orders

**Signal & cost validation**
- [ ] Live signal diffed bar-by-bar against backtest over point-in-time data; zero systematic difference (no look-ahead)
- [ ] Cost/slippage model re-validated; expected live Sharpe haircut and slippage (bps) written down — judged against a band, not a point
- [ ] Capacity estimated; max participation rate set

**Order handling**
- [ ] Deterministic client order IDs persisted *before* send
- [ ] Idempotent send: reconnect/restart queries venue and matches clOrdID before sending — no double-send; venue clOrdID dedup behavior confirmed
- [ ] Send timeouts resolved by querying the venue, not treated as rejects
- [ ] Every lifecycle state handled: ack, partial, fill, cancel, reject (categorized), expire (DAY vs IOC/FOK semantics)
- [ ] Position updated by filled deltas only

**State & recovery**
- [ ] Positions/orders/fills persisted durably; venue authoritative for fills, store authoritative for intent
- [ ] Crash-kill-restart test passes (including kill during a send): recovered state == venue state, no duplicate orders
- [ ] Reconnect resync of open orders/positions before resuming; stale data flagged and never traded on

**Risk gate**
- [ ] Price collar / fat-finger, max order size, max position per name, gross/net exposure, max participation, restricted list, self-trade prevention — all active, default-deny, logged on reject, applied to manual orders too

**Kill switches**
- [ ] Global flatten tested in paper (with a decided flatten-vs-cancel policy for illiquid books)
- [ ] Per-strategy disable works
- [ ] Auto-halts armed: drawdown, daily loss, error rate, runaway order rate, stale data, reconciliation break — thresholds pre-committed; PnL breakers gated on data freshness; resume requires human re-arm

**Reconciliation**
- [ ] SOD reconciliation (positions/cash/margin) vs. broker, corporate-action adjusted
- [ ] Intraday reconciliation scheduled and after every disconnect; tolerant of in-flight fills
- [ ] PnL reconciled to broker (fees, financing, dividends, FX, marking convention)
- [ ] Break triage runbook exists; material quantity break triggers halt

**Monitoring & alerting**
- [ ] Heartbeats + dead-man's switch; staleness detection on feeds/fills/reconciliation (thresholds calibrated per market)
- [ ] Live slippage-vs-model, latency (p50/p99), error rates, exposure, PnL vs. limits
- [ ] Tiered pager policy; every page maps to a runbook

**Time & calendars**
- [ ] NTP/PTP synced; clock drift monitored
- [ ] UTC internal, exchange-local at boundary; DST handled via a real tz database, not fixed offsets
- [ ] Point-in-time exchange calendar; assert market open for product before trading; half-days handled
- [ ] Corporate actions (splits/dividends/symbol changes) and futures roll/expiry applied before SOD recon

**Deployment & ops**
- [ ] Staging validated; paper-trading period completed (saw partials, rejects, cancels, reconnects)
- [ ] Canary live at minimal capital; ramp plan defined
- [ ] Rollback procedure tested (binary + config + state recon)
- [ ] Append-only UTC audit log reconstructs a full day; failover tested (split-brain guarded); runbooks written

---

## Pitfalls

| Pitfall | Detect | Fix |
|---|---|---|
| **No kill switch** | Strategy misbehaves and there's no way to stop it fast; "we'd have to log into the broker manually." | Build global flatten + per-strategy disable + auto-halts (Section 7) *before* go-live. Test in paper. |
| **No reconciliation** | Internal book silently diverges from broker; a phantom position accumulates unnoticed. | SOD + intraday + PnL reconciliation; halt on material quantity break (Section 8). |
| **Assuming backtest fills** | Live fills systematically worse than modeled; passive orders that "filled" in backtest don't fill live. | Conservative live fill model; reconcile modeled vs. actual fills; re-fit costs on live data (Sections 1, 5). |
| **Duplicate orders on reconnect** | Two orders for one intent after a disconnect/restart/timeout; doubled position. | Persist clOrdID before send; on reconnect query venue and match before sending; confirm and rely on venue dedup; resolve timeouts by query, not retry (Section 3). |
| **Timezone bugs** | Orders into closed markets; bars mislabeled; recon breaks around DST; end-of-day orders land after the close. | UTC internal, local at boundary via real tz db; point-in-time calendars; assert market open; handle DST and half-days (Section 10). |
| **Skipping paper trading** | Go-live is the first time prod code meets the real venue; first partial/reject/reconnect is in production. | Mandatory paper-trading period over real conditions before canary (Section 11). |
| **No slippage monitoring** | Edge quietly erodes; you find out from the monthly statement. | Per-fill realized-vs-model slippage tracking with drift alarms (Sections 5, 9). |
| **Split-brain / double primary** | Two instances both act as primary after a failover; every order doubles. | Single source of leadership; gateway rejects non-leader orders; test failover (Section 12). |