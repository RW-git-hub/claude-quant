"""Deterministic pre-trade risk gate.

A pre-trade check is the last deterministic gate between a strategy's order
intent and the exchange. It is intentionally dumb, fast, and side-effect free:
given an order, the current book of positions, and a set of hard limits, it
answers a single question -- may this order be sent? -- and, if not, why not.

Design principles (these are the point of the template, not the arithmetic):

  * Deterministic. No RNG, no clock, no network, no I/O. The same inputs always
    produce the same verdict, so the gate is unit-testable and replayable. Any
    non-determinism here is a production incident waiting to happen.
  * Fail closed. Missing or malformed inputs should reject, never silently pass.
  * Notional-based. Limits are expressed in currency notional (qty * price), not
    share/contract counts, so they are comparable across symbols and price
    levels. (For futures/options you would multiply by a contract multiplier
    before calling this -- pass a notional-consistent qty/price, or extend the
    order dict with a `multiplier` field. Documented, not silently assumed.)
  * Check the RESULTING state, not just the order. Position and gross limits are
    evaluated AFTER applying the fill, because that is the state you are
    actually authorizing. A small order that flips you through a limit must be
    caught.
  * All violations, not just the first. We collect every breach so an operator
    sees the full picture in one pass rather than fixing-and-resubmitting in a
    loop.

Sign convention: positions are signed (long > 0, short < 0). A 'buy' adds +qty,
a 'sell' adds -qty. qty is always passed as a positive magnitude; side carries
the direction. (Detect: a negative qty is rejected as malformed rather than
reinterpreted, so a flipped sign can't quietly double your exposure.)

This module is standard-library only (dataclasses + typing). numpy/pandas are
not imported and not required -- a pre-trade gate sits on the hot path and
should have the smallest possible dependency surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RiskLimits:
    """Hard pre-trade limits. All notionals are in account currency.

    Attributes:
        max_order_notional:     Max |qty * price| for a single order.
        max_position_notional:  Max |resulting signed position notional| per symbol.
        max_gross_notional:     Max portfolio gross (sum of |position notional|)
                                after the fill. Each held symbol is valued at its
                                own mark (via the `marks` arg of check_order; the
                                traded symbol's leg uses the order price). A
                                missing mark fails closed (see check_order).
        price_collar_bps:       Max allowed |price/ref_price - 1| in basis points.
                                Guards against fat-finger / stale-quote prices.
        max_participation_pct:  Max qty/ADV as a percent (only checked if adv given).
        kill_switch:            If True, every order is rejected. The master
                                "stop trading now" flag flipped by an operator
                                or an automated circuit breaker.

    frozen=True makes a limit set hashable and immutable: once constructed for a
    session it cannot be mutated in place by accident. Build a new RiskLimits to
    change limits.
    """

    max_order_notional: float
    max_position_notional: float
    max_gross_notional: float
    price_collar_bps: float
    max_participation_pct: float
    kill_switch: bool = False


def _signed_delta(side: str, qty: float) -> float:
    """Signed position change from an order. buy -> +qty, sell -> -qty.

    Raises ValueError on an unknown side so a typo cannot be silently treated as
    a buy (fail closed).
    """
    s = side.lower()
    if s == "buy":
        return +qty
    if s == "sell":
        return -qty
    raise ValueError(f"unknown side {side!r} (expected 'buy' or 'sell')")


def check_order(
    order: Dict[str, object],
    positions: Dict[str, float],
    limits: RiskLimits,
    adv: Optional[Dict[str, float]] = None,
    working_orders: Optional[Dict[str, Dict[str, object]]] = None,
    marks: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Evaluate one order against hard risk limits.

    Args:
        order: dict with keys
            symbol:    str
            side:      'buy' | 'sell'
            qty:       positive magnitude of shares/contracts to trade
            price:     limit/expected fill price used for notional and collar
            ref_price: reference (e.g. last/mid/arrival) price for the collar
                       and as the fallback mark for existing positions in the
                       gross check
            clOrdID:   optional client order id. Required (must be present) only
                       when `working_orders` is supplied, so the duplicate /
                       self-cross checks have a stable identity to key on.
        positions: symbol -> current signed position (qty). Long > 0, short < 0.
        limits:    RiskLimits instance.
        adv:       optional symbol -> average daily volume (same units as qty).
                   If provided, the participation check is applied.
        working_orders: optional clOrdID -> resting order dict already live at the
                   venue. Each value is an order-shaped dict carrying at least
                   `symbol` and `side`. When supplied (opt-in), two extra gates
                   run:
                     * double-send: if this order's `clOrdID` already keys a
                       working order, reject (an idempotent client must not
                       re-send the same id; a re-send usually means a retry storm
                       or a lost ack, never a second intended order).
                     * self-cross: if a resting working order on the SAME symbol
                       sits on the OPPOSITE side, flag it -- crossing your own
                       book is wash-trade-adjacent and almost always an error.
                   Causal/leak-free: `working_orders` is the set of orders live
                   *as of* this decision; it contains no future state. This makes
                   the gate safe to replay in a backtest with a point-in-time
                   working-order book.
        marks:     optional symbol -> mark price for the gross-notional check. Each
                   HELD symbol is valued at ITS OWN mark, not the traded symbol's
                   price. When `marks` is supplied, a held symbol with NO mark
                   fails closed (we will not silently mismark a position to zero or
                   to an unrelated symbol's price). When `marks` is None the legacy
                   behaviour is kept: held legs are marked at the order's
                   ref_price. The traded symbol's resulting leg always uses the
                   order price, consistent with the per-symbol position check.
                   Leak-free: marks are point-in-time observable inputs.

    Returns:
        dict(ok: bool, violations: list[str]). ok == (len(violations) == 0).

    The function never raises for a *limit breach* -- a breach is data, returned
    in `violations`. It only raises for *malformed input* (bad side, etc.),
    which is a programming error upstream and must not be swallowed.
    """
    violations: List[str] = []

    symbol = str(order["symbol"])
    side = str(order["side"])
    qty = float(order["qty"])
    price = float(order["price"])
    ref_price = float(order["ref_price"])

    # --- Malformed-input guards: fail closed -------------------------------
    if qty < 0:
        violations.append(f"qty {qty} is negative (use side to express direction)")
    if price <= 0:
        violations.append(f"price {price} is not positive")
    if ref_price <= 0:
        violations.append(f"ref_price {ref_price} is not positive")

    # If inputs are unusable, stop here -- downstream arithmetic is meaningless.
    if violations:
        return {"ok": False, "violations": violations}

    # --- Kill switch: reject everything ------------------------------------
    if limits.kill_switch:
        violations.append("kill_switch engaged: all orders rejected")
        # Short-circuit: when trading is halted no other check is informative.
        return {"ok": False, "violations": violations}

    # --- Single-order notional ---------------------------------------------
    order_notional = qty * price
    if order_notional > limits.max_order_notional:
        violations.append(
            f"order notional {order_notional:,.2f} exceeds "
            f"max_order_notional {limits.max_order_notional:,.2f}"
        )

    # --- Resulting per-symbol position notional ----------------------------
    # Evaluate the state AFTER the fill, marked at the order price.
    delta = _signed_delta(side, qty)
    current_qty = float(positions.get(symbol, 0.0))
    resulting_qty = current_qty + delta
    resulting_pos_notional = abs(resulting_qty) * price
    if resulting_pos_notional > limits.max_position_notional:
        violations.append(
            f"resulting position notional {resulting_pos_notional:,.2f} for "
            f"{symbol} exceeds max_position_notional "
            f"{limits.max_position_notional:,.2f}"
        )

    # --- Resulting portfolio gross notional --------------------------------
    # Gross = sum of |position notional| across all symbols after the fill.
    # Each HELD symbol is valued at ITS OWN mark (marks[sym]) -- marking the whole
    # book at the traded symbol's price systematically mis-states gross whenever
    # symbols trade at different price levels. If `marks` is supplied and a held
    # symbol has no mark, we FAIL CLOSED rather than substitute an unrelated price
    # (a missing mark is a data gap, not a free pass). If `marks` is None we keep
    # the legacy behaviour: held legs marked at the order's ref_price. The traded
    # symbol's resulting leg always uses the order price, consistent with the
    # per-symbol position check above.
    gross = 0.0
    gross_data_ok = True
    for sym, q in positions.items():
        if sym == symbol:
            continue
        if marks is not None:
            if sym not in marks:
                violations.append(
                    f"no mark for held symbol {sym} in gross check "
                    f"(failing closed)"
                )
                gross_data_ok = False
                continue
            sym_mark = float(marks[sym])
            if sym_mark <= 0:
                violations.append(
                    f"mark for held symbol {sym} is {sym_mark} (not positive)"
                )
                gross_data_ok = False
                continue
        else:
            sym_mark = ref_price
        gross += abs(float(q)) * sym_mark
    gross += resulting_pos_notional
    # Only assert against the limit when every held leg was marked. With a missing
    # or bad mark the gross is incomplete; we have already failed closed above and
    # an incomplete sum could otherwise look spuriously compliant.
    if gross_data_ok and gross > limits.max_gross_notional:
        violations.append(
            f"resulting gross notional {gross:,.2f} exceeds "
            f"max_gross_notional {limits.max_gross_notional:,.2f}"
        )

    # --- Price collar (fat-finger / stale-quote guard) ---------------------
    collar = abs(price / ref_price - 1.0)
    collar_limit = limits.price_collar_bps / 1e4
    if collar > collar_limit:
        violations.append(
            f"price {price} is {collar * 1e4:,.1f} bps from ref_price "
            f"{ref_price}, exceeds collar {limits.price_collar_bps:,.1f} bps"
        )

    # --- Participation vs ADV ----------------------------------------------
    # Only enforced when ADV is supplied for the symbol. A missing ADV is NOT
    # treated as zero (which would falsely pass): it simply skips the check, and
    # the absence should be alerted on upstream if ADV coverage is expected.
    if adv is not None and symbol in adv:
        symbol_adv = float(adv[symbol])
        if symbol_adv > 0:
            participation = qty / symbol_adv
            part_limit = limits.max_participation_pct / 100.0
            if participation > part_limit:
                violations.append(
                    f"participation {participation * 100:,.2f}% of ADV exceeds "
                    f"max_participation_pct {limits.max_participation_pct:,.2f}%"
                )
        else:
            # ADV present but non-positive is malformed data -> fail closed.
            violations.append(f"adv for {symbol} is {symbol_adv} (not positive)")

    # --- Working-order gates (opt-in: only when working_orders supplied) ----
    # These guard against operationally-broken sends rather than risk-limit
    # breaches. Both read only the point-in-time working-order book, so they are
    # causal/leak-free and safe to replay.
    if working_orders is not None:
        cl_ord_id = order.get("clOrdID")
        if cl_ord_id is None:
            # Identity is required to reason about duplicates: fail closed.
            violations.append(
                "clOrdID is required when working_orders is supplied "
                "(cannot check double-send without an id)"
            )
        elif cl_ord_id in working_orders:
            # Double-send: the same client id is already live at the venue.
            violations.append(
                f"clOrdID {cl_ord_id!r} is already working (double-send rejected)"
            )

        # Self-cross: any resting order on the same symbol, opposite side.
        this_side = side.lower()
        for w_id, w_order in working_orders.items():
            if w_id == cl_ord_id:
                continue  # the order itself, handled by the double-send check
            if str(w_order.get("symbol")) != symbol:
                continue
            w_side = str(w_order.get("side")).lower()
            if w_side and w_side != this_side:
                violations.append(
                    f"self-cross: working order {w_id!r} is {w_side} {symbol} "
                    f"against this {this_side} (crossing your own book)"
                )

    return {"ok": len(violations) == 0, "violations": violations}


# ---------------------------------------------------------------------------
# Self-tests. Run `python pretrade_checks.py`; it verifies itself or aborts.
# All cases are analytic/synthetic and deterministic (no RNG, no I/O).
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # A roomy baseline so that, unless a specific limit is targeted, nothing
    # else trips. Each test then tightens exactly one dimension.
    limits = RiskLimits(
        max_order_notional=1_000_000.0,
        max_position_notional=2_000_000.0,
        max_gross_notional=5_000_000.0,
        price_collar_bps=50.0,            # 0.50%
        max_participation_pct=10.0,       # 10% of ADV
        kill_switch=False,
    )

    # Existing book used by several tests.
    positions = {
        "AAPL": 1_000.0,    # long
        "MSFT": -500.0,     # short
    }
    adv = {"AAPL": 5_000_000.0, "MSFT": 3_000_000.0, "TSLA": 1_000_000.0}

    # --- 1. Compliant order -> ok True, no violations ----------------------
    ok_order = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": 100.0,
        "price": 200.0,        # order notional 20,000
        "ref_price": 200.0,    # collar = 0
    }
    res = check_order(ok_order, positions, limits, adv)
    assert res["ok"] is True, res
    assert res["violations"] == [], res

    # --- 2. Oversized order trips max_order_notional -----------------------
    # qty 10,000 * 200 = 2,000,000 > 1,000,000 order limit.
    # (Note: this also would breach the position limit; we assert the order-
    #  notional message is present rather than asserting it is the only one.)
    big_order = {
        "symbol": "TSLA",         # flat, isolates from existing book where possible
        "side": "buy",
        "qty": 10_000.0,
        "price": 200.0,
        "ref_price": 200.0,
    }
    res = check_order(big_order, positions, limits, adv)
    assert res["ok"] is False, res
    assert any("max_order_notional" in v for v in res["violations"]), res

    # --- 3. Position-breaching order trips max_position_notional ------------
    # AAPL already long 1,000. Buy 9,500 more @ 200 -> 10,500 shares ->
    # 2,100,000 notional > 2,000,000 position limit. Keep order notional under
    # 1,000,000 by pricing so this isolates the position check:
    #   qty 4,900 @ 200 = 980,000 order notional (under 1,000,000 ok);
    #   resulting 1,000 + 4,900 = 5,900 -> *200 = 1,180,000 ... not enough.
    # Use a higher price to push position notional over without breaching the
    # order limit: qty 800 @ 999 = 799,200 order notional (ok); resulting
    # 1,800 shares * 999 = 1,798,200 ... still under. Instead allow the order
    # limit to be the binding-but-separate one and assert the position message
    # specifically appears.
    pos_breach = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": 9_500.0,
        "price": 200.0,        # order notional 1,900,000 (also > order limit)
        "ref_price": 200.0,
    }
    res = check_order(pos_breach, positions, limits, adv)
    assert res["ok"] is False, res
    assert any("max_position_notional" in v for v in res["violations"]), res

    # Cleaner isolation of the position check: widen the order limit just for
    # this sub-case so ONLY the position limit can trip.
    wide_order_limit = RiskLimits(
        max_order_notional=10_000_000.0,
        max_position_notional=2_000_000.0,
        max_gross_notional=50_000_000.0,
        price_collar_bps=50.0,
        max_participation_pct=100.0,
        kill_switch=False,
    )
    res = check_order(pos_breach, positions, wide_order_limit, adv)
    assert res["ok"] is False, res
    assert any("max_position_notional" in v for v in res["violations"]), res
    assert not any("max_order_notional" in v for v in res["violations"]), res

    # --- 4. Off-collar price trips the collar ------------------------------
    # price 207 vs ref 200 -> 3.5% = 350 bps > 50 bps collar. Keep notional /
    # position / participation comfortably inside their limits.
    off_collar = {
        "symbol": "TSLA",
        "side": "buy",
        "qty": 100.0,
        "price": 207.0,        # order notional 20,700 (ok)
        "ref_price": 200.0,
    }
    res = check_order(off_collar, positions, limits, adv)
    assert res["ok"] is False, res
    assert any("collar" in v for v in res["violations"]), res

    # Collar boundary is inclusive: exactly 50 bps must PASS.
    on_collar_edge = {
        "symbol": "TSLA",
        "side": "buy",
        "qty": 100.0,
        "price": 200.0 * (1.0 + 50.0 / 1e4),  # exactly +50 bps
        "ref_price": 200.0,
    }
    res = check_order(on_collar_edge, positions, limits, adv)
    assert res["ok"] is True, res

    # --- 5. High participation trips the participation limit ---------------
    # TSLA ADV 1,000,000; qty 200,000 -> 20% > 10% limit. Keep notional small
    # via a low price so only participation trips.
    high_part = {
        "symbol": "TSLA",
        "side": "buy",
        "qty": 200_000.0,
        "price": 4.0,          # order notional 800,000 (ok); position 800,000 (ok)
        "ref_price": 4.0,
    }
    res = check_order(high_part, positions, limits, adv)
    assert res["ok"] is False, res
    assert any("participation" in v for v in res["violations"]), res

    # No-ADV symbol skips the participation check (does not falsely pass/fail).
    no_adv = {
        "symbol": "NOADV",
        "side": "buy",
        "qty": 200_000.0,
        "price": 4.0,
        "ref_price": 4.0,
    }
    res = check_order(no_adv, positions, limits, adv)  # NOADV not in adv dict
    assert res["ok"] is True, res
    assert res["violations"] == [], res

    # --- 6. Kill switch rejects even a compliant order ---------------------
    kill_limits = RiskLimits(
        max_order_notional=1_000_000.0,
        max_position_notional=2_000_000.0,
        max_gross_notional=5_000_000.0,
        price_collar_bps=50.0,
        max_participation_pct=10.0,
        kill_switch=True,
    )
    res = check_order(ok_order, positions, kill_limits, adv)
    assert res["ok"] is False, res
    assert any("kill_switch" in v for v in res["violations"]), res

    # --- 7. Gross notional check across the book ---------------------------
    # Build a book whose existing gross is large, then a small extra order that
    # pushes the resulting gross over the limit.
    #   AAPL 1,000 @ 100 ref = 100,000
    #   MSFT  -500 @ 100 ref =  50,000  -> existing gross 150,000
    # Set gross limit to 180,000. Buy 1,000 GOOG @ 100 -> +100,000 gross.
    # AAPL leg here is the *other* symbol so it marks at ref_price=100.
    gross_limits = RiskLimits(
        max_order_notional=1_000_000.0,
        max_position_notional=1_000_000.0,
        max_gross_notional=180_000.0,
        price_collar_bps=50.0,
        max_participation_pct=100.0,
        kill_switch=False,
    )
    gross_positions = {"AAPL": 1_000.0, "MSFT": -500.0}
    gross_order = {
        "symbol": "GOOG",
        "side": "buy",
        "qty": 1_000.0,
        "price": 100.0,        # resulting GOOG notional 100,000
        "ref_price": 100.0,    # marks AAPL/MSFT legs too
    }
    # Existing (AAPL 100,000 + MSFT 50,000) + new GOOG 100,000 = 250,000 > 180,000
    res = check_order(gross_order, gross_positions, gross_limits)
    assert res["ok"] is False, res
    assert any("max_gross_notional" in v for v in res["violations"]), res

    # A smaller order stays under the gross limit and passes.
    small_gross_order = {
        "symbol": "GOOG",
        "side": "buy",
        "qty": 200.0,          # +20,000 -> resulting gross 170,000 < 180,000
        "price": 100.0,
        "ref_price": 100.0,
    }
    res = check_order(small_gross_order, gross_positions, gross_limits)
    assert res["ok"] is True, res
    assert res["violations"] == [], res

    # --- 8. Resulting-state semantics: reducing a position is allowed ------
    # Even past a notional that a fresh open could not reach, trimming exposure
    # should pass the position check because resulting |position| shrinks.
    trim_positions = {"AAPL": 12_000.0}  # 12,000 * 200 = 2,400,000 > pos limit
    trim_order = {
        "symbol": "AAPL",
        "side": "sell",
        "qty": 2_500.0,          # resulting 9,500 * 200 = 1,900,000 < 2,000,000
        "price": 200.0,
        "ref_price": 200.0,
    }
    trim_limits = RiskLimits(
        max_order_notional=1_000_000.0,
        max_position_notional=2_000_000.0,
        max_gross_notional=5_000_000.0,
        price_collar_bps=50.0,
        max_participation_pct=100.0,
        kill_switch=False,
    )
    res = check_order(trim_order, trim_positions, trim_limits, adv)
    assert res["ok"] is True, res  # order notional 500,000 ok; resulting pos ok

    # --- 9. Malformed input fails closed and short-circuits ----------------
    bad = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": -100.0,         # negative magnitude
        "price": 200.0,
        "ref_price": 200.0,
    }
    res = check_order(bad, positions, limits, adv)
    assert res["ok"] is False, res
    assert any("negative" in v for v in res["violations"]), res

    # --- 10. Multiple simultaneous breaches are all reported ---------------
    multi = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": 50_000.0,       # order notional 50,000*210 = 10,500,000 > order limit
        "price": 210.0,        # collar 5% = 500 bps > 50 bps
        "ref_price": 200.0,
    }
    res = check_order(multi, positions, limits, adv)
    assert res["ok"] is False, res
    assert any("max_order_notional" in v for v in res["violations"]), res
    assert any("collar" in v for v in res["violations"]), res
    assert any("max_position_notional" in v for v in res["violations"]), res
    # AAPL participation: 50,000 / 5,000,000 = 1% < 10% -> should NOT appear.
    assert not any("participation" in v for v in res["violations"]), res

    # --- 11. Double-send: an already-working clOrdID is rejected ------------
    # The same client order id is resting at the venue. Re-sending it (a retry
    # storm / lost-ack) must be rejected even though the order is otherwise fine.
    working = {
        "CLORD-1": {"symbol": "AAPL", "side": "buy", "qty": 100.0, "price": 200.0},
    }
    dup_order = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": 100.0,
        "price": 200.0,        # fully compliant on every risk dimension
        "ref_price": 200.0,
        "clOrdID": "CLORD-1",  # collides with a working order
    }
    res = check_order(dup_order, positions, limits, adv, working_orders=working)
    assert res["ok"] is False, res
    assert any("double-send" in v for v in res["violations"]), res

    # A fresh clOrdID against the same working book passes the double-send gate
    # (and is not a self-cross: working CLORD-1 is buy, this is buy too).
    fresh_order = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": 100.0,
        "price": 200.0,
        "ref_price": 200.0,
        "clOrdID": "CLORD-2",
    }
    res = check_order(fresh_order, positions, limits, adv, working_orders=working)
    assert res["ok"] is True, res
    assert res["violations"] == [], res

    # Self-cross: a fresh id but the SAME symbol on the OPPOSITE side is flagged.
    cross_order = {
        "symbol": "AAPL",
        "side": "sell",        # opposite of the resting buy on AAPL
        "qty": 100.0,
        "price": 200.0,
        "ref_price": 200.0,
        "clOrdID": "CLORD-3",
    }
    res = check_order(cross_order, positions, limits, adv, working_orders=working)
    assert res["ok"] is False, res
    assert any("self-cross" in v for v in res["violations"]), res

    # Backwards-compatible: with no working_orders supplied, clOrdID is ignored
    # and the legacy behaviour is unchanged.
    res = check_order(dup_order, positions, limits, adv)
    assert res["ok"] is True, res
    assert res["violations"] == [], res

    # --- 12. Multi-price gross via per-symbol marks ------------------------
    # AAPL 1,000 long, MSFT 500 short. Mark AAPL @ 300, MSFT @ 50.
    #   AAPL leg: 1,000 * 300 = 300,000
    #   MSFT leg:   500 *  50 =  25,000   -> existing gross 325,000
    # Trade GOOG: buy 100 @ 100 -> +10,000 GOOG leg. Resulting gross = 335,000.
    # If the book were (wrongly) marked all-at-100 like the old code, existing
    # gross would be (1,000 + 500) * 100 = 150,000 -> total 160,000, which would
    # PASS a 200,000 limit. With correct marks (335,000) it must FAIL.
    mark_positions = {"AAPL": 1_000.0, "MSFT": -500.0}
    mark_marks = {"AAPL": 300.0, "MSFT": 50.0}
    mark_limits = RiskLimits(
        max_order_notional=1_000_000.0,
        max_position_notional=1_000_000.0,
        max_gross_notional=200_000.0,
        price_collar_bps=50.0,
        max_participation_pct=100.0,
        kill_switch=False,
    )
    mark_order = {
        "symbol": "GOOG",
        "side": "buy",
        "qty": 100.0,
        "price": 100.0,
        "ref_price": 100.0,
    }
    res = check_order(mark_order, mark_positions, mark_limits, marks=mark_marks)
    assert res["ok"] is False, res
    assert any("max_gross_notional" in v for v in res["violations"]), res
    # The reported gross is the correctly-marked 335,000, not the all-at-100
    # 160,000 the old single-price code would have produced.
    assert any("335,000" in v for v in res["violations"]), res

    # Same book, but a generous limit -> the correctly-marked gross passes.
    mark_limits_wide = RiskLimits(
        max_order_notional=1_000_000.0,
        max_position_notional=1_000_000.0,
        max_gross_notional=400_000.0,
        price_collar_bps=50.0,
        max_participation_pct=100.0,
        kill_switch=False,
    )
    res = check_order(mark_order, mark_positions, mark_limits_wide, marks=mark_marks)
    assert res["ok"] is True, res
    assert res["violations"] == [], res

    # --- 13. Missing mark fails closed -------------------------------------
    # marks supplied but MSFT has no mark -> reject; do NOT silently mismark.
    partial_marks = {"AAPL": 300.0}  # MSFT missing
    res = check_order(mark_order, mark_positions, mark_limits_wide, marks=partial_marks)
    assert res["ok"] is False, res
    assert any("no mark for held symbol MSFT" in v for v in res["violations"]), res
    # And it must not have spuriously emitted a gross-limit pass/fail off the
    # incomplete sum.
    assert not any("max_gross_notional" in v for v in res["violations"]), res

    print("pretrade_checks.py: all self-tests passed.")
