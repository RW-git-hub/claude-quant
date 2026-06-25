"""microstructure.py - Market-microstructure signals from L1 / trade data.

Tools that turn raw top-of-book (L1) quotes and trade prints into the classic
microstructure quantities: order-flow imbalance, the size-weighted microprice,
the Roll (1984) effective spread, VPIN order-flow toxicity, and Avellaneda-
Stoikov optimal market-making quotes. These are the building blocks for
execution signals, fair-value estimation and inventory-aware quoting.

This module is DISTINCT from execution.py (which schedules a parent order) and
costs.py (which models cost magnitude). Here we extract STATE/SIGNAL from the
order book and tape.

Conventions
-----------
- L1 book at time t: best bid price bp_t, bid size bq_t, best ask price ap_t,
  ask size aq_t. The mid is (bp_t + ap_t) / 2.
- Sizes are nonnegative share/contract counts; prices are positive.
- "Causal / leak-free" helpers use only information available up to and
  including the current bar. Where a function returns a per-bar series aligned
  to time t, it is stated in the docstring whether bar t's value is known at the
  CLOSE of bar t (and must therefore be lagged before use as a trade signal).

Iron-Law notes (look-ahead)
---------------------------
- order_flow_imbalance(t) uses book snapshots at t and t-1 only -> known at the
  close of bar t. To TRADE on OFI you must shift it forward one bar.
- microprice(t) uses only the current snapshot -> known at t (it is a fair-value
  estimate, not a forecast; using it as the execution reference is fine).
- roll_spread uses a trailing window of trade prices -> known at the end of the
  window; lag before using as a cost input for the NEXT trade.
- vpin is computed on COMPLETED volume buckets -> a bucket's VPIN is known only
  once the bucket fills; align to the bucket-completion time, never earlier.
- avellaneda_stoikov_quotes uses current mid, inventory and time-to-horizon ->
  all known at t.

References
----------
- Cont, Kukanov & Stoikov (2014), "The price impact of order book events",
  J. Financial Econometrics (order-flow imbalance definition).
- Gatheral & Oomen (2010) / Stoikov (2018), "The micro-price" (size-weighted
  fair value; the simple L1 form used here).
- Roll (1984), "A simple implicit measure of the effective bid-ask spread",
  J. Finance.
- Easley, Lopez de Prado & O'Hara (2012), "Flow toxicity and liquidity in a
  high-frequency world", Rev. Financial Studies (VPIN).
- Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book",
  Quantitative Finance.

Pitfalls (detect / fix)
-----------------------
- DETECT: OFI computed from book at t+1 vs t (next snapshot) -> uses the future.
  FIX: e_t depends on snapshots t-1 and t only (this module).
- DETECT: microprice used as a FORECAST and compared to a future mid without
  lagging the entry. FIX: it is a contemporaneous fair value; lag any signal
  derived from it before entering.
- DETECT: Roll spread on a positively autocorrelated (trending) series gives a
  positive autocovariance and an undefined sqrt -> the model is mis-specified.
  FIX: clamp to 0 (this module) and treat it as "Roll inapplicable".
- DETECT: VPIN bucketed by clock time instead of volume -> destroys its
  invariance to volume clustering. FIX: bucket by equal volume (this module).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Order-flow imbalance (Cont-Kukanov-Stoikov)
# ---------------------------------------------------------------------------
def order_flow_imbalance(
    bid_price: Sequence[float],
    bid_size: Sequence[float],
    ask_price: Sequence[float],
    ask_size: Sequence[float],
) -> np.ndarray:
    """Order-flow imbalance (OFI) from L1 bid/ask price and size changes.

    OFI aggregates the net change in supply/demand at the top of book between
    consecutive snapshots. Following Cont, Kukanov & Stoikov (2014), the
    contribution of one snapshot transition (t-1 -> t) is

        bid side e^b_t:
            if bp_t  > bp_{t-1}:   +bq_t              (bid stepped up: new demand)
            if bp_t == bp_{t-1}:   +(bq_t - bq_{t-1}) (size added/removed at bid)
            if bp_t  < bp_{t-1}:   -bq_{t-1}          (bid stepped down: demand pulled)

        ask side e^a_t:
            if ap_t  > ap_{t-1}:   +aq_{t-1}          (ask stepped up: supply pulled)
            if ap_t == ap_{t-1}:   +(aq_t - aq_{t-1}) (size added/removed at ask)
            if ap_t  < ap_{t-1}:   -aq_t              (ask stepped down: new supply)

        OFI_t = e^b_t - e^a_t.

    A positive OFI means net buying pressure (demand grew faster than supply),
    which empirically pushes the price up over the next interval. Note the SIGN
    convention: adding SIZE at the ask raises e^a, which LOWERS OFI (more resting
    supply is bearish); adding size at the bid raises OFI (more demand is
    bullish).

    Causal: OFI_t uses snapshots t-1 and t only, so it is known at the CLOSE of
    bar t. Lag by one bar before using it as a tradeable signal.

    Parameters
    ----------
    bid_price, bid_size, ask_price, ask_size : equal-length sequences of L1
        best-bid/ask prices and sizes, one entry per snapshot. Length N.

    Returns
    -------
    np.ndarray of length N. The first element is 0.0 (no prior snapshot to diff
    against); element t (t>=1) is OFI_t for the transition (t-1 -> t).
    """
    bp = np.asarray(bid_price, dtype=float)
    bq = np.asarray(bid_size, dtype=float)
    ap = np.asarray(ask_price, dtype=float)
    aq = np.asarray(ask_size, dtype=float)
    if not (bp.shape == bq.shape == ap.shape == aq.shape):
        raise ValueError("all four inputs must have the same shape")
    if bp.ndim != 1 or bp.size == 0:
        raise ValueError("inputs must be non-empty 1-D sequences")
    if np.any(bq < 0) or np.any(aq < 0):
        raise ValueError("sizes must be nonnegative")

    n = bp.size
    ofi = np.zeros(n, dtype=float)
    # Vectorized transition logic for t = 1..n-1.
    bp0, bp1 = bp[:-1], bp[1:]
    bq0, bq1 = bq[:-1], bq[1:]
    ap0, ap1 = ap[:-1], ap[1:]
    aq0, aq1 = aq[:-1], aq[1:]

    # Bid contribution e^b.
    e_b = np.where(
        bp1 > bp0,
        bq1,
        np.where(bp1 == bp0, bq1 - bq0, -bq0),
    )
    # Ask contribution e^a (CKS 2014, eq. for the supply side).
    e_a = np.where(
        ap1 > ap0,
        aq0,
        np.where(ap1 == ap0, aq1 - aq0, -aq1),
    )
    ofi[1:] = e_b - e_a
    return ofi


# ---------------------------------------------------------------------------
# Microprice (size-weighted mid)
# ---------------------------------------------------------------------------
def microprice(
    bid_price: float | Sequence[float],
    bid_size: float | Sequence[float],
    ask_price: float | Sequence[float],
    ask_size: float | Sequence[float],
) -> float | np.ndarray:
    """Microprice: size-weighted mid (the simple L1 fair-value estimate).

    The microprice tilts the mid toward the side with LESS size, since the
    thinner side is where the price is more likely to move next:

        I = bid_size / (bid_size + ask_size)        # imbalance in [0, 1]
        microprice = I * ask_price + (1 - I) * bid_price

    Equivalently, weighting each price by the OPPOSITE side's size. When the book
    is symmetric (bid_size == ask_size) the microprice collapses to the plain mid
    (bid_price + ask_price) / 2. When bid_size >> ask_size (buyers stacked), I->1
    and the microprice -> ask_price (price likely to tick up).

    Causal: uses only the current snapshot. It is a contemporaneous fair value,
    not a forecast; lag any signal you derive from comparing it to the mid.

    Scalar inputs return a float; array inputs return a per-snapshot np.ndarray.

    Parameters
    ----------
    bid_price, bid_size, ask_price, ask_size : scalars or equal-length
        sequences. Sizes must be nonnegative and not both zero at a snapshot.

    Returns
    -------
    float (scalar inputs) or np.ndarray (sequence inputs).
    """
    bp = np.asarray(bid_price, dtype=float)
    bq = np.asarray(bid_size, dtype=float)
    ap = np.asarray(ask_price, dtype=float)
    aq = np.asarray(ask_size, dtype=float)
    scalar = bp.ndim == 0
    bp, bq, ap, aq = np.atleast_1d(bp, bq, ap, aq)
    if not (bp.shape == bq.shape == ap.shape == aq.shape):
        raise ValueError("all four inputs must have the same shape")
    if np.any(bq < 0) or np.any(aq < 0):
        raise ValueError("sizes must be nonnegative")
    total = bq + aq
    if np.any(total <= 0):
        raise ValueError("bid_size + ask_size must be positive at every snapshot")

    imbalance = bq / total
    mp = imbalance * ap + (1.0 - imbalance) * bp
    return float(mp[0]) if scalar else mp


# ---------------------------------------------------------------------------
# Roll (1984) effective spread
# ---------------------------------------------------------------------------
def roll_spread(prices: Sequence[float]) -> float:
    """Roll (1984) implied effective spread from trade-price autocovariance.

    Roll's model: the efficient price follows a random walk, but observed
    transaction prices bounce between bid and ask by half the spread s due to the
    bid-ask "bounce". The trade-price CHANGES then have a negative first-order
    autocovariance:

        cov(dp_t, dp_{t-1}) = -s^2 / 4   =>   s = 2 * sqrt(-cov)

    where dp_t = p_t - p_{t-1}. s is the implied effective (round-trip) spread in
    PRICE units. The autocovariance is negative when the bid-ask bounce dominates;
    if it is positive (a trending / autocorrelated tape) the model is
    inapplicable and we return 0.0.

    Causal: uses a trailing window of trade prices, so the estimate is known at
    the end of the window. Lag before using as a cost input for the next trade.

    Parameters
    ----------
    prices : sequence of consecutive trade prices (length >= 3).

    Returns
    -------
    Implied effective spread s in price units (float, >= 0). Returns 0.0 when the
    first-order autocovariance of price changes is nonnegative (Roll model not
    applicable).
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 3:
        raise ValueError("prices must be a 1-D sequence of length >= 3")
    dp = np.diff(p)
    if dp.size < 2:
        return 0.0
    # First-order autocovariance of price changes (population form, mean-removed).
    d = dp - dp.mean()
    cov = float(np.sum(d[:-1] * d[1:]) / dp.size)
    if cov >= 0:
        return 0.0
    return 2.0 * float(np.sqrt(-cov))


# ---------------------------------------------------------------------------
# VPIN (volume-bucketed order-flow toxicity)
# ---------------------------------------------------------------------------
def vpin(
    buy_volume: Sequence[float],
    sell_volume: Sequence[float],
    bucket_size: float,
    n_buckets: int,
) -> np.ndarray:
    """VPIN: Volume-synchronized Probability of INformed trading.

    Easley, Lopez de Prado & O'Hara (2012). Trades are classified into buy/sell
    volume (e.g. via the tick rule or bulk-volume classification) and accumulated
    into equal-VOLUME buckets of size `bucket_size`. For each completed bucket i,
    the order-flow imbalance is |V^buy_i - V^sell_i|. VPIN at bucket i is the
    rolling average of that imbalance, normalized by bucket volume, over the last
    `n_buckets` buckets:

        VPIN_i = (1 / (n_buckets * bucket_size))
                 * sum_{j=i-n+1..i} |V^buy_j - V^sell_j|

    Since each bucket has total volume == bucket_size, this equals the mean of
    the per-bucket fractional imbalance over the window, lying in [0, 1]. High
    VPIN => one-sided (toxic) flow.

    Causal: a bucket's value is known only once the bucket FILLS to bucket_size,
    and VPIN_i needs n_buckets completed buckets. Align each VPIN value to its
    bucket-completion time; never use it before the bucket closes.

    Inputs are an incremental tape of per-event buy and sell volume; this routine
    re-buckets them into equal-volume buckets internally (volume clock).

    Parameters
    ----------
    buy_volume, sell_volume : equal-length sequences of per-event buy and sell
        volume (nonnegative). These are the *classified* volumes, not raw trades.
    bucket_size : total volume per bucket V (> 0). Each bucket holds exactly V.
    n_buckets   : window length (number of buckets in the moving average, >= 1).

    Returns
    -------
    np.ndarray of VPIN values, one per COMPLETED bucket that has a full window
    behind it. Length = max(0, n_completed_buckets - n_buckets + 1). Values lie
    in [0, 1]. Returns an empty array if fewer than n_buckets buckets complete.
    """
    bv = np.asarray(buy_volume, dtype=float)
    sv = np.asarray(sell_volume, dtype=float)
    if bv.shape != sv.shape or bv.ndim != 1:
        raise ValueError("buy_volume and sell_volume must be equal-length 1-D")
    if np.any(bv < 0) or np.any(sv < 0):
        raise ValueError("volumes must be nonnegative")
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")

    # Walk the tape, splitting each event across bucket boundaries so every
    # completed bucket holds exactly bucket_size of total (buy+sell) volume.
    # A bucket is emitted ONLY when it reaches bucket_size; a partially filled
    # trailing bucket is discarded (not yet a complete observation).
    bucket_buy: list[float] = []
    bucket_sell: list[float] = []
    cur_buy = 0.0
    cur_sell = 0.0
    cur_total = 0.0
    for b, s in zip(bv, sv):
        rem_buy = float(b)
        rem_sell = float(s)
        rem_total = rem_buy + rem_sell
        while rem_total > 1e-12:
            room = bucket_size - cur_total
            if rem_total <= room + 1e-12:
                # The remaining event fits entirely in the current bucket.
                cur_buy += rem_buy
                cur_sell += rem_sell
                cur_total += rem_total
                rem_buy = rem_sell = rem_total = 0.0
            else:
                # Pour `room` worth into the current bucket, split in the same
                # buy/sell proportion as the remaining event.
                frac = room / rem_total
                take_buy = rem_buy * frac
                take_sell = rem_sell * frac
                cur_buy += take_buy
                cur_sell += take_sell
                rem_buy -= take_buy
                rem_sell -= take_sell
                rem_total = rem_buy + rem_sell
                # Bucket is now exactly full: emit and reset.
                bucket_buy.append(cur_buy)
                bucket_sell.append(cur_sell)
                cur_buy = cur_sell = cur_total = 0.0
            # Emit if a fit-entirely event landed exactly on the boundary.
            if cur_total >= bucket_size - 1e-9 and rem_total <= 1e-12:
                bucket_buy.append(cur_buy)
                bucket_sell.append(cur_sell)
                cur_buy = cur_sell = cur_total = 0.0

    imbalances = np.abs(np.asarray(bucket_buy) - np.asarray(bucket_sell))
    n_complete = imbalances.size
    if n_complete < n_buckets:
        return np.array([], dtype=float)

    # Rolling mean of per-bucket imbalance, normalized by bucket_size.
    out = np.empty(n_complete - n_buckets + 1, dtype=float)
    for i in range(out.size):
        window = imbalances[i : i + n_buckets]
        out[i] = window.sum() / (n_buckets * bucket_size)
    return out


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov optimal market-making quotes
# ---------------------------------------------------------------------------
def avellaneda_stoikov_quotes(
    mid: float,
    inventory: float,
    gamma: float,
    sigma: float,
    kappa: float,
    time_to_horizon: float,
) -> Tuple[float, float, float, float]:
    """Avellaneda-Stoikov (2008) reservation price and optimal bid/ask quotes.

    A market maker with inventory q, risk aversion gamma, facing a mid that
    diffuses with volatility sigma, sets quotes around a RESERVATION PRICE that is
    shifted away from the mid in proportion to inventory (to mean-revert the
    book toward flat), then posts a half-spread that balances fill probability
    (governed by order-arrival decay kappa) against inventory risk.

        Reservation price:
            r = mid - q * gamma * sigma^2 * (T - t)

        Optimal total spread:
            delta_total = gamma * sigma^2 * (T - t)
                          + (2 / gamma) * ln(1 + gamma / kappa)

        Half-spread:
            delta = delta_total / 2

        Quotes:
            bid = r - delta
            ask = r + delta

    Here (T - t) = time_to_horizon is the remaining time. With long inventory
    (q > 0) the reservation price drops below the mid, skewing quotes DOWN to
    encourage selling; short inventory skews them up.

    Inventory-risk monotonicity: the half-spread's first term grows with gamma,
    sigma^2 and time_to_horizon, so a more risk-averse maker (larger gamma) quotes
    a WIDER half-spread for q != 0 / nonzero remaining time. (The second term
    decreases in gamma but the first dominates over the relevant range; the
    self-test checks the total-spread monotonicity at fixed inventory exposure.)

    Causal: uses current mid, current inventory and remaining time only - all
    known at decision time t.

    Parameters
    ----------
    mid             : current mid price (> 0 typical, not enforced).
    inventory       : current signed inventory q (long > 0, short < 0).
    gamma           : inventory risk aversion (> 0).
    sigma           : volatility of the mid (per unit time consistent with T-t).
    kappa           : order-book liquidity / arrival-rate decay (> 0). Larger
                      kappa => fills are easier => tighter quotes.
    time_to_horizon : remaining time T - t (>= 0). At 0 the inventory term
                      vanishes (terminal: no future risk).

    Returns
    -------
    (reservation_price, bid, ask, half_spread).
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if time_to_horizon < 0:
        raise ValueError("time_to_horizon must be nonnegative")

    inv_term = inventory * gamma * sigma**2 * time_to_horizon
    reservation = mid - inv_term

    spread_total = gamma * sigma**2 * time_to_horizon + (2.0 / gamma) * np.log(
        1.0 + gamma / kappa
    )
    half = spread_total / 2.0
    bid = reservation - half
    ask = reservation + half
    return float(reservation), float(bid), float(ask), float(half)


# ---------------------------------------------------------------------------
# Self-tests (analytic anchors)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- microprice: symmetric book -> plain mid ---------------------------
    mp = microprice(99.0, 500.0, 101.0, 500.0)
    assert abs(mp - 100.0) < 1e-12, mp  # symmetric => mid
    # Plain mid regardless of size when sizes are equal.
    assert abs(microprice(50.0, 10.0, 52.0, 10.0) - 51.0) < 1e-12
    # Bid stacked (more bid size) -> microprice tilts toward the ASK.
    mp_up = microprice(99.0, 900.0, 101.0, 100.0)
    assert mp_up > 100.0, mp_up
    # I = 900/1000 = 0.9 -> 0.9*101 + 0.1*99 = 100.8
    assert abs(mp_up - 100.8) < 1e-12, mp_up
    # Ask stacked -> tilts toward the BID (below mid).
    mp_dn = microprice(99.0, 100.0, 101.0, 900.0)
    assert mp_dn < 100.0 and abs(mp_dn - 99.2) < 1e-12, mp_dn
    # Microprice is bounded by [bid, ask].
    assert 99.0 <= mp_up <= 101.0 and 99.0 <= mp_dn <= 101.0
    # Vectorized form matches scalar form elementwise.
    mp_vec = microprice([99.0, 99.0], [900.0, 100.0], [101.0, 101.0], [100.0, 900.0])
    assert np.allclose(mp_vec, [100.8, 99.2]), mp_vec

    # --- order_flow_imbalance: balanced flow ~ 0 ---------------------------
    # A perfectly static book -> no events -> OFI all zero.
    bp = [100.0, 100.0, 100.0]
    bq = [500.0, 500.0, 500.0]
    ap = [101.0, 101.0, 101.0]
    aq = [500.0, 500.0, 500.0]
    ofi = order_flow_imbalance(bp, bq, ap, aq)
    assert ofi.shape == (3,)
    assert ofi[0] == 0.0  # no prior snapshot
    assert np.allclose(ofi, 0.0), ofi

    # Balanced add: +100 to bid size AND +100 to ask size cancels in OFI,
    # because e^b = +100 (demand) and e^a = +100 (supply) net out.
    ofi_sym = order_flow_imbalance(
        [100.0, 100.0], [500.0, 600.0], [101.0, 101.0], [500.0, 600.0]
    )
    # e^b = bq1-bq0 = +100, e^a = aq1-aq0 = +100, OFI = e^b - e^a = 0.
    assert abs(ofi_sym[1]) < 1e-12, ofi_sym

    # Clean balanced anchor: bid up-tick paired with ask up-tick of equal size.
    # bid steps up (+bq1 demand) and ask steps up (+aq0 supply pulled).
    ofi_bal = order_flow_imbalance(
        [100.0, 100.5], [400.0, 400.0], [101.0, 101.5], [400.0, 400.0]
    )
    # e^b = bq1 = 400 (bid stepped up). e^a = aq0 = 400 (ask stepped up).
    # OFI = 400 - 400 = 0  => balanced upward shift of the whole book.
    assert abs(ofi_bal[1]) < 1e-12, ofi_bal

    # Pure buying pressure: only bid size grows -> positive OFI.
    ofi_buy = order_flow_imbalance(
        [100.0, 100.0], [400.0, 600.0], [101.0, 101.0], [400.0, 400.0]
    )
    # e^b = 600-400 = +200, e^a = 400-400 = 0 -> OFI = +200.
    assert abs(ofi_buy[1] - 200.0) < 1e-12, ofi_buy
    # Pure selling pressure: only ask size grows -> negative OFI (more resting
    # supply is bearish under the CKS sign convention).
    ofi_sell = order_flow_imbalance(
        [100.0, 100.0], [400.0, 400.0], [101.0, 101.0], [400.0, 600.0]
    )
    # ask same price, size +200: e^a = aq1-aq0 = +200; e^b = 0; OFI = -200.
    assert abs(ofi_sell[1] - (-200.0)) < 1e-12, ofi_sell
    # Symmetry: a +200 bid add and a +200 ask add are exact mirror images.
    assert abs(ofi_sell[1] + ofi_buy[1]) < 1e-12, (ofi_sell[1], ofi_buy[1])
    # Bid pull (demand removed) -> negative OFI (bearish).
    ofi_pull = order_flow_imbalance(
        [100.0, 99.5], [400.0, 400.0], [101.0, 101.0], [400.0, 400.0]
    )
    # bid stepped DOWN: e^b = -bq0 = -400; ask unchanged: e^a = 0. OFI = -400.
    assert abs(ofi_pull[1] - (-400.0)) < 1e-12, ofi_pull

    # --- roll_spread -------------------------------------------------------
    # Roll's model: observed price = efficient_price + (s/2)*q_t, q_t = +/-1
    # i.i.d. trade-direction indicators. With a CONSTANT efficient price this
    # gives cov(dp_t, dp_{t-1}) -> -s^2/4 in expectation, recovering s.
    s_true = 0.10
    half = s_true / 2.0
    mid_px = 100.0
    rng = np.random.default_rng(7)
    q = rng.choice([-1.0, 1.0], size=20000)  # i.i.d. trade directions
    bounce = mid_px + half * q  # pure bounce, no efficient-price drift
    s_est = roll_spread(bounce)
    # Large sample -> recovers s_true within sampling error.
    assert abs(s_est - s_true) < 5e-3, (s_est, s_true)
    # Adding a small random-walk efficient price keeps the estimate near s_true
    # (the random walk is uncorrelated with the bounce and adds no lag-1 cov).
    eff = np.cumsum(rng.standard_normal(20000) * 0.001)
    s_est2 = roll_spread(mid_px + eff + half * q)
    assert abs(s_est2 - s_true) < 1.5e-2, (s_est2, s_true)
    # Trending series (positive autocovariance) -> Roll inapplicable -> 0.
    trend = np.cumsum(np.full(50, 0.5)) + 100.0
    assert roll_spread(trend) == 0.0
    # A flat series -> zero spread.
    assert roll_spread([100.0, 100.0, 100.0, 100.0]) == 0.0
    # Spread is nonnegative on noisy data.
    assert roll_spread(bounce) >= 0.0

    # --- vpin --------------------------------------------------------------
    # Perfectly balanced flow: every event is half buy half sell -> imbalance 0.
    # 20 events x 100 total = 2000 / 100 = 20 full buckets; window 2 -> 19 vals.
    bvol = [50.0] * 20
    svol = [50.0] * 20
    v_bal = vpin(bvol, svol, bucket_size=100.0, n_buckets=2)
    assert v_bal.size == 20 - 2 + 1, v_bal.size
    assert np.allclose(v_bal, 0.0), v_bal
    # Perfectly toxic flow: all buys -> imbalance == bucket_size -> VPIN == 1.
    # 10 events x 100 = 10 buckets; window 2 -> 9 values.
    v_tox = vpin([100.0] * 10, [0.0] * 10, bucket_size=100.0, n_buckets=2)
    assert v_tox.size == 10 - 2 + 1, v_tox.size
    assert np.allclose(v_tox, 1.0), v_tox
    # Half-toxic: 75 buy / 25 sell per 100-volume event -> imbalance 50/100=0.5.
    v_mid = vpin([75.0] * 10, [25.0] * 10, bucket_size=100.0, n_buckets=2)
    assert np.allclose(v_mid, 0.5), v_mid
    # VPIN always in [0, 1].
    assert np.all((v_mid >= 0.0) & (v_mid <= 1.0))
    # Bucket SPLITTING: one big all-buy event spanning two buckets -> both
    # buckets are fully toxic (VPIN == 1) and total buy volume is conserved.
    v_split = vpin([200.0], [0.0], bucket_size=100.0, n_buckets=1)
    assert v_split.size == 2, v_split.size  # 200 total -> exactly 2 buckets
    assert np.allclose(v_split, 1.0), v_split
    # Proportional split: a 200-volume event of 60% buy / 40% sell splits into
    # two buckets each preserving the 60/40 ratio -> imbalance 20/100 = 0.2.
    v_prop = vpin([120.0], [80.0], bucket_size=100.0, n_buckets=1)
    assert v_prop.size == 2 and np.allclose(v_prop, 0.2), v_prop
    # Too few buckets for the window -> empty.
    assert vpin([100.0], [0.0], bucket_size=100.0, n_buckets=5).size == 0
    # Output stays in [0, 1] on irregular event sizes.
    big_b = [37.0, 88.0, 12.0, 63.0, 100.0]
    big_s = [13.0, 12.0, 88.0, 37.0, 100.0]
    v_big = vpin(big_b, big_s, bucket_size=100.0, n_buckets=1)
    assert np.all((v_big >= 0.0) & (v_big <= 1.0)), v_big

    # --- avellaneda_stoikov_quotes -----------------------------------------
    mid0 = 100.0
    sig = 2.0
    kap = 1.5
    T = 1.0

    # Flat inventory -> reservation price == mid; quotes symmetric around mid.
    r0, b0, a0, h0 = avellaneda_stoikov_quotes(mid0, 0.0, gamma=0.1, sigma=sig,
                                               kappa=kap, time_to_horizon=T)
    assert abs(r0 - mid0) < 1e-12, r0
    assert abs((a0 - mid0) - (mid0 - b0)) < 1e-12  # symmetric
    assert h0 > 0.0

    # Long inventory -> reservation price BELOW mid (skew to sell).
    r_long, b_l, a_l, h_l = avellaneda_stoikov_quotes(
        mid0, 5.0, gamma=0.1, sigma=sig, kappa=kap, time_to_horizon=T
    )
    assert r_long < mid0, r_long
    # Short inventory -> reservation ABOVE mid.
    r_short, _, _, _ = avellaneda_stoikov_quotes(
        mid0, -5.0, gamma=0.1, sigma=sig, kappa=kap, time_to_horizon=T
    )
    assert r_short > mid0, r_short
    # Symmetric inventory skew.
    assert abs((mid0 - r_long) - (r_short - mid0)) < 1e-12

    # KEY ANCHOR: half-spread increases with inventory-risk gamma.
    # The inventory term gamma*sigma^2*(T-t) dominates the total spread's
    # gamma-dependence; compare total spread at a high vs low gamma.
    _, _, _, h_lo = avellaneda_stoikov_quotes(mid0, 0.0, gamma=0.05, sigma=sig,
                                              kappa=kap, time_to_horizon=T)
    _, _, _, h_hi = avellaneda_stoikov_quotes(mid0, 0.0, gamma=0.50, sigma=sig,
                                              kappa=kap, time_to_horizon=T)
    assert h_hi > h_lo, (h_hi, h_lo)

    # Half-spread also grows with volatility and with time-to-horizon.
    _, _, _, h_lowvol = avellaneda_stoikov_quotes(mid0, 0.0, gamma=0.1, sigma=1.0,
                                                  kappa=kap, time_to_horizon=T)
    _, _, _, h_hivol = avellaneda_stoikov_quotes(mid0, 0.0, gamma=0.1, sigma=4.0,
                                                 kappa=kap, time_to_horizon=T)
    assert h_hivol > h_lowvol
    _, _, _, h_t0 = avellaneda_stoikov_quotes(mid0, 0.0, gamma=0.1, sigma=sig,
                                              kappa=kap, time_to_horizon=0.0)
    _, _, _, h_t1 = avellaneda_stoikov_quotes(mid0, 0.0, gamma=0.1, sigma=sig,
                                              kappa=kap, time_to_horizon=2.0)
    assert h_t1 > h_t0

    # At horizon (T-t = 0) the inventory term vanishes -> reservation == mid
    # even with nonzero inventory.
    r_term, _, _, _ = avellaneda_stoikov_quotes(
        mid0, 10.0, gamma=0.1, sigma=sig, kappa=kap, time_to_horizon=0.0
    )
    assert abs(r_term - mid0) < 1e-12, r_term

    print("microstructure.py: all self-tests passed.")
