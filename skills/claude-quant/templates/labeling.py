"""Triple-barrier labeling and sample-uniqueness weights for financial ML.

Why this module exists
----------------------
Fixed-horizon labels (sign of the return `horizon` bars ahead) ignore the path:
a position that would have been stopped out intraday gets the same label as one
that ran cleanly to expiry. The *triple-barrier method* (Lopez de Prado,
"Advances in Financial Machine Learning", ch. 3) labels by which barrier is
touched FIRST along the realized price path:

    upper (profit-take)  -> +1
    lower (stop-loss)    -> -1
    vertical (max hold)  ->  0   (time-out; neither horizontal barrier hit)

Because the holding spans of overlapping events share the same underlying
returns, the resulting samples are NOT IID. `average_uniqueness` quantifies how
much each label's outcome is "shared" with concurrent labels, yielding sample
weights for bagging / cross-validation (ch. 4). `meta_label` produces the
secondary {0,1} target of meta-labeling: take/skip a bet whose SIDE is decided
by a primary model.

Leakage conventions enforced here
---------------------------------
- A label uses ONLY information strictly AFTER the entry bar. The path scan over
  `j in (i, t1]` excludes bar `i` itself; the entry price is `close[i]`.
- `triple_barrier_labels` returns `t1`, the integer bar at which the outcome is
  realized. Anything downstream (CV purge/embargo, sample weights) must respect
  `t1`: a feature/label observed at `i` is only "known" at `t1`, not at `i`.
  See templates/validation.py for purged + combinatorial CV that consumes `t1`.
- Barrier widths are scaled by an EX-ANTE volatility estimate (`get_daily_vol`,
  an EWMA std using returns available up to and including the entry bar). Never
  scale a barrier by realized vol measured over the holding period.

Conventions
-----------
- Simple returns: r(i->j) = close[j] / close[i] - 1.
- Positional integer indices throughout (entry bars, t1) so the same code works
  for any DatetimeIndex/RangeIndex. Map back with `close.index[t1]` if needed.
- numpy / pandas / stdlib only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Volatility (for scaling barriers)
# --------------------------------------------------------------------------- #
def get_daily_vol(close: pd.Series, span: int = 20) -> pd.Series:
    """EWMA standard deviation of simple returns, for scaling barrier widths.

    Returns an ex-ante vol estimate aligned to `close.index`: the value at bar
    `i` uses returns observed up to and including bar `i`, so it is safe to use
    as the barrier scale for an event entered at `i` (no look-ahead).

    Parameters
    ----------
    close : price series.
    span  : EWMA span (center of mass span, pandas `span=` convention).

    Returns
    -------
    pd.Series of return-stdev, same index as `close`. The first bar (no prior
    return) is NaN and forward-filled is the caller's choice; here it is left
    as produced by `ewm` (NaN at position 0).
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pd.Series")
    ret = close.pct_change()
    vol = ret.ewm(span=span).std()
    vol.name = "daily_vol"
    return vol


# --------------------------------------------------------------------------- #
# Fixed-horizon labels (the naive baseline)
# --------------------------------------------------------------------------- #
def fixed_horizon_labels(
    close: pd.Series, horizon: int, threshold: float
) -> pd.Series:
    """Sign of the forward return over `horizon` bars AFTER t, vs +/-threshold.

    Label at bar t is computed from r = close[t+horizon]/close[t] - 1:
        +1 if r >  threshold
        -1 if r < -threshold
         0 otherwise (inside the band).

    The last `horizon` bars have no full forward window and are dropped (no
    look-ahead, no partial-window labels).

    Parameters
    ----------
    close     : price series.
    horizon   : positive number of bars to look forward.
    threshold : non-negative return band half-width (absolute, e.g. 0.01 = 1%).

    Returns
    -------
    pd.Series in {-1, 0, 1}, indexed like `close[:-horizon]`.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    px = close.values.astype(float)
    n = len(px)
    if n <= horizon:
        return pd.Series(dtype="int64")
    fwd = px[horizon:] / px[:-horizon] - 1.0  # r over (t, t+horizon]
    lab = np.zeros(len(fwd), dtype="int64")
    lab[fwd > threshold] = 1
    lab[fwd < -threshold] = -1
    return pd.Series(lab, index=close.index[: n - horizon], name="label")


# --------------------------------------------------------------------------- #
# Triple-barrier labels
# --------------------------------------------------------------------------- #
def triple_barrier_labels(
    close: pd.Series,
    event_idx,
    pt_mult: float,
    sl_mult: float,
    vertical_barrier: int,
    vol: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Label each event by the FIRST barrier touched along the price path.

    For event entered at positional bar `i` (entry price = close[i]):
        upper threshold = +pt_mult * vol_i     (profit-take, as a return)
        lower threshold = -sl_mult * vol_i     (stop-loss,   as a return)
    If `vol is None`, `pt_mult` / `sl_mult` are used directly as ABSOLUTE return
    thresholds (upper = +pt_mult, lower = -sl_mult).

    The path return r(i->j) = close[j]/close[i] - 1 is scanned for
        j in (i, min(i + vertical_barrier, last_bar)]   # excludes entry bar i
    and the first j where r >= upper (label +1) or r <= lower (label -1) decides
    the outcome. If neither is hit, the event times out at the vertical bar with
    label 0. `t1` is the touch bar (or the vertical bar); `ret` is r(i->t1).

    Parameters
    ----------
    close            : price series (positional bars).
    event_idx        : iterable of positional integer entry-bar indices.
    pt_mult, sl_mult : profit-take / stop-loss multipliers (>= 0). With `vol`
                       they scale per-event vol; without `vol` they are absolute
                       return thresholds.
    vertical_barrier : max holding in bars (>= 1).
    vol              : optional ex-ante vol series aligned to `close.index`,
                       used as the per-event barrier scale (see get_daily_vol).

    Returns
    -------
    pd.DataFrame indexed by the event entry bar, columns:
        t1    : int   first-touch (or vertical) bar
        ret   : float return r(i->t1)
        label : int   in {-1, 0, 1}
    """
    if vertical_barrier < 1:
        raise ValueError("vertical_barrier must be >= 1")
    if pt_mult < 0 or sl_mult < 0:
        raise ValueError("pt_mult and sl_mult must be non-negative")

    px = close.values.astype(float)
    last = len(px) - 1
    if vol is not None:
        # Align vol to close's positional order.
        vol_vals = vol.reindex(close.index).values.astype(float)

    events = list(event_idx)
    rows_t1 = np.empty(len(events), dtype="int64")
    rows_ret = np.empty(len(events), dtype="float64")
    rows_lab = np.empty(len(events), dtype="int64")

    for k, i in enumerate(events):
        i = int(i)
        if i < 0 or i > last:
            raise IndexError(f"event index {i} out of bounds [0, {last}]")

        if vol is not None:
            v = vol_vals[i]
            if not np.isfinite(v):
                # No ex-ante vol available (e.g. warmup): cannot set barriers.
                rows_t1[k] = min(i + vertical_barrier, last)
                rows_ret[k] = px[rows_t1[k]] / px[i] - 1.0
                rows_lab[k] = 0
                continue
            upper = pt_mult * v
            lower = -sl_mult * v
        else:
            upper = pt_mult
            lower = -sl_mult

        vbar = min(i + vertical_barrier, last)
        entry = px[i]
        touch = vbar          # default: time-out at vertical barrier
        label = 0
        # Scan strictly after the entry bar: j in (i, vbar].
        for j in range(i + 1, vbar + 1):
            r = px[j] / entry - 1.0
            hit_up = r >= upper
            hit_dn = r <= lower
            if hit_up or hit_dn:
                touch = j
                # If both barriers are crossed at the same bar, attribute to the
                # larger move in magnitude (conservative tie-break).
                if hit_up and hit_dn:
                    label = 1 if r >= -r else -1
                else:
                    label = 1 if hit_up else -1
                break

        rows_t1[k] = touch
        rows_ret[k] = px[touch] / entry - 1.0
        rows_lab[k] = label

    out = pd.DataFrame(
        {"t1": rows_t1, "ret": rows_ret, "label": rows_lab},
        index=pd.Index(events, name="event"),
    )
    return out


# --------------------------------------------------------------------------- #
# Sample uniqueness (concurrency-based weights)
# --------------------------------------------------------------------------- #
def average_uniqueness(events: pd.DataFrame, n_bars: int) -> pd.Series:
    """Per-event average uniqueness in (0, 1] from label-span concurrency.

    Two labels overlap when their holding spans [start, t1] share bars; the
    shared bars carry redundant (non-IID) information. For each bar b, let
    concurrency c_b = number of events whose span covers b. An event spanning
    bars B has average uniqueness:

        u = mean over b in B of (1 / c_b)

    u = 1 means the event shares no bar with any other (fully unique). Smaller u
    means heavier overlap. These weights feed sequential bootstrapping and
    sample-weighting in CV / bagging (Lopez de Prado, ch. 4).

    Parameters
    ----------
    events : DataFrame whose index is the event START bar (as produced by
             `triple_barrier_labels`) and which has an integer `t1` column.
             The span of an event is the inclusive bar range [start, t1].
    n_bars : total number of bars in the underlying series (must cover every
             t1). Used to size the concurrency accumulator.

    Returns
    -------
    pd.Series of average uniqueness, indexed like `events.index`, values in
    (0, 1].
    """
    if "t1" not in events.columns:
        raise ValueError("events must have a 't1' column")
    starts = events.index.values.astype(int)
    t1 = events["t1"].values.astype(int)
    if len(starts) == 0:
        return pd.Series(dtype="float64", name="avg_uniqueness")
    if t1.max() >= n_bars or starts.min() < 0:
        raise ValueError("event spans fall outside [0, n_bars)")

    # Bar-level concurrency over inclusive spans [start, t1].
    conc = np.zeros(n_bars, dtype="int64")
    for s, e in zip(starts, t1):
        conc[s : e + 1] += 1

    u = np.empty(len(starts), dtype="float64")
    for k, (s, e) in enumerate(zip(starts, t1)):
        u[k] = np.mean(1.0 / conc[s : e + 1])
    return pd.Series(u, index=events.index, name="avg_uniqueness")


# --------------------------------------------------------------------------- #
# Meta-labeling
# --------------------------------------------------------------------------- #
def meta_label(side: int, realized_ret: float) -> int:
    """Secondary {0,1} target for meta-labeling.

    Given a primary model's bet SIDE (+1 long / -1 short) and the realized
    return of acting on that side, the meta-label is whether the bet would have
    made money:

        1 if side * realized_ret > 0   (correct directional call)
        0 otherwise                    (wrong, or exactly flat)

    The meta-model learns to size/filter bets (take vs skip); it never chooses
    the side. Train it on out-of-sample primary predictions to avoid leakage.
    """
    return 1 if side * realized_ret > 0 else 0


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _test_get_daily_vol() -> None:
    close = pd.Series([100.0, 101.0, 102.0, 100.0, 105.0, 103.0])
    vol = get_daily_vol(close, span=3)
    assert len(vol) == len(close)
    assert np.isnan(vol.iloc[0])           # no prior return
    assert np.all(np.isfinite(vol.iloc[2:].values))
    assert np.all(vol.iloc[2:].values >= 0.0)


def _test_fixed_horizon_labels() -> None:
    # Strictly rising: every 1-bar forward return is positive -> all +1.
    rising = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    lab = fixed_horizon_labels(rising, horizon=1, threshold=0.0)
    assert len(lab) == 4
    assert (lab.values == 1).all()

    # Strictly falling -> all -1.
    falling = pd.Series([1.4, 1.3, 1.2, 1.1, 1.0])
    lab = fixed_horizon_labels(falling, horizon=1, threshold=0.0)
    assert (lab.values == -1).all()

    # Inside the band -> 0. Moves of ~1% with a 5% threshold.
    flat = pd.Series([100.0, 100.5, 101.0, 100.7])
    lab = fixed_horizon_labels(flat, horizon=1, threshold=0.05)
    assert (lab.values == 0).all()

    # Horizon > 1 uses r over (t, t+horizon].
    px = pd.Series([100.0, 100.0, 100.0, 110.0])
    lab = fixed_horizon_labels(px, horizon=3, threshold=0.05)
    assert len(lab) == 1 and lab.iloc[0] == 1


def _test_triple_barrier_rising() -> None:
    # Monotonically rising by ~1%/bar. Absolute upper barrier at 2.5% must be
    # first crossed at the third bar after entry (cumulative ~3.03%).
    px = pd.Series([100.0 * (1.01 ** i) for i in range(10)])
    out = triple_barrier_labels(
        px, event_idx=[0], pt_mult=0.025, sl_mult=0.025,
        vertical_barrier=8, vol=None,
    )
    assert out.loc[0, "label"] == 1
    # r(0->1)=1.01%, r(0->2)=2.03%, r(0->3)=3.03% -> first >= 2.5% at bar 3.
    assert out.loc[0, "t1"] == 3
    assert out.loc[0, "ret"] > 0.025


def _test_triple_barrier_falling() -> None:
    px = pd.Series([100.0 * (0.99 ** i) for i in range(10)])
    out = triple_barrier_labels(
        px, event_idx=[0], pt_mult=0.025, sl_mult=0.025,
        vertical_barrier=8, vol=None,
    )
    assert out.loc[0, "label"] == -1
    # -0.99%, -1.97%, -2.94% -> first <= -2.5% at bar 3.
    assert out.loc[0, "t1"] == 3
    assert out.loc[0, "ret"] < -0.025


def _test_triple_barrier_timeout() -> None:
    # Flat-ish path that never reaches +/-5% within the window -> label 0 at the
    # vertical barrier.
    px = pd.Series([100.0, 100.5, 99.8, 100.2, 100.1, 99.9, 100.3])
    out = triple_barrier_labels(
        px, event_idx=[0], pt_mult=0.05, sl_mult=0.05,
        vertical_barrier=3, vol=None,
    )
    assert out.loc[0, "label"] == 0
    assert out.loc[0, "t1"] == 3            # min(0+3, last=6) = 3
    assert abs(out.loc[0, "ret"] - (px.iloc[3] / px.iloc[0] - 1.0)) < 1e-12


def _test_triple_barrier_no_lookahead() -> None:
    # The entry bar itself must never count as a touch even if its return
    # (trivially 0) would satisfy a zero barrier; scanning starts at i+1.
    px = pd.Series([100.0, 100.0, 100.0, 110.0])
    out = triple_barrier_labels(
        px, event_idx=[0], pt_mult=0.0, sl_mult=10.0,
        vertical_barrier=3, vol=None,
    )
    # upper = 0.0; first j>i with r>=0 is bar 1 (r=0) -> +1 at bar 1, not bar 0.
    assert out.loc[0, "t1"] == 1
    assert out.loc[0, "label"] == 1


def _test_triple_barrier_vol_scaled() -> None:
    px = pd.Series([100.0 * (1.01 ** i) for i in range(6)])
    vol = pd.Series(0.01, index=px.index)   # constant 1% vol
    out = triple_barrier_labels(
        px, event_idx=[0], pt_mult=2.0, sl_mult=2.0,
        vertical_barrier=5, vol=vol,
    )
    # upper = 2 * 0.01 = 2% ; crossed at bar 2 (r=2.01%).
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "t1"] == 2


def _test_average_uniqueness_nonoverlapping() -> None:
    # Disjoint spans: [0,1] and [2,3] -> every bar has concurrency 1 -> u == 1.
    ev = pd.DataFrame({"t1": [1, 3]}, index=pd.Index([0, 2], name="event"))
    u = average_uniqueness(ev, n_bars=4)
    assert np.allclose(u.values, 1.0)
    assert ((u.values > 0.0) & (u.values <= 1.0)).all()


def _test_average_uniqueness_overlapping() -> None:
    # Identical spans [0,2] for two events -> concurrency 2 on every shared bar
    # -> u == 0.5 for both. Strictly < 1 and within (0, 1].
    ev = pd.DataFrame({"t1": [2, 2]}, index=pd.Index([0, 0], name="event"))
    u = average_uniqueness(ev, n_bars=3)
    assert np.allclose(u.values, 0.5)
    assert ((u.values > 0.0) & (u.values < 1.0)).all()

    # Partial overlap: spans [0,2] and [1,3]. Concurrency = [1,2,2,1].
    #   event0 over bars {0,1,2}: mean(1, 1/2, 1/2) = 2/3
    #   event1 over bars {1,2,3}: mean(1/2, 1/2, 1) = 2/3
    ev2 = pd.DataFrame({"t1": [2, 3]}, index=pd.Index([0, 1], name="event"))
    u2 = average_uniqueness(ev2, n_bars=4)
    assert np.allclose(u2.values, 2.0 / 3.0)
    assert ((u2.values > 0.0) & (u2.values <= 1.0)).all()


def _test_meta_label() -> None:
    assert meta_label(1, 0.02) == 1     # long, market up -> correct
    assert meta_label(1, -0.02) == 0    # long, market down -> wrong
    assert meta_label(-1, -0.02) == 1   # short, market down -> correct
    assert meta_label(-1, 0.02) == 0    # short, market up -> wrong
    assert meta_label(1, 0.0) == 0      # flat outcome -> no profit


def _run_all_tests() -> None:
    _test_get_daily_vol()
    _test_fixed_horizon_labels()
    _test_triple_barrier_rising()
    _test_triple_barrier_falling()
    _test_triple_barrier_timeout()
    _test_triple_barrier_no_lookahead()
    _test_triple_barrier_vol_scaled()
    _test_average_uniqueness_nonoverlapping()
    _test_average_uniqueness_overlapping()
    _test_meta_label()


if __name__ == "__main__":
    _run_all_tests()
    print("labeling.py: all self-tests passed.")
