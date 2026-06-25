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
# Concurrency and return-attribution weights
# --------------------------------------------------------------------------- #
def num_concurrent_events(events: pd.DataFrame, n_bars: int) -> pd.Series:
    """Bar-level concurrency: how many event spans cover each bar.

    For each bar b, c_b = number of events whose inclusive holding span
    [start, t1] contains b (Lopez de Prado, ch. 4). This is the building block
    for `average_uniqueness`, `return_attribution_weights`, and time-decay.

    CAUSALITY: c_b depends only on event START bars and their first-touch `t1`
    bars (which are themselves realized strictly after the entry). It mixes no
    information outside the union of the event spans, so a per-event statistic
    derived from it uses only that event's own [start, t1] window.

    Parameters
    ----------
    events : DataFrame indexed by event START bar with an integer `t1` column
             (as produced by `triple_barrier_labels`).
    n_bars : total bars in the underlying series (must cover every t1).

    Returns
    -------
    pd.Series of integer concurrency, length `n_bars`, indexed 0..n_bars-1.
    Bars covered by no event have count 0.
    """
    if "t1" not in events.columns:
        raise ValueError("events must have a 't1' column")
    starts = events.index.values.astype(int)
    t1 = events["t1"].values.astype(int)
    conc = np.zeros(n_bars, dtype="int64")
    if len(starts) == 0:
        return pd.Series(conc, index=pd.RangeIndex(n_bars), name="concurrency")
    if t1.max() >= n_bars or starts.min() < 0:
        raise ValueError("event spans fall outside [0, n_bars)")
    if np.any(t1 < starts):
        raise ValueError("each t1 must be >= its start bar")
    for s, e in zip(starts, t1):
        conc[s : e + 1] += 1
    return pd.Series(conc, index=pd.RangeIndex(n_bars), name="concurrency")


def return_attribution_weights(
    events: pd.DataFrame, close: pd.Series
) -> pd.Series:
    """Sample weights by absolute log-return attributed over concurrency.

    A label's importance should reflect the magnitude of the (log) return
    realized over its span, but each bar's return is shared among all events
    concurrent at that bar. Following Lopez de Prado ch. 4, attribute to each
    event only its 1/c_b share of every bar's log return inside its span:

        w_raw_event = | sum over b in (start, t1] of  log_ret_b / c_b |

    where log_ret_b = log(close[b] / close[b-1]) and c_b is the concurrency at
    bar b. Raw weights are then scaled so they sum to the number of events
    (mean weight 1), the convention expected by sklearn-style `sample_weight`.

    CAUSALITY: each event's weight uses only bar returns inside its own
    [start+1, t1] window and the concurrency over that same window; nothing is
    read past `t1`. Safe as a backtest / CV sample weight.

    Parameters
    ----------
    events : DataFrame indexed by event START bar with integer `t1` column.
    close  : price series (positional bars) covering every t1.

    Returns
    -------
    pd.Series of non-negative weights summing to len(events), indexed like
    `events.index`. Empty events -> empty series.
    """
    if "t1" not in events.columns:
        raise ValueError("events must have a 't1' column")
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pd.Series")
    starts = events.index.values.astype(int)
    if len(starts) == 0:
        return pd.Series(dtype="float64", name="ret_weight")
    n_bars = len(close)
    conc = num_concurrent_events(events, n_bars).values.astype("float64")
    px = close.values.astype(float)
    log_ret = np.zeros(n_bars, dtype="float64")
    log_ret[1:] = np.log(px[1:] / px[:-1])
    t1 = events["t1"].values.astype(int)

    w = np.empty(len(starts), dtype="float64")
    for k, (s, e) in enumerate(zip(starts, t1)):
        # Sum over bars strictly after entry: (s, e]. Concurrency >= 1 there.
        seg = slice(s + 1, e + 1)
        c = conc[seg]
        # Guard the degenerate empty span (e == s): zero realized return.
        if c.size == 0:
            w[k] = 0.0
        else:
            w[k] = abs(np.sum(log_ret[seg] / c))

    total = w.sum()
    if total > 0.0:
        w = w * (len(starts) / total)
    return pd.Series(w, index=events.index, name="ret_weight")


# --------------------------------------------------------------------------- #
# Time-decay weights
# --------------------------------------------------------------------------- #
def time_decay_weights(
    av_uniqueness: pd.Series, clf_last_w: float = 1.0
) -> pd.Series:
    """Linear time-decay multipliers over cumulative uniqueness (recency bias).

    Newer observations are weighted up and older ones down, with decay applied
    in the dimension of CUMULATIVE average uniqueness rather than raw time, so
    that highly overlapping (redundant) periods do not dominate the decay axis
    (Lopez de Prado, ch. 4, snippet 4.11).

    Let x be the cumulative sum of `av_uniqueness` ordered oldest->newest, and
    X = x[-1] its maximum. The weight is the line d(x) = slope * x + const with:
        clf_last_w >= 0 : d(X) = 1 and d(0) = clf_last_w
                          slope = (1 - clf_last_w) / X,  const = clf_last_w
        clf_last_w <  0 : the OLDEST fraction has its weight clipped to 0
                          slope = 1 / ((clf_last_w + 1) * X),  const = 1 - slope*X
                          and d is floored at 0 (so d(X) = 1 still holds).
    The newest observation always gets weight 1.

    Monotonicity: with clf_last_w in [0, 1] the weights are non-decreasing from
    oldest to newest (strictly increasing when clf_last_w < 1 and uniqueness is
    positive), so more recent samples carry at least as much weight.

    CAUSALITY: this is a weighting over already-realized, ordered events; it
    reads no future prices. The ordering must be by event time so that the
    "newest" end is the most recent realized label.

    Parameters
    ----------
    av_uniqueness : per-event average uniqueness (from `average_uniqueness`),
                    ordered oldest -> newest.
    clf_last_w    : weight applied to the OLDEST observation when in [0, 1].
                    Values in [-1, 0) clip the oldest fraction to zero weight.
                    clf_last_w == 1 -> all weights 1 (no decay).

    Returns
    -------
    pd.Series of non-negative decay weights, indexed like `av_uniqueness`.
    """
    if not isinstance(av_uniqueness, pd.Series):
        raise TypeError("av_uniqueness must be a pd.Series")
    if clf_last_w < -1.0:
        raise ValueError("clf_last_w must be >= -1")
    if len(av_uniqueness) == 0:
        return pd.Series(dtype="float64", name="time_decay_w")

    x = av_uniqueness.cumsum().values.astype("float64")
    X = x[-1]
    if X <= 0.0:
        # No uniqueness mass: fall back to flat weights of 1.
        return pd.Series(
            np.ones(len(x)), index=av_uniqueness.index, name="time_decay_w"
        )

    if clf_last_w >= 0.0:
        slope = (1.0 - clf_last_w) / X
        const = clf_last_w
    else:
        slope = 1.0 / ((clf_last_w + 1.0) * X)
        const = 1.0 - slope * X
    d = const + slope * x
    d[d < 0.0] = 0.0
    return pd.Series(d, index=av_uniqueness.index, name="time_decay_w")


# --------------------------------------------------------------------------- #
# Sequential bootstrap (uniqueness-aware resampling)
# --------------------------------------------------------------------------- #
def _indicator_matrix(events: pd.DataFrame, n_bars: int) -> np.ndarray:
    """Binary (n_bars x n_events) span-membership matrix: 1 where bar in span.

    Column j has 1s on every bar in event j's inclusive span [start_j, t1_j].
    """
    starts = events.index.values.astype(int)
    t1 = events["t1"].values.astype(int)
    ind = np.zeros((n_bars, len(starts)), dtype="float64")
    for j, (s, e) in enumerate(zip(starts, t1)):
        ind[s : e + 1, j] = 1.0
    return ind


def sequential_bootstrap(
    events: pd.DataFrame,
    n_bars: int,
    size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw event indices with probability that favors UNIQUE (non-overlapping).

    Standard iid bootstrap oversamples clustered, highly-overlapping events. The
    sequential bootstrap (Lopez de Prado, ch. 4, snippets 4.4-4.5) draws one
    event at a time; before each draw it recomputes, for every event j, its
    average uniqueness GIVEN the events already drawn, and samples j with
    probability proportional to that conditional uniqueness. The result is a
    sample whose realized average uniqueness is materially higher than an iid
    draw, reducing redundancy in bagged/CV estimators.

    Algorithm (correct indicator-matrix form):
        ind[b, j] = 1 if bar b in span of event j.
        Given already-drawn columns, c_b = 1 + sum of drawn ind columns at bar b
        is the concurrency that a candidate would face. For candidate j,
            u_j = mean over bars in span_j of (ind[b, j] / c_b),
        i.e. its average uniqueness if added next. Sample j with prob u_j / sum_k u_k.

    CAUSALITY: operates purely on event span geometry (start, t1) already
    realized; it draws nothing from future prices. The returned indices are
    positional rows into `events` for use as a bagging sample.

    Parameters
    ----------
    events : DataFrame indexed by event START bar with integer `t1` column.
    n_bars : total bars in the underlying series (must cover every t1).
    size   : number of draws (default: number of events).
    rng    : optional numpy Generator for reproducibility.

    Returns
    -------
    np.ndarray of positional event indices (length `size`), drawn WITH
    replacement under the uniqueness-weighted scheme.
    """
    if "t1" not in events.columns:
        raise ValueError("events must have a 't1' column")
    n_events = len(events)
    if n_events == 0:
        return np.empty(0, dtype="int64")
    if rng is None:
        rng = np.random.default_rng()
    if size is None:
        size = n_events

    ind = _indicator_matrix(events, n_bars)        # (n_bars, n_events)
    span_len = ind.sum(axis=0)                      # bars per event (>= 1)
    span_len[span_len == 0.0] = 1.0                # guard empty spans

    # Running concurrency contributed by already-drawn events (excludes the +1
    # for the candidate itself, which is added inside the loop).
    conc = np.zeros(ind.shape[0], dtype="float64")
    drawn = np.empty(size, dtype="int64")

    for d in range(size):
        # For each candidate j, average uniqueness if it were added next:
        #   c_b(candidate) = conc_b + 1 on bars in its span.
        # u_j = (1/|span_j|) * sum_{b in span_j} 1 / (conc_b + 1)
        inv = 1.0 / (conc + 1.0)                    # (n_bars,)
        u = (ind * inv[:, None]).sum(axis=0) / span_len  # (n_events,)
        prob = u / u.sum()
        j = int(rng.choice(n_events, p=prob))
        drawn[d] = j
        conc += ind[:, j]                           # commit the draw
    return drawn


# --------------------------------------------------------------------------- #
# Trend-scanning labels
# --------------------------------------------------------------------------- #
def trend_scanning_labels(
    close: pd.Series, span: tuple[int, int]
) -> pd.DataFrame:
    """Label each bar by the sign + strength of its most significant trend.

    For each anchor bar t, fit a straight line  close[t..t+L] ~ a + b * k  by
    ordinary least squares for every horizon L in [span[0], span[1]] (in bars),
    and keep the horizon whose slope t-statistic is largest in magnitude. The
    label is sign(best slope); `ret` records the slope's t-stat (a signed trend
    strength), `t1` the end bar of the chosen window (Lopez de Prado, "Machine
    Learning for Asset Managers", ch. 5). t-stat of the slope uses the closed
    form  t = b / se(b), se(b) = sqrt( s^2 / Sxx ),  with s^2 the residual
    variance and Sxx the centered sum of squares of the regressor.

    CAUSALITY NOTE: the label at bar t is a FORWARD-looking outcome (it reads
    close[t..t+L]) and is realized only at `t1`, exactly like a triple-barrier
    label. It is a TARGET, not a feature: never feed it (or `ret`) into a model
    as an input observable at t. Downstream CV must purge/embargo on `t1`.

    Parameters
    ----------
    close : price series (positional bars).
    span  : (L_min, L_max) inclusive horizon range in bars, L_min >= 3 so the
            slope t-stat has >= 1 residual degree of freedom (n - 2 >= 1).

    Returns
    -------
    pd.DataFrame indexed by anchor bar t (only bars with a full L_max window
    available), columns:
        t1    : int   end bar of the most-significant window
        ret   : float signed slope t-statistic (trend strength)
        label : int   sign of the slope in {-1, 0, 1}
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pd.Series")
    l_min, l_max = int(span[0]), int(span[1])
    if l_min < 3 or l_max < l_min:
        raise ValueError("span must satisfy 3 <= L_min <= L_max")

    px = close.values.astype(float)
    n = len(px)
    anchors = []
    rows_t1 = []
    rows_ret = []
    rows_lab = []

    for t in range(0, n - l_max):
        best_abs_t = -1.0
        best_t1 = t
        best_tstat = 0.0
        for L in range(l_min, l_max + 1):
            end = t + L                      # inclusive end bar
            y = px[t : end + 1]
            m = y.size                       # m = L + 1 points
            k = np.arange(m, dtype="float64")
            k_mean = k.mean()
            kc = k - k_mean
            sxx = np.dot(kc, kc)             # > 0 since m >= 4
            y_mean = y.mean()
            b = np.dot(kc, y - y_mean) / sxx
            a = y_mean - b * k_mean
            resid = y - (a + b * k)
            dof = m - 2
            s2 = np.dot(resid, resid) / dof
            if s2 <= 0.0:
                # Perfect fit: infinite t-stat -> treat as maximally significant.
                tstat = np.inf if b > 0 else (-np.inf if b < 0 else 0.0)
            else:
                se_b = np.sqrt(s2 / sxx)
                tstat = b / se_b
            if abs(tstat) > best_abs_t:
                best_abs_t = abs(tstat)
                best_t1 = end
                best_tstat = tstat
        anchors.append(t)
        rows_t1.append(best_t1)
        # np.sign handles +/-inf; store a large finite stand-in for printing.
        rows_ret.append(float(best_tstat))
        rows_lab.append(int(np.sign(best_tstat)))

    return pd.DataFrame(
        {"t1": rows_t1, "ret": rows_ret, "label": rows_lab},
        index=pd.Index(anchors, name="event"),
    )


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


def _test_num_concurrent_events() -> None:
    # Hand-built example: spans [0,2] and [1,3] over 5 bars.
    #   bar:  0  1  2  3  4
    #   ev0:  x  x  x
    #   ev1:     x  x  x
    #   conc: 1  2  2  1  0
    ev = pd.DataFrame({"t1": [2, 3]}, index=pd.Index([0, 1], name="event"))
    c = num_concurrent_events(ev, n_bars=5)
    assert list(c.values) == [1, 2, 2, 1, 0]
    # Three identical spans [0,1] -> concurrency 3 on bars 0,1 then 0.
    ev2 = pd.DataFrame({"t1": [1, 1, 1]}, index=pd.Index([0, 0, 0], name="event"))
    c2 = num_concurrent_events(ev2, n_bars=3)
    assert list(c2.values) == [3, 3, 0]
    # Empty.
    empty = pd.DataFrame({"t1": []}, index=pd.Index([], name="event"))
    assert list(num_concurrent_events(empty, n_bars=2).values) == [0, 0]


def _test_return_attribution_weights() -> None:
    # Two disjoint events on a path; each spans one log return. With no overlap,
    # raw weights are the absolute log returns; normalized to mean 1.
    close = pd.Series([100.0, 110.0, 110.0, 99.0])
    ev = pd.DataFrame({"t1": [1, 3]}, index=pd.Index([0, 2], name="event"))
    w = return_attribution_weights(ev, close)
    assert abs(w.sum() - 2.0) < 1e-12          # sums to n_events
    # raw0 = |log(110/100)|, raw1 = |log(99/110)|; bigger move -> bigger weight.
    r0 = abs(np.log(110.0 / 100.0))
    r1 = abs(np.log(99.0 / 110.0))
    assert (w.iloc[0] > w.iloc[1]) == (r0 > r1)
    # Overlap halves the attribution: two identical spans [0,1] share bar 1.
    ev2 = pd.DataFrame({"t1": [1, 1]}, index=pd.Index([0, 0], name="event"))
    w2 = return_attribution_weights(ev2, pd.Series([100.0, 110.0]))
    assert abs(w2.sum() - 2.0) < 1e-12
    assert np.allclose(w2.values, 1.0)         # symmetric -> equal weights


def _test_time_decay_weights_monotonic() -> None:
    au = pd.Series([1.0, 1.0, 1.0, 1.0], index=[0, 1, 2, 3])
    # clf_last_w in [0,1): strictly increasing oldest -> newest, newest == 1.
    w = time_decay_weights(au, clf_last_w=0.5)
    assert np.all(np.diff(w.values) > 0.0)
    assert abs(w.iloc[-1] - 1.0) < 1e-12
    # au=[1,1,1,1] -> x=[1,2,3,4], X=4, slope=0.125, const=0.5 -> d[0]=0.625.
    assert abs(w.iloc[0] - 0.625) < 1e-12
    # clf_last_w == 1 -> no decay, all ones (non-decreasing, not strict).
    w1 = time_decay_weights(au, clf_last_w=1.0)
    assert np.allclose(w1.values, 1.0)
    # clf_last_w < 0 clips the oldest fraction to 0 but stays non-decreasing.
    wc = time_decay_weights(au, clf_last_w=-0.5)
    assert np.all(np.diff(wc.values) >= -1e-12)
    assert wc.iloc[0] == 0.0
    assert abs(wc.iloc[-1] - 1.0) < 1e-12


def _test_sequential_bootstrap_raises_uniqueness() -> None:
    # Build a set with heavy redundancy: many overlapping events plus a few
    # unique ones. Sequential bootstrap should achieve higher realized average
    # uniqueness than an iid (uniform) draw, in expectation over many trials.
    rng = np.random.default_rng(0)
    n_bars = 12
    # 6 events: 0..3 all span [0,4] (very redundant), 4 spans [5,6], 5 spans [7,8].
    starts = [0, 0, 0, 0, 5, 7]
    t1 = [4, 4, 4, 4, 6, 8]
    ev = pd.DataFrame({"t1": t1}, index=pd.Index(starts, name="event"))
    u_all = average_uniqueness(ev, n_bars=n_bars).values

    def realized_avg_uniqueness(idx: np.ndarray) -> float:
        # Average uniqueness of the SAMPLED multiset, recomputed on its spans.
        sub = ev.iloc[idx].reset_index(drop=True)
        sub.index.name = "event"
        # Reindex spans by a synthetic event id but keep original geometry.
        sub2 = pd.DataFrame(
            {"t1": ev["t1"].values[idx]},
            index=pd.Index(ev.index.values[idx], name="event"),
        )
        return float(average_uniqueness(sub2, n_bars=n_bars).mean())

    trials = 200
    seq_u = np.empty(trials)
    iid_u = np.empty(trials)
    for k in range(trials):
        s_idx = sequential_bootstrap(ev, n_bars=n_bars, size=6, rng=rng)
        i_idx = rng.integers(0, len(ev), size=6)
        seq_u[k] = realized_avg_uniqueness(s_idx)
        iid_u[k] = realized_avg_uniqueness(i_idx)
    assert seq_u.mean() > iid_u.mean()
    # Sanity: a single full-uniqueness reference is in (0, 1].
    assert ((u_all > 0.0) & (u_all <= 1.0)).all()
    # Indices are valid positional rows.
    s_idx = sequential_bootstrap(ev, n_bars=n_bars, size=10, rng=rng)
    assert s_idx.min() >= 0 and s_idx.max() < len(ev)


def _test_trend_scanning_labels() -> None:
    # Strictly rising line -> positive slope -> label +1 at every anchor.
    rising = pd.Series([float(i) for i in range(12)])
    out = trend_scanning_labels(rising, span=(3, 5))
    assert (out["label"].values == 1).all()
    assert (out["ret"].values > 0).all()
    # Strictly falling -> label -1.
    falling = pd.Series([float(-i) for i in range(12)])
    out2 = trend_scanning_labels(falling, span=(3, 5))
    assert (out2["label"].values == -1).all()
    assert (out2["ret"].values < 0).all()
    # Analytic anchor: noiseless line y = 2k. Slope b = 2 exactly, residuals 0,
    # so the t-stat is +inf -> label +1, and t1 is within the scanned window.
    line = pd.Series([2.0 * i for i in range(8)])
    out3 = trend_scanning_labels(line, span=(3, 4))
    t = out3.index[0]
    assert out3.loc[t, "label"] == 1
    assert out3.loc[t, "t1"] in (t + 3, t + 4)
    # t1 always realized strictly after the anchor (forward-looking outcome).
    assert (out["t1"].values > out.index.values).all()


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
    _test_num_concurrent_events()
    _test_return_attribution_weights()
    _test_time_decay_weights_monotonic()
    _test_sequential_bootstrap_raises_uniqueness()
    _test_trend_scanning_labels()
    _test_meta_label()


if __name__ == "__main__":
    _run_all_tests()
    print("labeling.py: all self-tests passed.")
