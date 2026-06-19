# Machine Learning for Alpha

Financial ML is not generic ML with price data. The data-generating process is non-stationary, adversarial, and dominated by noise; samples overlap and are not IID; the same dataset gets queried thousands of times during research. Default ML tooling assumes none of this and will produce backtests that look brilliant and trade like coin flips. This file covers how to build ML signals that survive contact with live markets, with leakage and overfitting avoidance as the through-line.

References used throughout: `templates/labeling.py` (triple-barrier, meta-labeling, sample weights), `templates/validation.py` (purged/embargoed CV, CPCV, constant-correlation shrinkage), `references/stats-risk.md` (Deflated Sharpe Ratio, Probability of Backtest Overfitting). House conventions: positions are lagged versus the returns they earn (`pnl_t = pos.shift(1) * ret_t`); the Information Coefficient is the per-date cross-sectional *rank* correlation (Spearman) of a factor at `t` against FORWARD returns (`fwd_h`, strictly returns of bars after `t`); annualized Sharpe = `mean(excess)/std(excess, ddof=1)*sqrt(ppy)` (per `templates/metrics.py`, which assumes IID returns — see `lo_annualization_factor` when returns are autocorrelated).

---

## 1. Why finance is hard for ML

Generic ML benchmarks (ImageNet, MNIST) have high signal-to-noise, IID samples, stationary distributions, and effectively unlimited labeled data. Markets violate every one of these. Understand the violations before reaching for a model.

| Property | Benchmark ML | Markets | Consequence |
|---|---|---|---|
| Signal-to-noise | High | Very low (R² of a good daily signal ~0.01–0.05) | Models memorize noise; need brutal regularization and skeptical evaluation |
| Stationarity | Stationary | Non-stationary, regime-switching | A model fit on 2017–2019 may be anti-predictive in 2020; distributions drift |
| Sample independence | IID | Overlapping, autocorrelated, serially dependent | Plain k-fold and naive bootstrap leak; standard errors are wildly optimistic |
| Effective sample size | Large | Small — overlapping labels and regimes shrink it far below row count | "1M rows" of minute bars may carry the information of a few hundred independent observations |
| Adversary | None | Other capitalized, adaptive participants | Discovered edges decay; the market adapts to your trades |
| Feature distribution | Fixed | Shifts (vol regimes, microstructure changes, tick-size changes) | Train-time scaling and thresholds go stale |

**Low signal-to-noise.** A daily equity factor with a rank IC of 0.03 is genuinely useful, yet a classifier predicting next-day up/down will sit near 50–52% accuracy. If your model reports 80% accuracy on next-bar direction, suspect leakage, not alpha. Calibrate expectations: the edge lives in the third decimal of correlation, harvested across many bets.

**Non-stationarity / regime change.** The relationship between features and returns is time-varying. A single train/test split assumes one stable mapping. Walk-forward and combinatorial CV exist precisely because the future is drawn from a shifting distribution. Always test whether performance concentrates in one regime.

**Non-IID overlapping samples.** If you label each observation by the return over the next 5 days, consecutive labels share 4 days of returns. They are not independent. This breaks the IID assumption behind cross-validation, bootstrapping, and most significance tests (including the `sqrt(ppy)` Sharpe annualization in `metrics.py`, which assumes IID returns). Sample weights and purging (Sections 5–6) are the fixes.

**Small effective sample size.** Effective N is governed by the number of *non-overlapping, non-redundant* observations across *distinct regimes*, not the number of rows. This is why deep nets usually lose to well-regularized gradient-boosted trees on tabular financial data.

**Efficient / adversarial markets.** Any persistent, easily-found pattern is arbitraged away. Edges are faint, conditional, and decaying. This raises the bar for what counts as a real discovery and is why multiple-testing correction (DSR/PBO) is mandatory, not optional.

---

## 2. Features: stationarity vs memory

ML estimators do not strictly *require* stationary features, but their generalization degrades sharply when the conditional distribution of `y | X` drifts between train and live, and tree splits / linear coefficients learned on price *levels* extrapolate to unseen ranges. Raw prices are non-stationary (they wander and trend), so models fit on price levels do not generalize. The naive fix — integer differencing (returns) — achieves stationarity but **erases memory**: returns are nearly serially uncorrelated, so the level information (where price sits relative to history) is gone. This is the central tension: stationarity vs memory.

### Fractional differentiation

Fractional differentiation applies a fractional order `d` of differencing (typically searched on `d ∈ [0, 1]`, though `d > 1` is well-defined). `d=0` is the raw (full-memory, non-stationary) series; `d=1` is ordinary first differencing of the input. The goal is the *minimum* `d` that passes a stationarity test (ADF), preserving as much memory as possible.

The fractionally differenced series is a weighted sum of lagged levels with binomially-derived weights:

```
w_0 = 1
w_k = -w_{k-1} * (d - k + 1) / k     for k = 1, 2, ...
X_frac_t = sum_{k>=0} w_k * X_{t-k}
```

Weights decay (for `0 < d < 1` they alternate in sign and shrink); the **fixed-width window (FFD)** variant truncates them once `|w_k|` falls below a threshold `tau`, giving a constant-width, well-defined kernel that does not look unboundedly far back.

```python
import numpy as np
import pandas as pd

def ffd_weights(d: float, tau: float = 1e-5, max_k: int = 10_000) -> np.ndarray:
    """Fixed-width fractional-diff weights, truncated at |w_k| < tau.
    Returned newest-first reversed so the LAST element aligns with X_t and the
    first with the oldest lag, matching the windowing in frac_diff_ffd."""
    w = [1.0]
    for k in range(1, max_k):
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < tau:
            break
        w.append(wk)
    return np.array(w[::-1])  # reverse: oldest lag first, X_t weight (w_0=1) last

def frac_diff_ffd(series: pd.Series, d: float, tau: float = 1e-5) -> pd.Series:
    """Causal FFD: X_frac_t uses only X_t, X_{t-1}, ... — no forward leakage."""
    w = ffd_weights(d, tau)
    width = len(w)
    out = pd.Series(index=series.index, dtype=float)
    vals = series.to_numpy(dtype=float)   # positional indexing on values, not labels
    for i in range(width - 1, len(series)):
        window = vals[i - width + 1 : i + 1]   # oldest..newest, aligns with w
        if np.any(np.isnan(window)):
            continue
        out.iloc[i] = float(np.dot(w, window))
    return out

def min_ffd_d(log_price: pd.Series, ds=np.linspace(0, 1, 21), alpha=0.05) -> float:
    """Smallest d whose FFD series rejects the ADF unit-root null at `alpha`.
    Choose d on TRAINING data only; apply the fixed d to later/test data."""
    from statsmodels.tsa.stattools import adfuller
    for d in ds:
        fd = frac_diff_ffd(log_price, d).dropna()
        if len(fd) < 50:
            continue
        # autolag=None with maxlag=1 is a fixed-lag ADF (fast, deterministic);
        # use autolag="AIC" for a data-driven lag if you can afford it.
        pval = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
        if pval < alpha:
            return float(d)
    return 1.0
```

Apply FFD to log-prices, find the minimum `d` per instrument (it varies — often well below 1, e.g. ~0.3–0.6 for many daily equity series, but verify per series), and feed the FFD series as a feature. You keep memory (the FFD series still correlates with the raw level) while keeping the feature roughly stationary. **Leakage note:** FFD at time `t` uses only `X_{t-k}` for `k>=0`, so it is causal — but choose `d` (and `tau`) on the training set and apply that fixed `d` to test data; do not re-fit `d` per fold using test observations.

### Structural breaks

Regime shifts (vol regime changes, policy shocks, microstructure changes like decimalization or tick-size rules) invalidate a fixed feature→return mapping. Detect them with CUSUM filters, explosiveness tests (SADF/Supremum ADF), or Chow-type break tests, and either (a) add a regime indicator feature, (b) restrict training to the current regime, or (c) downweight pre-break samples. Note that an in-sample break *date* is itself estimated and can be unstable — do not let a break test peek at test-period data when defining train-fold regimes. A model blind to a structural break will extrapolate a dead relationship.

### Scaling / normalization — fit on TRAIN ONLY

Any transform that learns parameters from data — standardization (`mean`, `std`), min-max, quantile/rank normalization, PCA, winsorization bounds, target encoding — **must be fit on the training fold only** and then applied to validation/test. Fitting a scaler on the full dataset leaks test-period statistics (and, with rolling features, future information) into training. In a `Pipeline`, put the scaler before the estimator so the CV splitter re-fits it per fold:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

pipe = Pipeline([
    ("scale", StandardScaler()),               # re-fit inside each CV fold
    ("clf", HistGradientBoostingClassifier()),
])
# Drive the fit/predict loop yourself with PurgedKFold / CombinatorialPurgedKFold
# from templates/validation.py (their .split yields purged train/test indices);
# pass `pipe` so the scaler is re-fit on each purged TRAIN fold only.
```

For cross-sectional features, normalize **within each timestamp** (rank or z-score across the universe at `t`), which is naturally point-in-time and avoids cross-period leakage. For time-series features, use only trailing windows.

### Point-in-time alignment and feature lags

Every feature value at decision time `t` must have been *knowable* at `t`.

- **Fundamentals / accounting data**: align by the *report/availability date*, not the fiscal-period-end date. A Q4 figure with period end Dec 31 may only be public in late February — use the as-reported timestamp. Survivorship- and restatement-free PIT databases matter here (see `templates/data_loader.py`).
- **Feature lag**: if a feature is computed from data through the close of `t`, the earliest you can trade on it is the next bar. Combined with the position-lag convention (`pnl_t = pos.shift(1) * ret_t`), make sure you are not double-counting or under-counting lags. Audit: shift a known-future column into your feature set; if performance jumps, you have look-ahead.
- **Vendor revisions**: macro series (GDP, employment) get revised. Use the first-print/vintage value available at `t`, not the final revised value.

---

## 3. Labeling

The label defines what the model learns to predict. Bad labels are the most common silent source of either non-predictability or leakage.

### Fixed-horizon labels and their flaw

The default approach: label observation `t` by the sign (or bucket) of the return over a fixed horizon `h`, often against fixed thresholds:

```
y_t = +1 if r_{t->t+h} >  threshold
      -1 if r_{t->t+h} < -threshold
       0 otherwise
```

**The flaw:** a fixed return threshold ignores volatility. A +1% move in a sleepy bond is a major event; in a meme stock during an earnings week it is noise. Fixed thresholds mislabel quiet periods as "no move" and volatile periods as "signal," so the model learns the volatility calendar, not predictive structure. They also ignore the *path* — a position that hits a stop-loss intraday before recovering is recorded as a winner.

### The triple-barrier method

Label each observation by which of three barriers is touched first over the holding window:

1. **Upper barrier** — profit-take, at `+pt * sigma_t`.
2. **Lower barrier** — stop-loss, at `-sl * sigma_t`.
3. **Vertical barrier** — a maximum holding time (time-out).

Horizontal barriers are **scaled by realized volatility** `sigma_t` (e.g., an EWMA of returns estimated *as of* `t`), so the label adapts to the regime: wider barriers in volatile periods, tighter when calm. The label is the sign of the first barrier touched; if the vertical barrier hits first, the label is the sign of the realized return at time-out (or 0). This respects the path and produces volatility-consistent, economically meaningful labels.

```python
def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """EWMA std of daily returns, computed causally (info as of t). ewm is
    backward-looking, so sigma_t uses only returns up to and including t."""
    ret = close / close.shift(1) - 1.0
    return ret.ewm(span=span).std()

def triple_barrier_labels(close, events, pt_sl=(1.0, 1.0), vol=None, vbar_days=5,
                          drop_unresolved=True):
    """
    events: DataFrame indexed by event start time t with optional 'side' column.
    pt_sl:  (profit_take_mult, stop_loss_mult) applied to `vol`.
    Returns label in {-1, 0, +1} = sign of first barrier touched.
    By default DROPS events whose vertical barrier extends past the data end
    (unresolved labels), instead of mislabeling them as a time-out at the edge.
    See templates/labeling.py for the production version (with t1, ret, bin).
    """
    if vol is None:
        vol = get_daily_vol(close)
    last_ts = close.index[-1]
    out = pd.Series(index=events.index, dtype=float)
    for t in events.index:
        t1 = t + pd.Timedelta(days=vbar_days)            # vertical barrier
        if drop_unresolved and t1 > last_ts:
            continue                                      # label cannot resolve
        path = close.loc[t:t1]                            # inclusive label slice
        if len(path) < 2:
            continue
        rets = path / close.loc[t] - 1.0
        side = events.at[t, "side"] if "side" in events.columns else 1.0
        rets = rets * side                                # orient by side if given
        up = pt_sl[0] * vol.loc[t]
        dn = -pt_sl[1] * vol.loc[t]
        # exclude the t=0 row (rets==0) so a zero-width barrier can't self-trigger
        hit_up = rets[(rets >  up) & (rets.index > t)].index.min()
        hit_dn = rets[(rets <  dn) & (rets.index > t)].index.min()
        first = min([x for x in [hit_up, hit_dn] if pd.notna(x)], default=None)
        if first is None:
            out.at[t] = float(np.sign(rets.iloc[-1]))     # timed out (may be 0)
        else:
            out.at[t] = 1.0 if first == hit_up else -1.0
    return out.dropna() if drop_unresolved else out
```

The vertical barrier `t1` (or the *actual* first-touch time when a horizontal barrier hits earlier) for each observation is what purging/embargo (Section 6) needs: it is the time at which the label is *resolved*, and any training sample whose `[t, t1]` window overlaps a test window must be purged. (The conceptual purge is by per-sample `[t, t1]`; the `PurgedKFold` in `templates/validation.py` approximates this with a single integer `label_horizon` — see Section 6 for the exact API.)

### Trend-scanning labels

When you want to label by the *existence and direction of a trend* rather than a fixed exit, use trend-scanning: for each `t`, fit a linear trend over each candidate forward horizon `[t, t+L]` for `L` in a range, and pick the horizon whose trend t-statistic is largest in absolute value. The label is the sign of that t-stat; its magnitude can serve as a sample weight or regression target. This adaptively finds the horizon at which a trend is statistically clearest, instead of imposing one. Note the look-back/look-forward horizons used per `t` define that sample's label window for purging.

### Look-ahead dangers in labels

Labels look forward by construction — that is correct (IC is defined against *forward* returns). The danger is letting forward information leak into **features or sample selection**:

- Do not compute the volatility scaling `sigma_t` using returns after `t`. Use a trailing/EWMA estimate as of `t`.
- Do not select which observations to label using future outcomes (e.g., "only keep events that eventually moved"). That is survivorship of the most blatant kind.
- When the label horizon extends past your dataset end, those observations have *unresolved* labels — drop them; never assume time-out at the edge (the `drop_unresolved` flag above).
- Remember the resolution time `t1` per sample; it is required for purging. A label that resolves inside a test fold contaminates any overlapping training sample.

---

## 4. Meta-labeling

Meta-labeling splits the decision into two models:

1. **Primary model** — decides the *side* (long/short, or whether a setup exists). This can be a simple rule, a technical signal, an existing factor, or another ML model. It is tuned for **recall**: catch all potential opportunities, tolerate false positives.
2. **Secondary (meta) model** — a binary classifier that decides **whether to act** on the primary signal (and, via its predicted probability, **how big**). Its label is whether the primary model's call would have been *correct* (e.g., a triple-barrier outcome of +1 conditioned on the primary's side). It is tuned for **precision**: filter out the primary's false positives.

```
primary side  ->  triple-barrier outcome (was the side right?)  ->  meta label in {0,1}
meta features ->  meta classifier  ->  P(act)  ->  size = f(P(act)), trade only if P > threshold
```

**Why it helps.** It is hard for one model to learn both *direction* and *confidence* under low SNR. Separating them lets the primary cast a wide net and the secondary raise precision by suppressing low-confidence trades. Note the tradeoff: gating *removes* trades, so it raises precision at the cost of recall — it improves risk-adjusted returns when the primary has real but noisy edge, not by magic. The meta-model's probability is also a natural bet-sizing input (e.g., size monotonically increasing in `P(act)` above the act threshold).

**When to use it.**
- You have a primary signal with decent recall but mediocre precision (too many bad trades).
- You want to add ML on top of a quant strategy you already trust for *direction* but want to filter.
- You want principled bet sizing from a calibrated probability (calibration is required — see Section 9).
- You need an interpretable separation between "what to trade" and "when to trust it" — useful for risk sign-off.

It is **not** a fix for a primary model with no edge: if the side is random, there is nothing for the secondary to gate (the meta-model can only learn to suppress, never to flip a coin-flip into alpha). Build/validate the primary first. See `templates/labeling.py` for the meta-label construction (label = 1 if primary side matches the realized barrier sign, else 0).

---

## 5. Sample weights

Overlapping labels break IID, which biases both fitting and evaluation. Weight samples to correct for redundancy and to emphasize informative observations.

**Concurrency and uniqueness.** Two labels whose `[t, t1]` windows overlap share return information; they are partially redundant. Compute, for each time bar, the number of labels concurrently "live" (`c_t`). A label's **average uniqueness** is the mean of `1/c_t` over its lifespan. Labels that live during crowded periods get lower uniqueness. Use average uniqueness as a base sample weight so the effective sample size reflects independent information, not raw row count.

**Return-attribution weights.** Weight each label by the magnitude of the (concurrency-adjusted) return it captures, so large, unique moves dominate fitting over trivial ones. Concretely, attribute to each bar only its `1/c_t` share of that bar's return and sum over the label's life, then take `|.|` — this avoids double-counting a return across overlapping labels. This focuses the model on observations that actually matter economically.

**Time decay.** Optionally decay weights with age so recent observations (closer to the live regime) count more, while older ones taper toward a floor. This hedges non-stationarity without discarding history outright.

**Sequential bootstrap.** Standard bootstrap (sampling rows IID with replacement) oversamples overlapping, redundant observations. Sequential bootstrap draws samples one at a time with probability that *down-weights* candidates overlapping already-drawn samples (the draw probability is proportional to the candidate's average uniqueness *given* the samples drawn so far), yielding a bootstrap set closer to IID. Use it for bagging (e.g., a bagged classifier / random-forest-style ensemble) on overlapping financial labels.

```python
# Sketch — see templates/labeling.py for the full, tested implementations.
def num_concurrent_events(bar_index, t1):
    """t1: Series mapping each event start -> its label end time. Returns the
    count of events live on each bar in bar_index."""
    counts = pd.Series(0, index=bar_index)
    for start, end in t1.items():
        counts.loc[start:end] += 1
    return counts

def average_uniqueness(t1, concurrency):
    """Mean of 1/c_t over each label's [start, end] span. Guard against any
    zero-concurrency bar (shouldn't happen if concurrency was built from t1)."""
    w = pd.Series(index=t1.index, dtype=float)
    for start, end in t1.items():
        c = concurrency.loc[start:end]
        w.loc[start] = (1.0 / c[c > 0]).mean()
    return w
```

Pass these weights to the estimator's `sample_weight` argument **and** use them in evaluation aggregation. Forgetting them produces optimistic significance and a model that over-fits crowded periods.

---

## 6. Cross-validation

### Why plain k-fold leaks here

Standard k-fold shuffles rows and assumes independence. In financial data:

- **Label overlap leaks across the split.** A training sample with window `[t, t1]` can overlap a test sample's window, so the model sees test-period returns during training.
- **Serial correlation leaks adjacency.** A test point sandwiched between training points is trivially interpolable.
- **Shuffling destroys time order**, mixing future into past.

The result: CV scores far above live performance. Never use vanilla `KFold`/`cross_val_score` on time-dependent financial labels.

### Purged + embargoed CV

Two corrections, both in `templates/validation.py`:

- **Purge**: remove from the training set any sample whose label window overlaps the test window. This eliminates information overlap from the label horizon. (Note: `PurgedKFold` purges by a single integer `label_horizon` — the number of forward bars in your label — applied on *both* sides of each contiguous test block, rather than per-sample `[t, t1]`. Set `label_horizon` to your longest label horizon to be safe.)
- **Embargo**: additionally drop training samples for a small buffer *after* the test window (e.g., 1% of the sample, or a few bars) to kill residual serial correlation that purging by label-window alone misses.

```python
# templates/validation.py exposes PurgedKFold(n_splits, *, embargo_pct, label_horizon)
# NOTE: the constructor takes an INTEGER `label_horizon` (forward bars in the
# label), NOT a per-sample t1 series. It does not import sklearn; .split yields
# integer-position train/test index arrays for contiguous, time-ordered folds.
from validation import PurgedKFold

cv = PurgedKFold(n_splits=6, embargo_pct=0.01, label_horizon=5)  # label uses next 5 bars
for tr_idx, te_idx in cv.split(X):
    Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
    model.fit(Xtr, ytr, sample_weight=w.iloc[tr_idx])
    score = evaluate(model, X.iloc[te_idx], y.iloc[te_idx])
```

### Combinatorial Purged CV (CPCV)

A single train/test path gives one estimate of OOS performance — fragile under non-stationarity. CPCV splits the data into `N` contiguous groups, then forms all `C(N, k)` combinations that leave out `k` groups for testing, with purge+embargo applied around each test group. This yields *many* backtest paths (`C(N-1, k-1)` distinct paths) from the same data, producing a **distribution** of OOS Sharpes instead of a point estimate. The per-path per-period returns are the columns you feed to `templates/overfitting.py:pbo_cscv` for the **shipped** Probability of Backtest Overfitting (PBO/CSCV) verdict and `performance_degradation` for the OOS-on-IS slope (see also `references/stats-risk.md` §1.5). CPCV is the recommended evaluation engine for ML alpha; `templates/validation.py` provides it as `CombinatorialPurgedKFold`.

```python
from validation import CombinatorialPurgedKFold

cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_groups=2,
                                embargo_pct=0.01, label_horizon=5)
# get_n_splits() == C(6,2) == 15 train/test combinations;
# n_paths() == C(5,1) == 5 reconstructable backtest paths.
for tr_idx, te_idx in cpcv.split(X):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=w.iloc[tr_idx])
    ...  # collect per-path OOS returns, then summarize the Sharpe distribution
```

### Walk-forward

Walk-forward trains on a trailing (expanding or rolling) window and tests strictly forward, marching through time. It is the most faithful to live deployment (you only ever use the past) and best exposes regime decay, at the cost of fewer, longer-horizon test paths and less data efficiency than CPCV. Use walk-forward as the final realism check; use CPCV/purged CV for model selection and significance. **All three require purging+embargo where label windows overlap fold boundaries** (in walk-forward, drop the `label_horizon` bars straddling the train→test boundary so a train label cannot reach into the test window).

---

## 7. Feature importance

Feature importance guides research and guards against spurious features — but the method matters, and substitution effects mislead.

- **MDI (Mean Decrease Impurity)** — tree-based, computed *in-sample* from how much each feature reduces node impurity. Fast, but **biased toward high-cardinality / continuous features** (they offer more split points) and computed on training data, so it rewards overfitting. Use only as a quick, in-sample sanity check, never as the final word.
- **MDA (Mean Decrease Accuracy / permutation importance)** — shuffle one feature in the **out-of-sample** fold and measure the drop in OOS performance. Model-agnostic, measures true predictive contribution, and is the **preferred** method. Run it *inside purged/CPCV folds* with sample weights so the OOS score is honest. A feature whose MDA is ~0 (or negative) is not earning its place. When features are autocorrelated in time, permute by shuffling *blocks* (or permuting across non-overlapping samples) rather than single rows, or the broken serial structure inflates the apparent importance.
- **SFI (Single Feature Importance)** — train/evaluate each feature *in isolation* (one-feature models) under CV. Immune to substitution effects (below), but blind to interactions a feature only expresses jointly.

**Substitution effects.** When two features are correlated (e.g., two momentum lookbacks), permuting one barely hurts accuracy because the other substitutes — so MDA *understates* both, and importance gets arbitrarily split between them. This makes a genuinely useful feature look discardable. Mitigate by clustering correlated features and computing importance per cluster (permute the whole cluster together), or use SFI to read standalone strength. Do not prune features on raw MDA when collinearity is present.

Compute importance under the *same* purged CV and weights you use for evaluation; in-sample importance on overlapping labels is meaningless.

---

## 8. Model choice and regularization

- **Trees / gradient-boosted machines (XGBoost, LightGBM, CatBoost) and bagged random forests** are the default for tabular financial data: they handle nonlinearity and interactions, are robust to monotone feature transforms and outliers, and regularize well with shallow depth and shrinkage. Pair bagging with **sequential bootstrap** (Section 5) to respect overlap.
- **Linear / regularized linear (Ridge, Lasso, ElasticNet)** models are strong baselines under low SNR — fewer parameters, less overfitting, interpretable coefficients. Often a well-regularized linear model on good features beats a tuned GBM; always benchmark against it.
- **Deep nets** rarely justify themselves on daily/low-frequency tabular data given the small effective sample size; they become relevant mainly with genuinely large, high-frequency or alternative-data sets, and demand even stricter leakage control.

**Regularization is non-negotiable** given the noise floor: shallow trees, high `min_samples_leaf` / `min_child_weight`, low learning rate with early stopping (on a *purged* validation fold — early stopping reads the validation score, so an un-purged validation fold leaks), subsampling of rows and columns, strong L1/L2, and a deliberately small feature set. Prefer the simplest model that captures the edge.

**Hyperparameter search INSIDE purged CV.** Tuning is itself a fitting procedure and a prime leakage/overfit vector. Run grid/random/Bayesian search using purged+embargoed CV (or CPCV) as the inner scorer — never plain CV, and never tune on the same folds you report final performance on. Use **nested CV**: an inner purged loop selects hyperparameters, an outer purged loop estimates honest OOS performance. Score the search on an **economic objective** (Sharpe of the resulting strategy net of costs), not classification accuracy (Section 9).

**Overfitting control.** Every hyperparameter combination you try is another test of the dataset. The number of configurations directly inflates the chance of a false discovery. Track how many models/configs you tried and deflate accordingly: report the **Deflated Sharpe Ratio** (`metrics.deflated_sharpe_ratio`) and **Probability of Backtest Overfitting** (`overfitting.pbo_cscv`, fed the CPCV path distribution). The `deflated_sharpe_ratio` in `metrics.py` assumes roughly *independent* trials — correlated grid points have a smaller *effective* `n_trials`, so passing the raw count is conservative but reasonable. A model that wins only after hundreds of configurations is almost certainly overfit.

---

## 9. Evaluation

**Economic metrics over classification metrics.** Accuracy, AUC, and F1 measure statistical fit, not money. A model can be 51% accurate and highly profitable (if it sizes up on high-conviction, high-payoff bets) or 60% accurate and unprofitable (if its wins are tiny and losses large). Convert predictions into positions and evaluate the **strategy**:

```python
# Predictions -> positions -> PnL, with the house position-lag convention.
pos = predictions_to_position(model_proba)        # e.g. (proba - 0.5) scaled, capped
pnl = pos.shift(1) * ret                           # pnl_t = pos_{t-1} * ret_t
# metrics.py: sharpe_ratio assumes IID returns; the sqrt(252) is the IID
# annualization. If pnl is autocorrelated, use lo_annualization_factor.
from metrics import sharpe_ratio
sharpe = sharpe_ratio(pnl.dropna(), risk_free=0.0, periods_per_year=252)
```

Report, on **out-of-sample / CPCV paths** and **net of realistic costs** (spread, commission, slippage, borrow): annualized Sharpe (the `metrics.py` convention above), turnover and cost drag, max drawdown, hit rate *and* average win/loss, and the rank IC of the raw prediction against forward returns (cross-sectional, per the house convention). For ML specifically, summarize the **distribution** of Sharpe across CPCV paths and report **DSR** (`metrics.deflated_sharpe_ratio`) / **PBO** (`overfitting.pbo_cscv`; see `references/stats-risk.md` §1.5) to account for the number of trials.

**Probability calibration.** If you size by predicted probability (meta-labeling, bet sizing), the probabilities must be calibrated — a "70%" must win ~70% of the time. Tree ensembles and many classifiers are not calibrated out of the box (boosting pushes probabilities toward 0/1; bagging shrinks them toward the base rate). Miscalibrated probabilities corrupt bet sizing even when the ranking (AUC) is fine: the position map `predictions_to_position(model_proba)` and any Kelly-style sizing read the *level* of `p`, not just its order.

Use `templates/calibration.py` (numpy-only, self-tested — no sklearn dependency):
- **Diagnose:** `reliability_curve(p, y, n_bins, strategy='quantile')` (quantile bins are robust when proba clusters), `expected_calibration_error` / `max_calibration_error`, `brier_score`, `log_loss`, and `brier_decomposition` (Murphy: `reliability − resolution + uncertainty` — the *reliability* term isolates miscalibration from sharpness, so you can tell a poorly-calibrated model from a low-resolution one).
- **Recalibrate:** `platt_scale(p_cal, y_cal)` (logistic squash of the logit; parametric, robust on small folds) or `isotonic_fit(p_cal, y_cal)` (monotone PAVA step map; flexible but data-hungry and overfits small folds). Each returns a closure `transform(p_new) → calibrated probs`.

**Fit the recalibrator on a held-out, PURGED calibration fold** that is disjoint from both the data the base model trained on *and* the test fold you report on. Recalibrating on the test fold trivially erases miscalibration — that is leakage (same Iron Law as the model itself), and it inflates the post-calibration Brier/ECE you quote. With overlapping labels, carve the calibration fold using `PurgedKFold` / `CombinatorialPurgedKFold` (purge+embargo by `label_horizon`) so a label resolving inside the calibration window can't reach a training or test sample. A clean recipe is nested: inner purged folds for model + calibrator fitting, an outer purged path for honest evaluation. Verify post-calibration with the reliability curve and a drop in ECE/log-loss on the *outer* (untouched) fold.

---

## 10. Pitfalls (detect / fix)

| Pitfall | How it sneaks in | Detect | Fix |
|---|---|---|---|
| **Leakage via scaling** | Fitting `StandardScaler`/PCA/quantile/winsorize bounds on the full dataset | Compare CV score with scaler fit on all-data vs per-fold; a gap = leak | Fit all learned transforms on TRAIN fold only; use a `Pipeline` inside CV (Section 2) |
| **Leakage via labels** | Vol scaling or event selection uses future returns; unresolved labels assumed timed-out at data edge | Implausibly high accuracy; results vanish when label `sigma_t` made strictly trailing | Trailing/EWMA vol as of `t`; drop unresolved labels; track `t1` per sample |
| **Leakage via features** | A feature uses data not knowable at `t` (final-revised fundamentals, period-end vs report date, rolling window peeking forward) | Inject a known-future column — if score jumps, you have look-ahead; audit each feature's timestamp | PIT alignment by availability date; trailing windows only; explicit feature lags |
| **Look-ahead in positions** | PnL computed with same-bar position instead of lagged | Returns suspiciously smooth/high; flip to `pos.shift(1)` and watch it collapse | `pnl_t = pos.shift(1) * ret_t` always |
| **Non-IID CV** | Plain k-fold / shuffled CV on overlapping labels | CV Sharpe >> walk-forward Sharpe | PurgedKFold / CombinatorialPurgedKFold with purge+embargo (`templates/validation.py`) |
| **Ignoring sample overlap** | IID `sample_weight`, standard bootstrap; effective N overstated | Significance tests far too confident vs CPCV path spread | Average-uniqueness + return-attribution weights; sequential bootstrap (Section 5) |
| **Overfit hyperparameter search** | Tuning on non-purged CV or on the reporting folds; hundreds of configs | OOS << inner-CV score; performance scales with number of configs tried | Nested purged CV; economic objective; count trials; deflate with DSR/PBO |
| **Train/test contamination** | Calibration/feature-selection/early-stopping done on test data; duplicate rows across splits | Re-run with a strictly isolated held-out set; gap reveals contamination | Separate purged calibration/validation folds; isolate the final test path |
| **Accuracy instead of PnL** | Optimizing AUC/F1; declaring victory on classification metrics | High accuracy, flat or negative net-of-cost equity curve | Optimize and report strategy Sharpe/IC net of costs (Section 9) |
| **Backtest overfitting from many tries** | Researcher tests many features/models/labels on the same data | Best result indistinguishable from best-of-N noise | CPCV path distribution; report DSR (`metrics.deflated_sharpe_ratio`) and PBO (`overfitting.pbo_cscv`); pre-register the test |
| **Substitution-masked importance** | Correlated features split/hide each other's MDA | Cluster correlations; importance unstable across folds | Cluster-level permutation importance; SFI cross-check (Section 7) |
| **Non-stationarity ignored** | One static train/test split; no regime check | Performance concentrated in one period; decays out of sample | Walk-forward + CPCV; structural-break detection; time-decay weights; regime features |
| **IID-Sharpe annualization** | `sqrt(252)` scaling on autocorrelated PnL (overlapping labels, trend models) | Realized live vol/Sharpe diverges from backtest | Use `lo_annualization_factor` (`metrics.py`); report effective-N–aware significance |

---

### Templates and references

- `templates/labeling.py` — triple-barrier, trend-scanning, meta-labeling, concurrency/uniqueness, return-attribution and time-decay weights, sequential bootstrap.
- `templates/validation.py` — `PurgedKFold`, embargo, `CombinatorialPurgedKFold` (CPCV), constant-correlation covariance shrinkage. (Walk-forward is described here; drive it with your own trailing-window loop plus a `label_horizon` purge at the train→test boundary.)
- `templates/overfitting.py` — `pbo_cscv` (Probability of Backtest Overfitting via CSCV), `performance_degradation` (OOS-on-IS slope, P[OOS loss]), `build_perf_matrix`. Feed it the CPCV path returns.
- `references/stats-risk.md` — Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO) §1.5 for multiple-testing correction.
- `templates/metrics.py` — Sharpe/drawdown/IC and PSR/DSR with the house conventions (IID-Sharpe caveat + `lo_annualization_factor`).
- `templates/calibration.py` — probability-calibration toolkit (numpy-only): reliability curves, ECE/MCE, Murphy Brier decomposition, Platt scaling, isotonic/PAVA. Calibrate meta-label / bet-sizing probabilities on a held-out purged fold before they drive position size (Section 9).
- `templates/factor_research.py`, `templates/backtest_skeleton.py`, `templates/data_loader.py` — PIT-correct data, factor IC pipeline, and cost-aware backtest scaffolding.

**The one rule that prevents most disasters:** whenever a result looks too good, assume leakage first and prove it isn't — by re-running with strictly causal features, purged CV, lagged positions, realistic costs, and a trial-count-deflated significance test — before you believe it.
