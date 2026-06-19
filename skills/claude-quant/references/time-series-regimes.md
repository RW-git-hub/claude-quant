# Time-Series & Regime Modeling for Quant

Volatility forecasting, regime detection, change-point detection, and state-space filtering — with formulas, conventions, and detect/fix framing. The unifying hazard throughout: **any estimate (vol, regime probability, hedge ratio) used to size or place a trade at time `t` must be a function of information available at `t-1` only.** Most "alpha" from regime models is leakage.

Companion code: vol-targeting position sizing lives in `templates/backtest_skeleton.py` (`vol_target_sizer`, which correctly lags realized vol by one bar) and cost accounting for the resulting turnover in `templates/costs.py`. Out-of-sample tuning of any model below should go through `templates/validation.py`, which provides `PurgedKFold` and `CombinatorialPurgedKFold` (Lopez de Prado CPCV) — purged + embargoed cross-validation for time-ordered data. (Runnable EWMA/GARCH/HMM/Kalman/CUSUM implementations are in `templates/regime.py`.)

Conventions used here: simple returns `r_t = P_t/P_{t-1} - 1`; positions lagged vs the returns they earn (`pnl_t = pos.shift(1) * ret_t`); annualization factor `ppy` (daily=252). Volatility annualizes as `sigma_ann = sigma_daily * sqrt(ppy)` (random-walk / iid scaling — breaks under autocorrelation in returns *or* in the variance process; see ARIMA section). Note: variance models below (EWMA, GARCH, RV) are conventionally written in **log** returns, where multi-period variances add cleanly; the simple-vs-log distinction is second-order at daily frequency but matters for RV aggregation and overnight gaps.

---

## 1. Volatility modeling

### Volatility clustering
Returns are approximately uncorrelated (`ACF(r_t) ≈ 0` beyond lag 0), but `|r_t|` and `r_t^2` are strongly positively autocorrelated and slowly decaying. Large moves cluster: a big `|r|` today predicts a big `|r|` tomorrow. This is the empirical fact every vol model exists to capture. It also means **vol is forecastable even when returns are not** — the tradable edge in this whole document lives in the second moment, not the first.

### EWMA / RiskMetrics
Exponentially weighted moving variance — the simplest persistent-vol estimator, with one fixed decay parameter (not fitted in classic RiskMetrics):

```
sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}     # lam ~ 0.94 daily, ~0.97 monthly
```

`lam` (decay) sets memory: the weight on the lag-`k` past squared return is `(1-lam)*lam^k`, so the center-of-mass (mean lag) ≈ `lam/(1-lam)` periods (0.94 → ~15.7 ≈ 16 days). Higher `lam` = smoother, slower to react. Note the index: `sigma^2_t` (forecast for day `t`) depends on `r^2_{t-1}` and `sigma^2_{t-1}` — strictly past info, so it is correctly lagged *by construction* when used to size a position entered at the close of `t-1` for day `t`.

```python
import numpy as np
import pandas as pd

def ewma_vol(rets: pd.Series, lam: float = 0.94, init: float | None = None) -> pd.Series:
    """Returns sigma_t (std) usable to size day t.
    Output at index t uses only returns up to t-1 (no look-ahead)."""
    r2 = (rets ** 2).to_numpy()
    out = np.empty(len(rets))
    # Burn-in seed. NOTE: seeding from the first ~1/(1-lam) obs technically peeks
    # at early sample returns; acceptable as warm-up but DROP/IGNORE the burn-in
    # region before evaluating PnL, and never reuse the seed window in OOS metrics.
    if init is None:
        seed_n = max(1, int(round(1 / (1 - lam))))
        prev = float(np.nanmean(r2[:seed_n]))
    else:
        prev = float(init)
    for i in range(len(rets)):
        out[i] = prev                       # forecast for day i (uses info <= i-1)
        prev = lam * prev + (1 - lam) * r2[i]   # roll in r_i for next step
    return pd.Series(np.sqrt(out), index=rets.index)
```
The pandas one-liner `rets.ewm(alpha=1-lam).std()` includes `r_t` in the estimate at index `t` — **off by one, leaks.** `.shift(1)` it before using it to size day `t`. (Also note `ewm(...).std()` uses a bias-corrected sample std with a different weighting normalization than the RiskMetrics recursion above, so the two will not match exactly even after shifting.)

### GARCH(1,1)
Adds a constant (long-run anchor) so vol mean-reverts to a finite level instead of drifting:

```
sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
```

- **Persistence** `= alpha + beta`. Must be `< 1` for covariance stationarity (a finite unconditional variance). Empirically equity daily persistence ≈ 0.98–0.995. (Constraints: `omega > 0`, `alpha >= 0`, `beta >= 0`.)
- **Unconditional variance** `uncond_var = omega / (1 - alpha - beta)`.
- **Variance targeting:** fix `omega = (1 - alpha - beta) * uncond_var` with `uncond_var` set to the sample variance; estimate only `alpha, beta`. More robust than free `omega` (which is tiny and badly identified).
- EWMA is the special case `omega=0, alpha=1-lam, beta=lam` — i.e. an IGARCH with persistence exactly 1 and no mean reversion (no finite unconditional variance).

`k`-step-ahead forecast mean-reverts geometrically toward `uncond_var`. Because `sigma^2_{t+1}` is already known at time `t` (it is a function of `r_t` and `sigma^2_t`), the exponent is `k-1`, not `k`:
```
E[sigma^2_{t+k} | t] = uncond_var + (alpha+beta)^(k-1) * (sigma^2_{t+1} - uncond_var)    # k >= 1
```
Check: at `k=1` this returns `sigma^2_{t+1}` exactly (the known one-step forecast), as it must. This mean reversion is *the* exploitable structure: vol high → expect it to fall; low → expect it to rise. Half-life of a vol shock = `ln(0.5)/ln(alpha+beta)` (persistence 0.98 → ~34 days).

Use `arch` (`from arch import arch_model; arch_model(100*rets, vol="GARCH", p=1, q=1).fit()`). Scale returns to ~percent units for numerical stability, and divide forecast variances back by `100**2` before use. For fat tails / leverage, GARCH-t (`dist="t"`) or GJR-GARCH / EGARCH (asymmetric: negative returns raise vol more) usually beats plain Gaussian GARCH on equities/indices. For a genuine OOS forecast, refit on a rolling window and use `forecast(horizon=...)`, not in-sample conditional volatility.

### Realized volatility from intraday
With intraday data, estimate today's vol directly from high-frequency squared (log) returns rather than from a single daily return:
```
RV_t = sum_i r_{t,i}^2          # r_{t,i} = intraday LOG returns within day t
sigma_realized_t = sqrt(RV_t)
```
- Sampling frequency trade-off: finer sampling → lower estimator variance under the ideal model, but **microstructure noise** (bid-ask bounce, discreteness) biases `RV` upward and the bias grows as sampling gets finer. Common fixes: sample at 5-min (the classic compromise), use **realized kernels** or **two-scale RV (TSRV)** to de-bias, or **pre-averaging**.
- `RV` is a far more accurate proxy for latent vol than `r_t^2`; **HAR-RV** (regress `RV_t` on `RV_{t-1}`, the trailing weekly avg, and the trailing monthly avg) is a strong, simple daily-vol forecaster and a good baseline to beat GARCH. When forecasting, all RHS terms must end at `t-1` (no contemporaneous `RV_t` on the right).
- Overnight gaps: intraday RV computed over the trading session misses the close-to-open jump. Add an overnight (squared close-to-open) variance term or scale up; otherwise you systematically under-forecast for assets with big gaps (single-name equities, anything with scheduled overnight events). 24h markets (crypto, FX) have no gap but do have a session/weekend seasonality in vol — deseasonalize (e.g. by time-of-day) before modeling.

### CRITICAL: lag the vol estimate
A vol estimate used to size a position held over `[t, t+1)` must use only returns realized through `t`. Concretely: if `sigma_hat[t]` already incorporates `r_t`, you have used the very move you are trying to react to. Symptoms of this leak: vol-targeted PnL looks suspiciously smooth, Sharpe drops sharply when you `.shift(1)` the vol series, drawdowns that "shouldn't have happened" are absent. **Detect:** recompute every metric with the vol input shifted one bar; a large degradation means you were leaking. **Fix:** define the convention once — `pos_t = target_vol / sigma_hat_{t-1}` — and unit-test that the sizing function never references a return at or after the bar it sizes. (`templates/backtest_skeleton.py::vol_target_sizer` shows the lagged convention.)

---

## 2. Regime detection

Markets alternate between qualitatively different states (calm/trending vs turbulent/mean-reverting). Modeling the state explicitly lets you switch parameters or de-risk.

### Markov regime-switching / Hidden Markov Models (HMM)
Latent state `S_t ∈ {1..K}` follows a Markov chain (transition matrix `A`, `A[i,j] = P(S_t=j | S_{t-1}=i)`, rows sum to 1). Observation `r_t` is drawn from a state-dependent (e.g. Gaussian) emission `N(mu_{S_t}, sigma_{S_t}^2)`. Fit by EM (Baum-Welch). Outputs:
- **Filtered** prob `P(S_t = k | r_{1:t})` — uses info up to `t`. Even this still incorporates `r_t`, so lag it by one bar before trading day `t+1`. **This is the only family safe for trading.**
- **Smoothed** prob `P(S_t = k | r_{1:T})` — uses the *entire* sample including the future. Great for explaining history, **catastrophic for backtests** (it knows tomorrow). Mixing up filtered vs smoothed is the single most common HMM leak.

```python
from hmmlearn.hmm import GaussianHMM

# Fit ONLY on the training window (walk-forward; never once on full history).
X_train = train_rets.reshape(-1, 1)
model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200, random_state=0)
model.fit(X_train)

# CAUTION: hmmlearn's predict_proba returns SMOOTHED posteriors
# P(S_t | r_{1:T}) over the array you pass in — i.e. it conditions on the whole
# array, including future rows. Passing all_rets here LEAKS the future into early t.
# For a causal (filtered) posterior you must either (a) score one expanding prefix
# at a time, or (b) run the forward recursion yourself. Example (a):
def filtered_last(model, x_1d):
    probs = []
    for t in range(1, len(x_1d) + 1):
        p = model.predict_proba(x_1d[:t].reshape(-1, 1))[-1]   # P(S_t | r_{1:t})
        probs.append(p)
    return np.asarray(probs)

filt = filtered_last(model, oos_rets)            # filtered posteriors, causal
regime_for_t = pd.Series(filt[:, 1], index=oos_index).shift(1)   # lag to trade day t
```
Even the forward/filtered posterior uses the *fitted parameters*, which were learned from a window; refit only on a walk-forward expanding/rolling window (use `templates/validation.py`), never once on the full history. A 2-state Gaussian HMM on equity returns *often* separates a low-vol/positive-drift state from a high-vol/negative-drift state — useful as a crisis switch — but this is not guaranteed every refit; validate the states and beware label switching (below).

### Gaussian mixtures
A mixture (GMM) is the i.i.d. analogue of an HMM with no time dependence (equivalently, an HMM whose transition rows are all equal to the stationary marginal). Use it when you want to cluster return distributions but have no reason to model persistence. For regimes that *persist*, the HMM's transition matrix is the whole point — a GMM will flicker because it ignores that yesterday's state predicts today's.

### Threshold / TAR / SETAR models
Regime is a deterministic function of an observable threshold variable (e.g. "if 20d realized vol > X, use parameter set B"). The threshold variable must be lagged/known at decision time. Transparent, no latent-state estimation, easy to audit, no smoothed/filtered confusion. The cost: you must choose the threshold variable and level — itself an overfitting surface (don't grid-search the threshold on the same data you evaluate on).

### Choosing the number of regimes (don't overfit)
- More states almost always fit in-sample better; rarely help out-of-sample. Penalize with **BIC** (stronger penalty, preferred for parsimony) or AIC; better, select `K` by **out-of-sample** log-likelihood / economic PnL under walk-forward. (Caveat: AIC/BIC's standard asymptotics do not strictly hold for the number of mixture/HMM components — a parameter is on the boundary of the space — so treat them as heuristics and lean on OOS evidence.)
- **Label switching:** EM has no inherent state ordering; the "high-vol" state can be index 0 in one refit and index 1 in the next. Re-identify states each refit by a fixed rule (e.g. order by fitted `sigma`) before using the labels downstream.
- **Sanity floor:** a regime that captures <~5% of observations, or whose membership changes on tiny data perturbations, is likely noise. Start with `K=2`, justify any increase with OOS evidence.

---

## 3. Change-point detection

Detect *when* the data-generating process shifts, rather than assigning every point to a recurring state.

### CUSUM
Cumulative sum of deviations from a reference mean; flag when it exceeds a threshold `h`:
```
S_t^+ = max(0, S_{t-1}^+ + (x_t - mu_0 - k))     # detects upward shift
S_t^- = min(0, S_{t-1}^- + (x_t - mu_0 + k))     # detects downward shift
# alarm when S^+ > h or S^- < -h
```
`mu_0` is the in-control reference mean and must be estimated from past/in-control data only. `k` (slack/reference value) ≈ half the smallest shift worth detecting; `h` trades off detection delay vs false-alarm rate. Cheap, online, one-sided or two-sided. Commonly run on *vol* or a *spread mean* rather than raw price.

### Bayesian online change-point detection (BOCPD)
Maintains a posterior over the **run length** (time since last change-point) updated each step, with a hazard function for the prior change probability. Gives a probabilistic, online "how likely did the regime just change" signal — naturally causal (uses only past data), so safe for trading if you act on the run-length posterior at `t` to size `t+1`.

### Structural breaks: Chow test
Tests whether regression coefficients differ across a *known* break date `T_b` (F-test comparing pooled vs split-sample residual sums of squares; assumes homoskedastic, independent errors — use a robust/Wald version if those fail). For an **unknown** break, use sup-Wald / Quandt-Andrews (max Chow statistic over candidate dates within a trimmed interior region, with the appropriate non-standard critical values — you cannot use plain F critical values after searching for the worst date) or Bai-Perron for multiple breaks.

### Distinguishing a real break from noise
- A single threshold crossing is not a break. Require **persistence** (the shift holds for N bars) or a **confirmation/hysteresis** band (enter "broken" state at `h_hi`, exit only below `h_lo < h_hi`) to avoid whipsaw.
- **Multiple-testing inflation:** scanning every date for the largest break statistic guarantees a "significant" result by chance. Use the sup-statistic's correct distribution, or a permutation/bootstrap null.
- Validate economically: a break that doesn't change the OOS behavior of the strategy isn't actionable, however significant.

---

## 4. State-space models & the Kalman filter

A linear-Gaussian state-space model:
```
state:        x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)      # Q = process noise cov
observation:  y_t = H x_t   + v_t,     v_t ~ N(0, R)      # R = observation noise cov
```
The Kalman filter computes `E[x_t | y_{1:t}]` (filtered) and its covariance recursively — exactly causal, ideal for online estimation of slowly-drifting quantities. (`w_t`, `v_t` assumed zero-mean, mutually uncorrelated.)

### Local-level model (random walk + noise)
`F = H = 1`: the state is a hidden level following a random walk, observed with noise. The filtered level is an adaptive EWMA-like estimate whose smoothing is governed by the **signal-to-noise ratio** `Q/R` (large `Q/R` → tracks fast/noisy; small → smooth/sluggish). A natural denoiser for a noisy mean (e.g. a slow-moving fair value).

### Time-varying parameter / dynamic regression (drifting hedge ratio / beta)
The headline quant use. Estimate a hedge ratio or beta that drifts over time. For a pairs/cointegration hedge `y_t = beta_t * x_t + alpha_t + noise`:
```
state    x_t = [beta_t, alpha_t]^T,   F = I   (random-walk coefficients)
obs      y_t = H_t x_t + v_t,         H_t = [x_t, 1]   (regressors enter the obs matrix, time-varying)
```
Because `H_t` depends on the contemporaneous regressor `x_t`, the model is conditionally linear-Gaussian (the standard KF still applies treating `H_t` as known at `t`). The filter outputs `beta_t` using only data through `t` — the leak-free analogue of a rolling OLS hedge ratio, adapting faster with fewer parameters than a fixed window. **Trade the spread using the lagged coefficients:** form the signal from `y_t - beta_{t-1} * x_t - alpha_{t-1}`, since `beta_t`/`alpha_t` already condition on `y_t`. See `templates/pairs_trading.py` for a worked Kalman hedge-ratio pair.

### Trend/cycle extraction
Local-linear-trend (level + slope states) or unobserved-components models decompose a series into trend + cycle + noise online. Unlike an HP filter or centered moving average, the Kalman *filtered* output is causal; the **smoothed** output (`x_t | y_{1:T}`) again peeks at the future — fine for analysis, not for signals. (The HP filter in particular is two-sided and notorious for spurious end-of-sample dynamics — never use it for live signals.)

### Filter recursion (predict / update)
```python
# Predict
x_pred = F @ x                              # prior state estimate (info up to t-1)
P_pred = F @ P @ F.T + Q                    # prior covariance
# Update (incorporate y_t)
S = H @ P_pred @ H.T + R                    # innovation covariance
K = P_pred @ H.T @ np.linalg.inv(S)         # Kalman gain (use solve(), not inv(), in practice)
x = x_pred + K @ (y_t - H @ x_pred)         # posterior state  (innovation = y - H x_pred)
P = (I - K @ H) @ P_pred                    # posterior covariance (use Joseph form for stability)
```
Use `x_pred` (the *prior*, info up to `t-1`) for trading decisions about `t`; `x` (posterior) already used `y_t`. Numerical note: prefer `np.linalg.solve(S, ...)` over explicit `inv(S)`, and the Joseph-form covariance update `P = (I-KH)P_pred(I-KH).T + K R K.T` to keep `P` symmetric positive-definite.

### Tuning process/observation noise
`Q` and `R` are the dials. For the *gain and the filtered point estimate*, only their **ratio** `Q/R` matters; their absolute scale matters for the reported uncertainty (`P`, `S`) and hence for the likelihood and any confidence bands.
- Estimate by **maximum likelihood** on the innovations (prediction-error decomposition), or set by a defensible prior. Tune on a training window, validate OOS — see in-sample-tuning pitfall below.
- **Diagnostic:** if the model is correct, standardized innovations `(y_t - H x_pred)/sqrt(S_t)` are iid `N(0,1)` and serially uncorrelated. Autocorrelated or heteroskedastic innovations ⇒ mis-specified `Q/R` (or non-Gaussian/structural-break dynamics). This is your detect tool for mis-tuning.
- Too-small `Q` ⇒ filter trusts its model, ignores new data, lags reality (stale beta during a regime change). Too-large `Q` ⇒ filter chases noise, parameter jitters, trading costs explode. Pick `Q/R` by OOS PnL or innovation diagnostics, not by eyeballing the smoothness of the in-sample fit.

---

## 5. ARIMA / Box-Jenkins (brief)

ARIMA(p,d,q) models the conditional **mean**; for quant it is mostly a baseline and a stationarity-discipline tool, because returns are near-white.

- **Stationarity:** ARMA requires (weak) stationarity. **Prices are non-stationary (typically unit root)**; **returns are roughly stationary.** Model returns, not prices. (Note: `d=1` differencing of the *log* price gives log returns; ARIMA's `d` differences the modeled series, so model log returns and you are already at the stationary level — don't double-difference.) Test with ADF / KPSS (note: ADF null = unit root, i.e. rejecting supports stationarity; KPSS null = stationary, i.e. rejecting supports a unit root — they answer opposite questions, use both).
- **ACF/PACF identification:** AR(p) → PACF cuts off at lag `p`, ACF decays; MA(q) → ACF cuts off at lag `q`, PACF decays. Use these to propose `(p,q)`, then check residuals are white (Ljung-Box).
- **Differencing:** `d` = number of differences to reach stationarity. Don't over-difference (introduces spurious negative MA structure / inflates variance). For fractional persistence, ARFIMA / fractional differencing preserves more memory while reaching stationarity (useful for long-memory series like RV).
- **Why returns are near-white but vol is persistent:** the mean is barely forecastable (`ACF(r) ≈ 0` ⇒ ARIMA on returns adds little alpha), but `ACF(r^2)` is large and slowly decaying ⇒ the action is in the variance. This is precisely why ARIMA-for-mean disappoints and GARCH/EWMA-for-variance pays off. Standard practice: **ARMA-GARCH** — a thin (often zero-order) mean model with a GARCH variance.

---

## 6. Regime-aware allocation

### Vol targeting
Scale exposure inversely to **forecast** vol to stabilize realized risk:
```
pos_t = (target_vol_annual / sigma_hat_{t-1, annual}) * base_signal_t
```
- `sigma_hat_{t-1}` is the lagged vol forecast (EWMA/GARCH/RV) — **never** the contemporaneous one (Section 1 CRITICAL). See `templates/backtest_skeleton.py::vol_target_sizer`.
- Cap leverage: as `sigma_hat → 0` the inverse blows up; impose a max gross and/or floor `sigma_hat` (the template caps via `max_leverage`).
- Vol targeting tends to raise Sharpe and tame drawdowns *because* vol is forecastable and clusters; it does **not** create alpha from a zero-edge signal, and it adds turnover (rebalancing as vol moves) — charge realistic costs (`templates/costs.py`). It interacts with the strategy: for momentum it's typically synergistic; for short-vol / mean-reversion it can de-risk right before the mean-reversion payoff. It can also *increase* tail risk if you lever up into a low-vol calm that precedes a jump.

### De-risk in high-vol / crisis regimes
Use the **filtered** (lagged) HMM crisis probability or a CUSUM/threshold-on-vol flag to cut gross (or flatten) when `P(crisis | info up to t-1)` is high. Combine with vol targeting: vol targeting handles the continuous dial, the regime flag handles the discrete "get out" decision. Add hysteresis (Section 3) so you don't flip-flop at the boundary, and account for the slippage of de-risking *into* a falling, illiquid market — the cost of exiting in a crisis is exactly when costs are worst (`templates/costs.py`, and `references/microstructure.md` for liquidity-conditional impact).

---

## Pitfalls (detect / fix)

| Pitfall | Detect | Fix |
|---|---|---|
| **Look-ahead in vol/regime estimate** (estimate at `t` includes `r_t`, then sizes day `t`) | Re-run all metrics with the vol/regime input `.shift(1)`; a large Sharpe/drawdown change ⇒ you were leaking. `ewm(...).std()` is indexed with the current bar. | Fix the convention `pos_t = f(estimate_{t-1})`; unit-test the sizing fn never reads a return ≥ the bar it sizes. |
| **Smoothed instead of filtered** (HMM smoothed posteriors / Kalman smoothed state used as a signal; note `hmmlearn.predict_proba` over a full array IS smoothed) | Backtest scores `predict_proba`/`smooth` over the full sample; signal "predicts" turning points implausibly well. | Use **causal filtered** posteriors (expanding-prefix scoring or the forward recursion) / the Kalman **prior** `x_pred`, then lag one bar; reserve smoothed output for ex-post analysis only. |
| **Overfitting the number of regimes** (`K` too large) | In-sample LL keeps rising with `K` but OOS PnL/LL doesn't; tiny or unstable regimes. | Select `K` by BIC (heuristic here — boundary issue) and **OOS** criteria; start at `K=2`; require regimes economically distinct and stable to perturbation. |
| **In-sample HMM/Kalman tuning** (params fit once on full history) | Model fit on all data, then "tested" on a subset of it. | Refit on rolling/expanding walk-forward windows (`templates/validation.py`: `PurgedKFold`/`CombinatorialPurgedKFold`) with **purge + embargo**; evaluate only on untouched future data. |
| **Treating non-stationary series as stationary** (ARMA/regression on prices) | ADF fails to reject unit root; spurious-regression-style high R² with autocorrelated residuals. | Difference to returns (or test ADF+KPSS); model returns; use a cointegration framework for level relationships. |
| **Kalman noise mis-tuning** (`Q/R` wrong) | Standardized innovations not iid `N(0,1)` (autocorrelated/heteroskedastic); beta either stale or jittery. | Estimate `Q,R` by MLE on innovations; pick `Q/R` by OOS PnL + innovation diagnostics, not in-sample smoothness. |
| **GARCH persistence ≈ 1 ignored** (near-IGARCH) | `alpha+beta` ≈ 0.999; long-horizon forecasts barely mean-revert; `omega` tiny/unstable. | Use **variance targeting** (`omega=(1-alpha-beta)*uncond_var`); treat long-horizon vol as near-unit-root (shocks barely die); consider GARCH-t/GJR; don't extrapolate vol forecasts far out as if mean-reverting fast. Remember the k-step exponent is `(alpha+beta)^(k-1)`. |
| **Label switching across refits** (regime indices permute) | "High-vol" regime is index 0 sometimes, 1 other times after refit. | Re-identify states each refit by a fixed rule (e.g. order by fitted `sigma`) before using labels. |
| **Whipsaw on regime/break flags** (flickering in/out) | Flag toggles many times near the threshold; turnover and costs spike. | Add hysteresis band / minimum-persistence requirement; charge realistic costs (`templates/costs.py`). |
| **iid-Sharpe annualization on autocorrelated PnL** (vol-targeted strategies are persistent) | Realized vol scaling `*sqrt(252)` mis-states risk when PnL is autocorrelated. | Note the iid caveat; for risk use Newey-West / block-bootstrap; report the autocorrelation. |

See also: `references/pitfalls.md` (general look-ahead/overfit), `references/stats-risk.md` (Sharpe, bootstrap, Newey-West), `references/factor-research.md` (IC, cross-sectional), `references/stat-arb.md` and `templates/pairs_trading.py` (Kalman hedge ratios), `references/microstructure.md` (crisis-liquidity costs), and `templates/validation.py` (`PurgedKFold`/`CombinatorialPurgedKFold`) + `templates/backtest_skeleton.py` (`vol_target_sizer`) for runnable implementations.