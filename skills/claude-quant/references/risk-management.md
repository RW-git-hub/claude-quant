# Risk Management & Stress Testing

Risk **measurement** produces numbers (VaR, ES, exposures). Risk **management** uses those numbers to bound losses (limits, sizing, hedges, kill-switches). A VaR report nobody trades against is not a control. This reference covers both, with formulas, estimation routes, backtests, and detect/fix pitfalls.

Conventions used throughout:
- Returns are simple returns; PnL series are strategy returns (post-cost, positions lagged: `pnl_t = pos.shift(1) * ret_t`).
- **Sign convention**: VaR and ES are reported as a *return* (a loss is negative). VaR at 99% might be `-0.031` meaning "we expect to lose 3.1% or more on 1 day in 100." Many vendors report VaR as a positive magnitude; pick one convention and enforce it everywhere or you *will* flip a sign in a limit check.
- `z_alpha` denotes the standard-normal quantile at probability `alpha`. For a 99% VaR (1% left tail), `alpha = 0.01`, `z_0.01 = -2.326`.
- Historical and Gaussian VaR/CVaR already live in `templates/metrics.py` as `value_at_risk(returns, level=0.05, method="historical"|"gaussian")` and `conditional_value_at_risk(returns, level=0.05, method=...)` — note the parameter is named `level` (the lower-tail probability), not `alpha`, and a `"gaussian"` route already exists. `templates/risk.py` adds Cornish-Fisher VaR, **Filtered Historical Simulation (FHS)**, **age-weighted (BRW) historical VaR**, **EVT peaks-over-threshold (POT) tail VaR/ES**, VaR backtests (Kupiec/Christoffersen), stress, and risk-of-ruin. See also `references/stats-risk.md` and `references/live-trading.md`.

---

## 1. VaR and Expected Shortfall

### Definitions

**Value at Risk** at confidence level `1 - alpha` over horizon `h` is the quantile of the return distribution: the loss threshold that is exceeded with probability `alpha`.

```
VaR_alpha = quantile_alpha(R_h)          # left-tail quantile, reported as a (negative) return
```

So `P(R_h <= VaR_alpha) = alpha`. With the sign convention above, a 99% 1-day VaR of `-0.031` says losses worse than 3.1% happen ~1% of days.

**Expected Shortfall** (a.k.a. CVaR, ES, TVaR) is the *average* loss conditional on being in the tail beyond VaR:

```
ES_alpha = E[ R_h | R_h <= VaR_alpha ]
```

ES is always at least as bad (more negative) than VaR. It answers "when it's bad, how bad on average?" — VaR only tells you the doorway, ES tells you the room behind it.

`templates/metrics.py` already implements `value_at_risk` and `conditional_value_at_risk` with both `method="historical"` and `method="gaussian"`. `templates/risk.py` adds the Cornish-Fisher, Filtered-Historical, age-weighted, and EVT/POT routes plus backtesting and stress.

### The estimation routes

#### (a) Historical (non-parametric)

Empirical quantile of realized returns. No distributional assumption; captures the actual fat tails and skew in your sample.

```python
import numpy as np

def historical_var(returns, alpha=0.01):
    """Return-space VaR (negative = loss). alpha = tail prob (0.01 -> 99% VaR).

    NOTE: this is the risk.py-local naming; metrics.value_at_risk uses `level`
    for the same quantity. Keep one convention per call site.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    return float(np.quantile(r, alpha, method="linear"))

def historical_es(returns, alpha=0.01):
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    var = np.quantile(r, alpha, method="linear")
    tail = r[r <= var]
    # `var` is interpolated, so the mask captures roughly (but not exactly)
    # alpha*N points; for small N the ES estimate is a handful of order
    # statistics. Use method="lower" if you want the mask count to equal
    # ceil(alpha*N) exactly.
    return float(tail.mean()) if tail.size else float(var)
```

Pros: honest about the empirical tail. Cons: **bounded by the worst thing in your window** — you cannot estimate a 99.9% VaR from 250 days, because at any `alpha < 1/N` the quantile is pinned at the sample minimum (a single order statistic that cannot extrapolate). It is also **slow to react to vol regime shifts**. The two principled fixes — both now implemented in `templates/risk.py` — are: weight recent data more (**age-weighted / BRW historical simulation**, `age_weighted_var`) or filter by a vol model (**Filtered Historical Simulation, FHS**, `filtered_historical_var_es`); and for the far tail, fit an **EVT/POT** model (`evt_pot_var_es`) that extrapolates beyond the sample. See §1(e).

#### (b) Parametric / Gaussian (variance-covariance)

Assume returns are normal with mean `mu` and std `sigma`. The quantile is closed-form:

```
VaR_alpha = mu + sigma * z_alpha          (z_0.01 = -2.326, z_0.05 = -1.645)
ES_alpha  = mu - sigma * phi(z_alpha) / alpha
```

where `phi` is the standard-normal pdf. (`phi(z_0.01)/0.01 = 2.665`, so Gaussian 99% ES ≈ `mu - 2.665*sigma`.)

```python
from scipy.stats import norm

def gaussian_var(mu, sigma, alpha=0.01):
    return mu + sigma * norm.ppf(alpha)

def gaussian_es(mu, sigma, alpha=0.01):
    return mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha
```

Pros: fast, composable across a portfolio via the covariance matrix (`sigma_p = sqrt(w' Σ w)`), easy to attribute. Cons: **systematically underestimates tail risk** for real financial returns, which are fat-tailed and left-skewed. The further into the tail, the worse the error. Do not use bare Gaussian VaR for capital on anything with crash risk (equities, credit, short-vol, crypto). `templates/risk.py:gaussian_var` implements the scipy-free version (`statistics.NormalDist`) on a returns array.

Often `mu` is set to 0 for short horizons (1-day): the mean is tiny relative to sigma and noisily estimated, and zeroing it is conservative for a long book.

#### (c) Cornish-Fisher (modified VaR)

Keep the parametric speed but correct the *quantile* for skewness `S` and excess kurtosis `K` using the Cornish-Fisher expansion. Replace `z_alpha` with an adjusted quantile `z_cf`:

```
z_cf = z + (z^2 - 1)/6 * S
         + (z^3 - 3z)/24 * K
         - (2z^3 - 5z)/36 * S^2

VaR_cf = mu + sigma * z_cf
```

(`z = z_alpha`, the Gaussian quantile.) Negative skew and positive excess kurtosis push `z_cf` more negative, widening the tail toward reality.

```python
from scipy.stats import norm, skew, kurtosis

def cornish_fisher_var(returns, alpha=0.01):
    r = np.asarray(returns, dtype=float); r = r[~np.isnan(r)]
    if r.size < 4:                              # need enough points for S, K
        return float("nan")
    mu, sigma = r.mean(), r.std(ddof=1)
    # bias=False gives the sample (unbiased) estimators; the default bias=True
    # understates |S|,|K| in small samples, biasing the tail correction toward
    # Gaussian. Pick deliberately.
    S = skew(r, bias=False)
    K = kurtosis(r, fisher=True, bias=False)    # excess kurtosis
    z = norm.ppf(alpha)
    z_cf = (z
            + (z**2 - 1)/6 * S
            + (z**3 - 3*z)/24 * K
            - (2*z**3 - 5*z)/36 * S**2)
    return float(mu + sigma * z_cf)
```

(`templates/risk.py:cornish_fisher_var` ships the scipy-free version.)

Caveat: Cornish-Fisher is an *approximation* that misbehaves for extreme skew/kurtosis (the implied quantile mapping can become non-monotonic in the deep tail, so very large `|S|`,`K` can produce nonsensical `z_cf`). It's a good cheap upgrade over Gaussian for moderate non-normality; **it is not a substitute for an actual fat-tailed model when the tail is the whole point** — use a Student-t, or for the far tail the EVT/POT route in §1(e) (`templates/risk.py:evt_pot_var_es`).

#### (d) Monte Carlo

Simulate returns from a chosen model (multivariate normal/t with estimated `Σ`, a copula for tail dependence, a GARCH path, or full revaluation of nonlinear instruments like options), then take the empirical quantile/ES of the simulated PnL.

```python
def monte_carlo_var(mu_vec, cov, weights, alpha=0.01, n=200_000, df=None, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, len(mu_vec)))
    if df is not None:                       # Student-t for fatter tails (df > 2)
        # A raw multivariate-t built this way has covariance cov * df/(df-2),
        # NOT cov. Rescale the standardizing factor so the simulated covariance
        # matches `cov`; otherwise your t-VaR is inflated purely by a variance
        # bug rather than by genuine tail weight.
        g = rng.chisquare(df, size=(n, 1)) / df
        z = z / np.sqrt(g) * np.sqrt((df - 2) / df)
    sims = mu_vec + z @ L.T
    port = sims @ np.asarray(weights)
    var = np.quantile(port, alpha)
    es = port[port <= var].mean()
    return float(var), float(es)
```

Monte Carlo is the only practical route for **nonlinear** portfolios: options/convexity, path-dependent payoffs, and barrier/knock-out structures where a linear (delta) approximation badly misstates tail PnL. For an options book, simulate the *risk factors* (spot, vol surface, rates) and **full-revalue** the book per scenario — do not VaR the deltas.

#### (e) Reactive & tail-aware non-parametric VaR (implemented in `templates/risk.py`)

When "the tail is the whole point" and the bare empirical quantile is too stale (a) or the parametric corrections (b/c) are unreliable, these three routes are the practitioner fixes. All are deterministic, scipy-free, and respect the loss-is-negative sign convention.

**Filtered Historical Simulation (FHS)** — `filtered_historical_var_es(returns, level, lam=0.94, rolling=True)`. Real returns are not iid (vol clusters), so the empirical quantile of *raw* returns lags a vol spike. FHS standardizes each return by a **causal** EWMA vol forecast to get approximately-iid residuals, takes the empirical quantile/ES of those residuals (keeping the true fat-tailed/skewed shape), then **rescales by the current vol forecast** so the number reacts immediately:

```
sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_{t-1}^2   # causal: sigma_t known at t-1
z_t       = r_t / sigma_t                                  # standardized residual
z_q       = empirical quantile_level( z up to t-1 )        # non-parametric tail shape
VaR_t     = sigma_t * z_q                                  # rescale by current vol
ES_t      = sigma_t * mean( z | z <= z_q )
```

This is the industry-standard *reactive* VaR (Barone-Adesi–Giannopoulos–Vosper). `rolling=True` returns **causal** `(var_series, es_series)` (NaN during warm-up) that are directly backtestable with `count_exceptions`/`kupiec_pof` — no further lagging needed, the t-1 measurability is baked in. `rolling=False` returns a single "today" `(VaR, ES)` snapshot (uses the whole-sample residual quantile rescaled by the last vol — for a current number, not a backtest). On a fat-tailed, vol-clustered (GARCH-t) path FHS coverage tracks `alpha` closely while bare Gaussian over-breaches and is rejected by Kupiec.

**Age-weighted (BRW) historical VaR** — `age_weighted_var(returns, level, decay=0.99)`. A non-parametric reactive VaR that needs *no* vol model: keep the empirical tail but weight recent observations more (Boudoukh–Richardson–Whitelaw). With the most recent observation at age 0,

```
w_i = decay^(age_i) / sum_j decay^(age_j)
VaR = smallest return r whose cumulative ascending-sorted weight first reaches level
```

`returns` must be chronological (oldest first). As `decay -> 1` every weight is equal and it collapses to plain historical VaR; lower `decay` is more reactive but uses fewer effective observations (more tail noise). Typical `decay` ∈ [0.97, 0.995].

**EVT peaks-over-threshold (POT)** — `evt_pot_var_es(returns, level=0.001, threshold_q=0.95)`. The principled route to **deep** quantiles (99%, 99.9%) from a short sample, where a historical quantile is just the worst one or two points. Work on losses `L = -returns`; pick a high threshold `u` (the `threshold_q` loss quantile). By the **Pickands–Balkema–de Haan** theorem the exceedances `(L - u | L > u)` converge to a **Generalized Pareto** `GPD(xi, beta)`. Fit it by **probability-weighted moments (PWM, Hosking–Wallis)** — closed-form, no optimizer:

```
a0 = mean(exceedances),  a1 = (1/n) sum_j ((n-1-j)/(n-1)) * x_(j)   # ascending order stats
xi   = 2 - a0 / (a0 - 2 a1)
beta = 2 a0 a1 / (a0 - 2 a1)
```

then the closed-form tail estimators (McNeil–Frey), with `n` obs, `Nu` exceedances, deep tail prob `level`:

```
VaR_p(L) = u + (beta/xi) * ( ( (n/Nu) * level )^(-xi) - 1 )      # xi != 0
ES_p(L)  = VaR_p(L)/(1 - xi) + (beta - xi*u)/(1 - xi)           # xi < 1
```

(an exponential-tail `xi=0` limit is handled separately), flipped back to negative-return sign. The fitted **`xi` is the tail index**: `xi > 0` heavy power-law tail with exponent `alpha = 1/xi` (equities/credit/crypto live at `xi ≈ 0.1–0.4`), `xi = 0` exponential, `xi < 0` bounded. A genuine `xi >= 1` means **infinite mean** (ES does not exist). One sharp gotcha the template enforces: **the PWM estimator is mathematically capped at `xi -> 1` and cannot exceed it** (it requires a finite first GPD moment), so it will *pin* a catastrophic tail just below 1 rather than reporting `xi >= 1`. The function therefore returns `xi_near_one` (set when `xi >= 0.9`): treat the ES as untrustworthy in that regime — raise the threshold, get more data, or switch to a maximum-likelihood GPD fit (which *can* return `xi >= 1`). Requires `level < 1 - threshold_q` (you can only extrapolate *beyond* the threshold you fit above).

### Horizon scaling

Single-period VaR scales to `h` periods under iid as `VaR_h ≈ mu*h + sigma*sqrt(h)*z`. The `sqrt(h)` (square-root-of-time) rule assumes iid returns; **return autocorrelation or vol clustering breaks it** (positive autocorrelation makes true multi-day risk larger than `sqrt(h)` implies). For regulatory/serious multi-day risk, prefer overlapping h-day returns directly or simulate paths rather than scaling a 1-day number.

---

## 2. Coherent risk measures: why ES beats VaR for capital

A risk measure `ρ` is **coherent** (Artzner et al.) if it is monotone, translation-invariant, positively homogeneous, and **subadditive**: `ρ(A + B) <= ρ(A) + ρ(B)` — combining books never increases risk. Subadditivity is the mathematical statement of "diversification helps."

**VaR is not subadditive in general.** It can report that a diversified book is *riskier* than the sum of its parts — penalizing diversification, the opposite of what a risk measure should do. Classic failure: two independent bonds each with a 4% default probability. At 95% VaR each bond alone looks safe (default prob 4% < 5%, so the single-name 95% VaR sits in the no-default region). But the combined portfolio's probability of at least one default is `1 - 0.96^2 ≈ 7.8% > 5%`, so a default loss now lands inside the 5% tail and the combined 95% VaR jumps — i.e. `VaR(A+B) > VaR(A) + VaR(B)`. VaR also **ignores the shape of the tail beyond the quantile** — it cannot distinguish "lose 3% in the worst 1% of days" from "lose 30%": both can have the same VaR.

**ES is coherent** (subadditive) and **tail-sensitive** (it integrates the whole tail). Basel's market-risk framework (FRTB) moved from 99% VaR to 97.5% ES for exactly these reasons. Use ES for capital allocation, risk budgeting, and limits where you care about the severity of bad outcomes. Keep VaR around for backtesting (it has clean, well-established exception tests — ES backtesting is harder, though Acerbi-Szekely tests exist) and for communication.

Practical default: **size and budget on ES; backtest on VaR; report both.** For the far-tail ES that capital actually depends on, prefer the EVT/POT ES (§1e) over a historical tail mean that is just one or two order statistics.

---

## 3. VaR backtesting

A VaR forecast is a falsifiable prediction: "next period's loss exceeds `VaR_alpha` with probability `alpha`." Backtesting checks whether realized **exceptions** (a.k.a. breaches/violations: `r_t < VaR_t`, comparing the *forecast made at t-1* to the *return realized at t*) match that claim. Always use **out-of-sample, rolling** VaR forecasts — backtesting an in-sample fitted VaR is circular and meaningless. The rolling FHS output (`filtered_historical_var_es(..., rolling=True)`) is causal by construction, so it can be fed straight into the tests below.

Define the hit/exception indicator:
```
I_t = 1 if r_t < VaR_t   else 0          # forecast VaR_t known at t-1
```
Expected number of exceptions over `N` periods is `alpha * N`. For 99% VaR over 250 trading days you expect ~2.5 exceptions/year.

### Kupiec POF (Proportion Of Failures) — unconditional coverage

Tests whether the exception *frequency* matches `alpha`. With `x` exceptions in `N` observations and observed rate `pi_hat = x/N`:

```
LR_uc = -2 * ln( [ (1-alpha)^(N-x) * alpha^x ]
                 / [ (1-pi_hat)^(N-x) * pi_hat^x ] )   ~  chi2(1)
```

Reject (VaR mis-calibrated in frequency) if `LR_uc > chi2.ppf(0.95, 1) = 3.841`.

```python
from scipy.stats import chi2

def kupiec_pof(exceptions, alpha, conf=0.95):
    e = np.asarray(exceptions)
    x = int(np.sum(e)); N = e.size
    if N == 0:
        return {"exceptions": 0, "expected": 0.0, "LR": float("nan"),
                "crit": chi2.ppf(conf, 1), "reject": False}
    if x == 0:
        lr = -2 * N * np.log(1 - alpha)
    elif x == N:
        lr = -2 * N * np.log(alpha)
    else:
        pi = x / N
        lr = -2 * ((N - x)*np.log(1-alpha) + x*np.log(alpha)
                   - (N - x)*np.log(1-pi) - x*np.log(pi))
    return {"exceptions": x, "expected": alpha*N, "LR": lr,
            "crit": chi2.ppf(conf, 1), "reject": lr > chi2.ppf(conf, 1)}
```

(`templates/risk.py:kupiec_pof` ships the scipy-free version and takes a returns series plus an aligned `var_series`.) Kupiec catches *too many* or *too few* exceptions but is blind to **clustering**.

### Christoffersen — independence + conditional coverage

Exceptions should not cluster (a good VaR reacts to rising vol; a stale VaR breaches several days in a row). The independence test models transitions between exception states. Let `n_ij` = count of moving from state `i` to state `j` (0 = no exception, 1 = exception), `pi_01 = n_01/(n_00+n_01)`, `pi_11 = n_11/(n_10+n_11)`, and the pooled `pi = (n_01+n_11)/(n_00+n_01+n_10+n_11)` — the denominator is the number of *transitions* (`N-1`), not `N`:

```
LR_ind = -2 * ln( [ (1-pi)^(n00+n10) * pi^(n01+n11) ]
                  / [ (1-pi01)^n00 * pi01^n01 * (1-pi11)^n10 * pi11^n11 ] )  ~ chi2(1)
```

**Conditional coverage** combines both: `LR_cc = LR_uc + LR_ind ~ chi2(2)` (crit `5.991`). A VaR model must pass *both* — correctly frequent **and** independent (non-clustered).

```python
def christoffersen(exceptions, alpha, conf=0.95):
    e = np.asarray(exceptions).astype(int)
    n00=n01=n10=n11=0
    for prev, cur in zip(e[:-1], e[1:]):
        if   prev==0 and cur==0: n00+=1
        elif prev==0 and cur==1: n01+=1
        elif prev==1 and cur==0: n10+=1
        else:                    n11+=1
    pi01 = n01/(n00+n01) if (n00+n01) else 0.0
    pi11 = n11/(n10+n11) if (n10+n11) else 0.0
    pi   = (n01+n11)/(n00+n01+n10+n11) if (n00+n01+n10+n11) else 0.0
    def safe(b, k): return (b**k) if b>0 else (1.0 if k==0 else 0.0)
    num = safe(1-pi, n00+n10) * safe(pi, n01+n11)
    den = safe(1-pi01,n00)*safe(pi01,n01)*safe(1-pi11,n10)*safe(pi11,n11)
    lr_ind = -2*np.log(num/den) if den>0 and num>0 else 0.0
    uc = kupiec_pof(e, alpha, conf)
    lr_cc = uc["LR"] + lr_ind
    return {"LR_ind": lr_ind, "ind_crit": chi2.ppf(conf,1),
            "LR_cc": lr_cc, "cc_crit": chi2.ppf(conf,2),
            "reject_ind": lr_ind > chi2.ppf(conf,1),
            "reject_cc":  lr_cc > chi2.ppf(conf,2)}
```

(`templates/risk.py` ships `christoffersen` (independence) and `christoffersen_cc(exceptions, level)` (the full df=2 conditional-coverage test against a target level), both scipy-free.) Note: `LR_ind` degenerates (set to 0) when a transition cell is empty — common when there are zero or one exceptions. In that regime the independence test has essentially no power; lean on Kupiec and a longer window.

### Basel traffic-light

A simple supervisory zone test on the count of 99% exceptions in 250 days:

| Zone | Exceptions (99%, 250d) | Interpretation |
|------|------------------------|----------------|
| Green | 0–4 | Model accepted |
| Yellow | 5–9 | Suspect; capital multiplier increases |
| Red | 10+ | Model rejected; rebuild |

Use it as a fast eyeball; use Kupiec + Christoffersen for the statistical verdict. Note the multiple-testing caveat: backtesting many models/assets inflates false rejections — adjust thresholds or interpret accordingly.

---

## 4. Stress testing and scenario analysis

VaR/ES describe the *distribution you've seen*. Stress testing asks **"what if something outside the sample happens?"** It is mandatory because tail risk is dominated by regime breaks your history may not contain.

### Historical replays

Re-apply the factor moves from named crises to *today's* portfolio:

| Scenario | Characteristic shocks |
|----------|----------------------|
| 1987 Black Monday | S&P 500 −20.5% in a day; vol spike |
| 1998 LTCM / Russia | Credit spreads blow out; correlation convergence trades fail; flight to quality |
| 2008 GFC | Equities ~−50% peak-to-trough, credit spreads explode, funding/liquidity freeze, financials lead |
| 2020 COVID crash | Fastest-ever ~−34% S&P drawdown, VIX to ~80, oil futures briefly negative, then V-recovery |
| 2022 rates shock | Bonds and equities fall *together* (60/40 fails), duration/growth crushed, USD up |
| Asset-specific | Crypto: −80% drawdowns, exchange/stablecoin failures, funding-rate spikes |

Map each crisis to *risk-factor* shocks (equity index %, rate bps, credit spread bps, vol points, FX %), then revalue. For a factor portfolio, shock the **factor exposures**:

```
stressed_pnl = sum_k ( exposure_k * factor_shock_k )    # linear
```
and full-revalue for nonlinear (options) books.

```python
def factor_stress(exposures: dict, shocks: dict) -> float:
    """exposures: {factor: $ or % exposure}, shocks: {factor: shock}. Returns PnL.

    Linear (delta) approximation only — valid for small-to-moderate moves on a
    roughly linear book. For options/convexity this understates the tail; use
    full revaluation per scenario instead.
    """
    return sum(exposures.get(f, 0.0) * s for f, s in shocks.items())
```

(`templates/risk.py:stress_pnl` / `stress_grid` ship this for dict- or array-keyed exposures/shocks and a named scenario grid.)

### Hypothetical / forward-looking shocks

Construct scenarios that haven't happened but plausibly could: +200bp parallel rate jump, −15% one-day equity gap, oil +50%, your two largest positions both gap against you, a key counterparty defaults. Cover correlated *combinations* — a single-factor shock library misses the joint moves that actually do damage.

### Reverse stress testing

Invert the question: **"What scenario makes me lose X / breach my limit / go bust?"** Solve for the factor moves that hit a target loss, then judge their plausibility. This surfaces hidden concentrations and convexity that forward scenarios miss (e.g. "we only break if 30Y yields rise *and* the curve inverts *and* our short-vol leg gaps — and all three correlate in a credit event"). Cheap version: scan the historical/MC simulation for the worst paths and inspect what factor configuration produced them.

Stress results feed **management**, not just a report: they should drive hedges, exposure caps, and a contingency playbook (who de-risks what, in what order).

---

## 5. Drawdown control, risk of ruin, leverage, and Kelly

### Drawdown

Drawdown is the decline from a running peak of cumulative wealth; **max drawdown (MDD)** is the worst such decline. It is the metric capital allocators actually feel (and redeem on). `templates/metrics.py` implements `max_drawdown` on the compounded equity curve.

```
DD_t = W_t / max_{s<=t}(W_s) - 1          # <= 0
MDD  = min_t DD_t
```

Drawdown *control* means acting on it: de-gross as drawdown deepens (drawdown-based vol targeting / a CPPI-style floor), cap exposure when underwater, and pre-commit a hard stop (e.g. "halt and review at −15%"). A drawdown limit is a circuit breaker, not a forecast.

### Risk of ruin

For a strategy with per-bet edge, the probability of hitting a ruin barrier before a target grows fast with leverage and bet size. `templates/risk.py:risk_of_ruin` is a seeded Monte-Carlo estimator (fixed-fraction betting; monotone in bet size — over-betting raises ruin probability). The intuition that matters for sizing: **losses compound asymmetrically** — a −50% drawdown needs +100% to recover; −80% needs +400%. The required recovery return after a drawdown `DD` (with `DD < 0`) is `1/(1+DD) - 1` (this is "gain needed to get back to the peak," not the trading "recovery factor" = net profit / max drawdown, which is a different, unrelated metric). Bounding drawdown is bounding the *required recovery*, which is the real constraint on survival.

### Leverage and margin

Leverage `L = gross_exposure / equity` scales *both* returns and risk linearly: `VaR_levered = L * VaR_unlevered` (to first order). Two failure modes:
- **Funding/margin risk**: a drawdown triggers margin calls; forced de-leveraging at the worst prices turns a paper loss into a realized one (2008, 2020, LTCM). Model the margin spiral, not just the price move.
- **Leverage masking risk**: a smooth, high-Sharpe-looking equity curve run at high leverage hides catastrophic tail exposure — the variance is small *until* it isn't (short-vol, carry, levered relative-value).

Track gross *and* net leverage, and stress the margin requirement under the scenarios in §4 (margins themselves rise in crises).

### The Kelly connection

Kelly sizing maximizes long-run log-growth: for a single edge with excess return `mu` over the risk-free rate and variance `sigma^2`, the continuous (Merton) approximation is `f* = mu / sigma^2`; for the discrete betting form, `f* = edge/odds` (equivalently `f* = p - (1-p)/b` for win prob `p` and win/loss ratio `b`; `templates/risk.py:kelly_fraction`). Full Kelly maximizes growth but has brutal drawdowns (the optimal-growth path routinely draws down ~50%+). Practitioners run **fractional Kelly** (¼–½) to trade a little growth for far smaller drawdowns and robustness to *estimation error in `mu`* — and `mu` is always badly estimated. Over-betting past `f*` *lowers* growth and raises ruin probability, so Kelly is also an upper bound on sane leverage, not a target. This links sizing (§5) directly to the risk budget (§6) and ES (§2): size so that ES at your chosen confidence stays within the loss you can survive.

---

## 6. Position / exposure limits framework

Limits are the **management** layer — the hard bounds that convert measurements into control. A typical hierarchy:

| Limit | Definition | Typical purpose |
|-------|-----------|-----------------|
| **Gross exposure** | `sum(|position_i|) / equity` | Cap total leverage |
| **Net exposure** | `sum(position_i) / equity` | Cap directional/market beta |
| **Per-name** | `|position_i| / equity` | Idiosyncratic / blow-up cap |
| **Per-sector / group** | `sum_{i in g} |position_i| / equity` | Avoid hidden thematic bets |
| **Factor** | beta to each factor (mkt, size, value, momentum, vol, …) | Bound systematic exposure |
| **Concentration** | top-N weight, Herfindahl `sum(w_i^2)` | Diversification floor |
| **Risk (VaR/ES)** | portfolio ES_alpha | Total tail budget |
| **Liquidity** | days-to-liquidate at participation cap | Exitability (see §8) |

```python
def check_limits(positions, equity, limits, sectors=None, betas=None):
    pos = np.asarray(positions, dtype=float)
    if equity <= 0:
        raise ValueError("equity must be positive")
    gross = np.sum(np.abs(pos)) / equity
    net   = np.sum(pos) / equity
    breaches = []
    if gross > limits["gross"]: breaches.append(("gross", gross, limits["gross"]))
    if abs(net) > limits["net"]: breaches.append(("net", net, limits["net"]))
    name = (np.max(np.abs(pos)) / equity) if pos.size else 0.0
    if name > limits["per_name"]: breaches.append(("per_name", name, limits["per_name"]))
    if sectors is not None:
        sec = np.asarray(sectors)
        for s in set(sectors):
            g = np.sum(np.abs(pos[sec==s])) / equity
            if g > limits["per_sector"]:
                breaches.append((f"sector:{s}", g, limits["per_sector"]))
    if betas is not None:
        for f, b in betas.items():
            cap = limits["factor"].get(f, np.inf)
            if abs(b) > cap:
                breaches.append((f"factor:{f}", b, cap))
    return breaches  # empty == pass
```

**Pre-trade vs post-trade.** *Pre-trade* checks run before an order is sent and **block** orders that would breach a limit — the primary control (cheapest to enforce, no cleanup). *Post-trade* monitoring runs continuously on the live book and **alerts/forces remediation** when market moves (not your trades) push you over (e.g. a position grows past its cap as it rallies). You need both: pre-trade to prevent self-inflicted breaches, post-trade because the market re-weights your book without asking. `templates/pretrade_checks.py` implements the deterministic, std-library-only order gate (notional-based, evaluates the *resulting* state after the fill, collects all violations); see `references/live-trading.md` for the live monitoring / kill-switch wiring.

Limits should be *binding and automated*. A limit that requires a human to notice a number is not a control (see §8).

---

## 7. Correlations break in crises

The single most dangerous modeling assumption in risk is **static correlation**. Cross-asset correlations are regime-dependent and tend toward extremes precisely in crises:

- **Diversification fails when you need it.** In a panic, risk assets sell off together — equities, credit, EM, crypto correlations spike toward 1; the "diversified" book behaves like one leveraged long. Your normal-times `Σ` understates crisis risk badly.
- **Tail dependence.** Even assets with low *linear* correlation can have high *lower-tail* dependence — they crash together more than a Gaussian (zero tail dependence) predicts. Model it with a **Student-t or Clayton copula** (positive lower-tail dependence) rather than a Gaussian copula, which is the structure that under-priced CDO tail risk in 2008.
- **Detection**: compare a full-sample correlation matrix to one estimated on the worst 5% of market days (exceedance correlation). If crisis correlations are materially higher, your VaR/stress must use the *stressed* matrix.

### Factor risk decomposition

Decompose portfolio variance into factor and specific components to see *what you're actually exposed to*:

```
Σ = B Σ_f B' + D            # B: exposures, Σ_f: factor cov, D: diag specific var
sigma_p^2 = w' Σ w
```

**Marginal** and **component** risk contributions tell you which factor/name drives the risk budget:

```python
def risk_contributions(weights, cov):
    w = np.asarray(weights, dtype=float)
    var = w @ cov @ w
    sigma = np.sqrt(var)
    if sigma <= 0:
        return 0.0, np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    mrc = (cov @ w) / sigma          # marginal contribution to risk
    crc = w * mrc                    # component contribution (sums to sigma)
    return sigma, mrc, crc, crc / sigma   # last = % of total risk (sums to 1)
```

This is how you find that a "diversified" 50-name book has 70% of its risk in one factor. Risk budgeting (equal-risk-contribution / risk parity) sets weights so component contributions are balanced rather than dollar-weights. (Note: component contributions sum to `sigma` and the % shares sum to 1 only when each `w_i` carries its sign; with net-short legs an individual contribution can be negative.)

---

## 8. Liquidity risk, and measurement ≠ management

### Liquidity risk: can you actually exit?

Every risk number above implicitly assumes you can transact at the marked price. In a crisis you often can't. Two flavors:
- **Market liquidity**: the cost/time to exit. Estimate **days-to-liquidate** = position size / (participation cap × ADV); a position you can't exit in N days at <X% participation is illiquid regardless of its VaR. Bid-ask widens, market impact convexifies, and depth evaporates exactly when you need to sell (see `references/transaction-costs.md` for impact models).
- **Funding liquidity**: can you meet margin/redemptions without forced sales? The interaction of the two (a margin call forcing sales into an illiquid market) is the classic death spiral.

Liquidity-adjust VaR by adding an exit-cost haircut, or simply impose a liquidity limit (§6) and a haircut on illiquid marks in stress scenarios. The honest version of a risk report states the *assumed* liquidation horizon.

### A number is not a control

The core discipline of this entire reference:

> **Risk measurement** = computing VaR, ES, exposures, stress losses.
> **Risk management** = pre-committed actions that bound losses: limits that *block* trades, vol targeting that *resizes* automatically, kill-switches that *halt* trading, hedges that *cap* tail loss, and a drawdown circuit breaker.

A perfect VaR model with no limit, no stop, and no one watching the post-trade monitor controls nothing. Conversely, crude measures with hard, automated limits will keep you alive. Build the measurement to inform the management, and make the management *automatic* — humans miss numbers, especially at 3am during the event your stress test predicted. Wire enforcement into `templates/pretrade_checks.py` and the live loop in `references/live-trading.md`.

---

## Pitfalls (detect / fix)

| Pitfall | Detect | Fix |
|---------|--------|-----|
| **Gaussian VaR underestimates fat tails** | Backtest exceptions far exceed `alpha*N`; sample excess kurtosis ≫ 0, negative skew | Use historical / Cornish-Fisher / FHS / EVT for the tail; size capital on ES not Gaussian VaR (`templates/risk.py`) |
| **Stale historical VaR misses vol regime shifts** | Exceptions cluster in high-vol periods (fails Christoffersen); VaR flat while realized vol doubles | Use Filtered Historical Simulation (`filtered_historical_var_es`, vol-rescaled) or age-weighting (`age_weighted_var`); both react while keeping the empirical tail shape |
| **Can't estimate the far tail (99.9%) from a short sample** | Historical VaR at `alpha < 1/N` is pinned at the sample minimum; ES is one or two order statistics | Fit EVT/POT (`evt_pot_var_es`): a GPD tail extrapolates beyond the worst observation and gives a principled 99.9% VaR/ES |
| **Trusting an EVT ES from a near-infinite-mean fit** | EVT `xi` very close to 1; `xi_near_one` flag set; ES wildly sensitive to a handful of exceedances | PWM caps `xi` just below 1; treat ES as unreliable when `xi_near_one`. Raise `threshold_q`, get more data, or use an ML GPD fit (can return `xi >= 1`) |
| **In-sample VaR** | VaR fit and "tested" on the same window; suspiciously clean coverage | Use rolling out-of-sample forecasts (VaR_t from data ≤ t-1) before counting exceptions; the rolling FHS output is causal by construction |
| **Ignoring ES / tail shape** | Risk limits and capital keyed only to VaR; two books with equal VaR sized identically | Adopt ES (coherent, tail-sensitive) for capital/limits; report VaR and ES together; use EVT ES for the far tail |
| **Static crisis correlations** | Normal-times `Σ` used in stress; exceedance corr ≫ full-sample corr | Estimate stressed `Σ` on worst-decile days; use Student-t/Clayton copula for tail dependence |
| **No VaR backtest** | A VaR number is reported but never validated | Run Kupiec (unconditional) + Christoffersen (independence + conditional) + Basel traffic-light on a rolling basis |
| **MC Student-t variance bug** | t-VaR much larger than historical even at moderate `df`; simulated cov ≠ input `Σ` | Rescale by `sqrt((df-2)/df)` so the simulated covariance equals `Σ`; require `df > 2` |
| **Leverage masking risk** | Smooth high-Sharpe curve, small daily vol, large gross/net leverage; short-vol/carry profile | Track gross & net leverage and ES at leverage; stress margin under §4; size with fractional Kelly cap |
| **Scenario library too narrow** | Only single-factor or only recent-history shocks; reverse stress never run | Add cross-factor combinations, asset-specific crises, hypothetical shocks, and reverse stress (solve for the breaking scenario) |
| **VaR sign/convention drift** | A limit check passes a loss because the sign flipped vs the vendor convention | Fix one convention (loss = negative return) and assert it at every boundary |
| **Square-root-of-time on autocorrelated returns** | Multi-day VaR via `sqrt(h)` while returns show autocorrelation / vol clustering | Use overlapping h-day returns or simulate paths; don't scale a 1-day number |
| **Liquidity ignored** | VaR assumes instant exit at mark; large position vs ADV | Compute days-to-liquidate, impose liquidity limits, haircut illiquid marks in stress |

---

## See also
- `templates/risk.py` — Cornish-Fisher / Filtered-Historical / age-weighted / EVT-POT VaR & ES, Kupiec/Christoffersen backtests, risk-of-ruin, stress & limits
- `templates/metrics.py` — historical & Gaussian VaR/CVaR (`value_at_risk`/`conditional_value_at_risk`, param `level`), `max_drawdown`, Sharpe
- `templates/pretrade_checks.py` — deterministic pre-trade limit gating (notional-based, std-lib only)
- `references/stats-risk.md` — distributions, estimators, hypothesis tests
- `references/live-trading.md` — post-trade monitoring, kill-switches, limit enforcement
- `references/transaction-costs.md` — market impact / liquidation cost models
