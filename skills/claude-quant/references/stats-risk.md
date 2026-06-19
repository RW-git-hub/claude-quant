# Statistics, Validation, and Risk

The correctness-critical reference. Every formula here states its convention. When in doubt, re-derive from these definitions rather than copying library defaults (libraries disagree on ddof, annualization, and risk-free handling).

Conventions used throughout:
- **Simple returns** `r_t = P_t/P_{t-1} - 1` aggregate multiplicatively: `total = prod(1+r) - 1`.
- **Log returns** `g_t = ln(P_t/P_{t-1})` add: `total_log = sum(g)`. Convert back with `exp(sum) - 1`.
- `periods_per_year` (PPY): daily ≈ 252, weekly 52, monthly 12. Crypto/24-7 often 365 (state it explicitly; mixing 252 and 365 silently inflates/deflates Sharpe by ~20% — `sqrt(365/252) ≈ 1.20`).
- `std(..., ddof=1)` (sample) everywhere unless noted.
- All vectorized PnL is `position.shift(1) * return_t` — same-bar execution is a look-ahead bug.
- In all code below, coerce inputs to NumPy arrays (`np.asarray`) before positional indexing. Positional integer indexing (`r[t-1]`) on a pandas Series is **label-based** and silently wrong; convert first.

---

## 1. Multiple testing and data snooping (the core enemy)

**Why it matters.** If you try N strategies/parameters and keep the best, the best Sharpe is a maximum of N noisy draws — it is biased high even if every strategy is worthless. This is the single largest source of false discoveries in quant research. A backtest Sharpe of 2 found after a 1000-config grid search is roughly what you'd expect from pure noise.

### 1.1 Family-wise error rate (FWER)

Probability of ≥1 false positive across N tests. If each test uses level α, then under independence `FWER = 1 - (1-α)^N ≈ N·α`. With α=0.05 and N=20, FWER ≈ 0.64.

**Bonferroni** (controls FWER): use per-test threshold `α/N`. Conservative; loses power when N is large or tests correlated.

### 1.2 Benjamini-Hochberg (controls FDR)

Controls the *expected proportion of false discoveries* among rejections — less conservative, appropriate when you expect some true signals among many.

```python
import numpy as np

def benjamini_hochberg(pvals, q=0.05):
    """Returns boolean mask of rejected (significant) hypotheses at FDR=q.

    Step-up procedure: find the largest rank k (1-indexed, sorted ascending)
    with p_(k) <= q*k/n, then reject all hypotheses with p <= p_(k).
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    k = np.max(np.nonzero(passed)[0])    # largest 0-based rank passing
    cutoff = ranked[k]
    return p <= cutoff                     # p<=cutoff handles ties correctly
```

### 1.3 Probabilistic Sharpe Ratio (PSR) — Bailey & López de Prado

Probability that the *true* Sharpe exceeds a benchmark `SR*`, given the estimation error inflated by non-normal returns.

```
PSR(SR*) = Φ( (SR_hat - SR*) * sqrt(n - 1)
              / sqrt(1 - γ3·SR_hat + ((γ4 - 1)/4)·SR_hat^2) )
```

where `SR_hat` is the observed (per-period) Sharpe, `n` = number of returns, `γ3` = skewness, `γ4` = kurtosis (raw / non-excess, normal=3), `Φ` = standard normal CDF. **SR_hat and SR\* must be in the same frequency** (both per-period, or both annualized consistently).

```python
from scipy.stats import norm, skew, kurtosis

def psr(returns, sr_benchmark=0.0):
    r = np.asarray(returns, dtype=float)
    n = r.size
    sr = r.mean() / r.std(ddof=1)                 # per-period Sharpe
    g3 = skew(r, bias=False)
    g4 = kurtosis(r, fisher=False, bias=False)    # raw kurtosis (normal=3)
    denom = np.sqrt(1 - g3 * sr + ((g4 - 1) / 4) * sr**2)
    return norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom)
```

Negative skew and fat tails *lower* PSR — the standard Sharpe SE understates uncertainty for strategies that look great until they blow up (carry, short vol, selling tails). (For a normal series `γ3=0, γ4=3`, the denominator collapses to `sqrt(1 + 0.5·SR_hat^2)` and PSR(0) becomes `Φ(SR_hat·sqrt(n-1)/sqrt(1+0.5·SR_hat^2))` — almost exactly the Lo t-stat of §3, off only by `sqrt(n-1)` vs `sqrt(n)`. The companion `templates/metrics.py` uses the biased/MLE moments — `np.mean(z^k)` with population `std(ddof=0)` — which is López de Prado's own PSR convention; the difference from scipy's `bias=False` moments is negligible at backtest sample sizes.)

### 1.4 Deflated Sharpe Ratio (DSR)

PSR where the benchmark `SR*` is set to the *expected maximum Sharpe under the null* given you ran N trials. This directly penalizes multiple testing.

Expected maximum of N iid standard-normal Sharpe estimates:
```
SR*  =  sqrt(Var(SR_trials)) * [ (1-γ)·Φ^{-1}(1 - 1/N)  +  γ·Φ^{-1}(1 - 1/(N·e)) ]
```
where `γ ≈ 0.5772` (Euler-Mascheroni), `Var(SR_trials)` is the variance of the (per-period) Sharpe ratios across your N trials, and `Φ^{-1}` is the normal quantile. Then `DSR = PSR(SR*)`. Note `SR*` and the `SR_hat` inside `psr` must be in the **same frequency** (both per-period here). (This Gumbel-style order-statistic approximation is accurate to a few percent versus simulation and tightens as N grows — verified ~2.5% relative error at N=10, ~0.6% at N=200.)

```python
def expected_max_sharpe(var_sr_trials, n_trials):
    e = np.e
    g = 0.5772156649
    return np.sqrt(var_sr_trials) * (
        (1 - g) * norm.ppf(1 - 1.0 / n_trials)
        + g * norm.ppf(1 - 1.0 / (n_trials * e)))

def deflated_sharpe(best_returns, var_sr_trials, n_trials):
    sr_star = expected_max_sharpe(var_sr_trials, n_trials)
    return psr(best_returns, sr_benchmark=sr_star)
```

**Detect/fix.** If you grid-searched, you MUST report DSR, not raw Sharpe. A DSR below ~0.95 means the "winner" is not distinguishable from the luckiest of N noise draws. Track `n_trials` honestly — it includes abandoned variants, not just the final grid.

#### 1.4.1 Effective number of trials — estimate N_eff BEFORE deflating

The formula above assumes the N trials are **independent**. A grid search almost never is: 1000 neighbouring parameter combinations produce 1000 *highly correlated* Sharpe estimates. The expected maximum of N correlated draws is much smaller than the expected maximum of N independent draws, so plugging the raw count `N = 1000` into `SR*` **over-deflates** — it treats a tight cluster of near-duplicate variants as 1000 independent bets and can bury a genuine edge under the noise floor. (This is why passing the raw count is "conservative but reasonable": it errs toward rejecting, never toward false discovery.) Current DSR practice (López de Prado 2014/2018, reaffirmed in production DSR pipelines) is to first estimate the **effective number of independent trials** from the *correlation structure of the trials*, then deflate by `N_eff`, not by the config count.

**The rule: estimate `N_eff` from the trial-return correlation matrix and pass that to the DSR — never count configurations.** Two near-identical grid points are one effective trial; ten genuinely different signals on different data are ten.

**Spectral / participation-ratio estimator (dependency-free, recommended).** Build the `N×N` correlation matrix `C` of the per-trial return streams (one column per config, aligned in time), take its eigenvalues `λ_1…λ_N`, and compute the participation ratio of the spectrum:
```
N_eff = (Σ_i λ_i)^2 / Σ_i λ_i^2
```
Because `C` is a correlation matrix, `Σ_i λ_i = trace(C) = N`, so equivalently
```
N_eff = N^2 / Σ_i λ_i^2 = N^2 / ‖C‖_F^2
```
(`‖·‖_F` = Frobenius norm = `sqrt(Σ_ij C_ij^2)`). This is the spectral count of effective independent dimensions:
- All trials orthogonal → `C = I`, every `λ_i = 1` → `N_eff = N` (no deflation relief).
- All trials identical → one eigenvalue `≈ N`, the rest `≈ 0` → `N_eff → 1` (one effective bet).
- `N_eff` is **monotone decreasing** in pairwise correlation and always lies in `[1, N]`.
- Equicorrelated sanity check (all pairwise corr = ρ): one eigenvalue `1+(N-1)ρ`, the rest `1-ρ`, so `N_eff = N^2 / [(1+(N-1)ρ)^2 + (N-1)(1-ρ)^2]` — e.g. N=15, ρ=0.5 gives `N_eff ≈ 3.3`. Matches the empirical participation ratio.

```python
def effective_number_of_trials(trial_returns_matrix):
    """N_eff from the (T x N) matrix of per-trial return streams (one column per
    config). Participation ratio of the trial-correlation eigenvalue spectrum."""
    m = np.asarray(trial_returns_matrix, dtype=float)
    C = np.corrcoef(m, rowvar=False)               # NxN trial correlation matrix
    eig = np.clip(np.linalg.eigvalsh(C), 0.0, None) # PSD; kill round-off negatives
    n_eff = eig.sum() ** 2 / (eig ** 2).sum()       # == N^2 / ||C||_F^2 (trace=N)
    return float(min(max(n_eff, 1.0), C.shape[0]))  # mathematically in [1, N]
```

**Conservative lower bound (correlation-threshold clusters).** As a sanity floor, count the number of correlation clusters at a high threshold (e.g. merge any two trials with `|corr| ≥ 0.95` into one cluster via connected components, then count clusters). This collapses blocks of near-duplicates to one and gives an *integer* lower bound on `N_eff`. Use it to bracket the spectral estimate, not as the DSR input — single-linkage at a hard threshold can chain unrelated trials and is sensitive to the cutoff.

**Getting the OTHER DSR input right.** `Var(SR_trials)` is the **cross-trial** variance of the **per-period** Sharpe estimates — the spread of each config's own Sharpe across the N configs, not the sampling SE of a single Sharpe (a common confusion). Compute each column's per-period `mean/std(ddof=1)`, then take the variance (or std) across columns. Keep it per-period to match the per-period `SR_hat` inside `psr`/`expected_max_sharpe`; multiply the std by `sqrt(PPY)` only if you deliberately work in annualized units everywhere.

> Self-tested, dependency-free implementations live in `templates/metrics.py`:
> `effective_number_of_trials(trial_returns_matrix, method='cluster')` (spectral participation ratio; `method='threshold'` for the cluster floor),
> `effective_number_of_trials_threshold(..., corr_threshold=0.95)`,
> `trial_sharpe_std_from_matrix(...)` (the cross-trial per-period Sharpe std), and
> `deflated_sharpe_ratio(returns, n_trials, trial_sharpe_std, ...)` — which now accepts either a raw integer count **or** a non-integer effective count (the order-statistic expected-max is smooth in N). The recommended call is:
> ```python
> n_eff = effective_number_of_trials(trial_returns_matrix)        # estimate FIRST
> tsr   = trial_sharpe_std_from_matrix(trial_returns_matrix)      # per-period std
> dsr   = deflated_sharpe_ratio(best_returns, n_eff, tsr)         # then deflate
> ```

### 1.5 Probability of Backtest Overfitting (PBO) via CSCV

Combinatorially Symmetric Cross-Validation: split the performance matrix (T observations × N configs) into S contiguous blocks, form all `C(S, S/2)` train/test partitions, pick the config that's best in-sample (IS), record its out-of-sample (OOS) rank. **PBO = fraction of partitions where the IS-best config's OOS performance ranks below the median.** High PBO (>0.5) means your selection process is overfitting.

```python
from itertools import combinations

def pbo_cscv(perf, n_blocks=16):
    """perf: (T, N) array of per-period returns/perf for N configs.

    n_blocks must be even. Returns PBO in [0, 1].
    """
    perf = np.asarray(perf, dtype=float)
    T, N = perf.shape
    assert n_blocks % 2 == 0, "n_blocks must be even for symmetric splits"
    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2
    logits = []
    for train_idx in combinations(range(n_blocks), half):
        test_idx = [b for b in range(n_blocks) if b not in train_idx]
        tr = np.concatenate([blocks[b] for b in train_idx])
        te = np.concatenate([blocks[b] for b in test_idx])
        is_sr  = perf[tr].mean(0) / (perf[tr].std(0, ddof=1) + 1e-12)
        oos_sr = perf[te].mean(0) / (perf[te].std(0, ddof=1) + 1e-12)
        n_star = np.argmax(is_sr)                         # IS winner
        # relative rank of the IS winner among OOS performances:
        # fraction of configs whose OOS Sharpe is below the winner's.
        rank = (oos_sr < oos_sr[n_star]).sum() / N
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return (logits <= 0).mean()                          # PBO = P(rank below median)
```

Note: the `(T, N)` performance matrix you build for PBO/CSCV is exactly the `trial_returns_matrix` you feed `effective_number_of_trials` (§1.4.1) — compute both from the same per-config return panel. The PBO estimate is only unbiased asymptotically (in `T`, in the number of configs, and in `C(S, S/2)` partitions); with a handful of configs expect noisy PBO even on pure noise — read it as a coarse over/under-0.5 flag, not a precise probability.

### 1.6 Minimum backtest length (MinBTL)

Bailey et al.: the backtest length needed so that an IS *annualized* Sharpe of `SR_annual` is not expected purely from selecting the best of N trials:
```
MinBTL (years) ≈ ( (1 - γ)·Φ^{-1}(1 - 1/N) + γ·Φ^{-1}(1 - 1/(N·e)) )^2 / SR_annual^2
```
Intuition: the more configs you try (N), the longer the history you need to trust a given Sharpe. With N=100 and target SR=1, you need ~6 years just to clear the noise floor. (Use the **effective** N from §1.4.1 here too — correlated configs are fewer effective trials and thus need less history than the raw count implies.)

---

## 2. Out-of-sample validation

### 2.1 Why plain KFold / shuffle leaks

Time series have (a) **serial autocorrelation** and (b) **labels built from forward windows**. Shuffling places adjacent (near-identical) bars in both train and test; forward-looking labels mean a training sample's label window can overlap the test period. Both let the model "see the future." Result: glowing CV scores that vanish live.

**Detect.** Any use of `sklearn.model_selection.KFold(shuffle=True)`, `train_test_split` without `shuffle=False`, or cross-validation that ignores label horizons on time-series data.

### 2.2 Splits and walk-forward

- **Train / validation / test**: tune on validation, touch test **once**. Re-tuning on test = leakage.
- **Walk-forward, rolling**: fixed-length train window slides forward (adapts to regimes, forgets old data).
- **Walk-forward, anchored**: train start fixed, window grows (more data, slower to adapt).

```python
def walk_forward_splits(n, train, test, anchored=False):
    start = 0
    while start + train + test <= n:
        tr0 = 0 if anchored else start
        tr = range(tr0, start + train)
        te = range(start + train, start + train + test)
        yield list(tr), list(te)
        start += test    # step by test size = non-overlapping OOS
```

Note: this baseline walk-forward does not purge/embargo the boundary between train and test. If labels look forward `h` bars, drop the last `h` training samples (and embargo the first `h` test samples) at each split — see §2.3.

### 2.3 Purged + embargoed K-Fold (López de Prado)

**Purge** training samples whose label window overlaps the test fold; add an **embargo** gap after the test fold to kill leakage from serial correlation.

```python
def purged_kfold(n, n_splits=5, embargo_pct=0.01, label_horizon=1):
    """Yields (train_idx, test_idx). label_horizon = bars a label looks forward.

    Purge: a train sample at i carries a label spanning [i, i+label_horizon];
    it leaks into a test fold starting at t0 if i + label_horizon >= t0, so we
    keep only i < t0 - label_horizon on the left.
    Embargo: drop the embargo bars immediately AFTER the test fold from training,
    because their features overlap the test labels' lookforward window.
    """
    idx = np.arange(n)
    folds = np.array_split(idx, n_splits)
    embargo = int(n * embargo_pct)
    for f in folds:
        t0, t1 = f[0], f[-1]
        test = idx[t0:t1 + 1]
        # purge left: train samples whose [i, i+horizon] overlaps the test span
        left  = idx[: max(t0 - label_horizon, 0)]
        # embargo right: skip 'embargo' bars after the test fold
        right = idx[min(t1 + 1 + embargo, n):]
        train = np.concatenate([left, right])
        yield train, test
```

### 2.4 Combinatorial Purged CV (CPCV)

Generalizes purged K-fold: choose k test groups out of N (instead of 1), giving `C(N, k)` train/test combinations and **multiple OOS paths** instead of one. This yields a distribution of OOS performance (feeds PBO directly) rather than a single point estimate. Use when you need confidence bands on OOS metrics. Apply purge+embargo around *every* test group in each combination (purge on both sides of each test group, embargo after each).

> Drop-in, self-tested, sklearn-style splitters live in `templates/validation.py`: `PurgedKFold` and `CombinatorialPurgedKFold` (both expose `.split` / `.get_n_splits`, and CPCV exposes `.n_paths()`). They purge `label_horizon` on **both** sides of each test block; the teaching snippet in §2.3 purges the left and embargoes the right only — correct for one contiguous fold, but import the template for the rigorous version.

---

## 3. Performance metrics — exact formulas

Let `r` be a series of per-period **simple** returns, `n = len(r)`, PPY = periods per year, `rf` = per-period risk-free rate.

```python
def geometric_annual_return(r, ppy=252):
    r = np.asarray(r, dtype=float)
    growth = np.prod(1 + r)
    if growth <= 0:           # cumulative wealth wiped out (e.g. leverage) -> total loss
        return -1.0
    return growth ** (ppy / r.size) - 1

def annual_vol(r, ppy=252):
    return np.std(np.asarray(r, dtype=float), ddof=1) * np.sqrt(ppy)

def sharpe(r, rf=0.0, ppy=252):
    excess = np.asarray(r, dtype=float) - rf   # per-period excess
    return excess.mean() / excess.std(ddof=1) * np.sqrt(ppy)

def sortino(r, mar=0.0, ppy=252):
    excess = np.asarray(r, dtype=float) - mar
    # downside deviation: RMS of below-target deviations, denominator = n (full
    # sample, the standard Sortino/LdP convention). Population RMS (ddof=0).
    dd = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    return excess.mean() / dd * np.sqrt(ppy)

def max_drawdown(r):
    eq = np.cumprod(1 + np.asarray(r, dtype=float))
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return dd.min()                        # negative number

def calmar(r, ppy=252):
    mdd = abs(max_drawdown(r))
    return geometric_annual_return(r, ppy) / mdd if mdd > 0 else np.inf

def hit_rate(r):                           # fraction of positive periods
    r = np.asarray(r, dtype=float)
    return (r > 0).mean()

def profit_factor(r):                      # gross profit / gross loss
    r = np.asarray(r, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return gains / losses if losses > 0 else np.inf
```

**Sharpe caveats.** `sqrt(PPY)` annualization assumes iid returns; correct for autocorrelation (§4.1). Use per-period excess returns, not `(annual_ret - annual_rf)/annual_vol` (subtly different and wrong with compounding). Crypto: pick PPY=365 and a 0 or stablecoin-yield rf.

**Sortino note.** Two conventions exist for the downside-deviation denominator: divide by `n` (all observations — what the code above uses, standard) vs by count of below-target only. State which; dividing by below-target count inflates Sortino. Never confuse them.

### Information Ratio (IR)

Active return over a benchmark divided by tracking error:
```
IR = mean(r_p - r_b) / std(r_p - r_b, ddof=1) * sqrt(PPY)
```
IR is Sharpe with the benchmark as the "risk-free" leg. Use for relative/long-short or benchmarked mandates.

### Turnover

```
turnover_t = sum_i |w_{i,t} - w_{i,t-1}^{drift}|   # one-sided, per rebalance
```
where `w_{i,t-1}^{drift}` is the prior weight after price drift (NOT the prior target weight — using the target overstates turnover). This one-sided definition counts the total absolute weight change; some shops report half of this (the round-trip-normalized version). State which. Annualize by summing per-rebalance turnover over a year. Drives transaction-cost drag: `cost ≈ turnover * cost_per_unit`. (`templates/metrics.py:turnover` reports the **half** version — `0.5·Σ|Δw|`, averaged over rebalances — so a full one-name-to-another swap in a fully-invested book reads as 100%; pick a convention and apply it consistently end-to-end.)

### t-stat of the Sharpe ratio

For a per-period Sharpe `SR` over `n` periods, under iid normality (Lo 2002):
```
SE(SR) = sqrt((1 + 0.5·SR^2) / n)              # SR per-period
t      = SR / SE(SR) = SR·sqrt(n) / sqrt(1 + 0.5·SR^2)
```
The common shortcut `t = SR·sqrt(n)` is only the small-SR approximation — it drops
the `1 + 0.5·SR^2` term, so it overstates significance for large Sharpes. Use the
full form. The t-stat is scale-invariant, so annualizing cancels: an **annualized**
Sharpe of 1.0 over 1 year of daily data (n=252) gives `t ≈ 1.0`; you need several
years for t > 2. See `templates/metrics.py:sharpe_tstat` / `sharpe_se`. Report the
t-stat or PSR, never a bare Sharpe.

---

## 4. Statistical significance of a strategy

### 4.1 Sharpe SE and the Lo (2002) autocorrelation adjustment

iid SE understates uncertainty when returns are autocorrelated (trend strategies: positive autocorr → SE too small → Sharpe too significant; mean-reversion: negative autocorr → opposite). Lo's correction scales the *annualization factor*:

```python
def lo_annualization_factor(r, q, min_pairs=20):
    """Scale a per-period Sharpe to a q-period (e.g. annual) one, corrected for
    autocorrelation. Replaces naive sqrt(q).
        eta(q) = q / sqrt( q + 2 * sum_{k=1}^{q-1} (q-k) * rho_k ).
    Equals sqrt(q) when all rho_k = 0.

    Guards the q ~ n case: a naive version computes lag-(q-1) autocorrelation on
    almost no points and returns NaN. Here we only sum lags with >= min_pairs
    overlapping observations (high lags carry tiny (q-k) weight) and fall back to
    sqrt(q) if no lag is estimable or the adjusted variance is non-positive.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    max_lag = min(q - 1, n - min_pairs)
    if max_lag < 1:
        return np.sqrt(q)
    mean = r.mean()
    var = np.mean((r - mean) ** 2)
    if var <= 0:
        return np.sqrt(q)
    s = sum((q - k) * np.mean((r[:-k] - mean) * (r[k:] - mean)) / var
            for k in range(1, max_lag + 1))
    denom = q + 2 * s
    return q / np.sqrt(denom) if denom > 0 else np.sqrt(q)
```

Use `lo_annualization_factor(r, PPY)` in place of `sqrt(PPY)`. Positive autocorrelation makes this *smaller* than √PPY (deflates Sharpe). Caveat: estimating PPY-1 autocorrelations (251 lags for daily-annual) is noisy, and a *naive* implementation returns NaN when `q` approaches `n`; truncate at a smaller lag (Newey-West / Bartlett weights) in practice. The guarded, self-tested version lives in `templates/metrics.py:lo_annualization_factor`.

### 4.2 Non-normality

Skew and kurtosis bias significance: the Sharpe SE has higher-moment terms (used in PSR §1.3). Strategies with negative skew + high kurtosis (option selling, carry) have far wider true confidence intervals than √PPY implies. Always inspect `skew`, `kurtosis`, and the worst-day/worst-week tails alongside Sharpe.

### 4.3 Bootstrap (with dependence)

iid bootstrap destroys serial correlation → understates risk for dependent series. Use **block** or **stationary** bootstrap to preserve dependence when building CIs for Sharpe, drawdown, etc.

```python
def stationary_bootstrap_indices(n, mean_block):
    """Politis-Romano: geometric block lengths, p = 1/mean_block.
    Uses circular (wrap-around) indexing, as in the original method."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = np.random.randint(n)
    for t in range(1, n):
        if np.random.rand() < p:
            idx[t] = np.random.randint(n)            # start new block
        else:
            idx[t] = (idx[t - 1] + 1) % n            # continue block (circular)
    return idx

def bootstrap_sharpe_ci(r, mean_block=20, B=2000, ppy=252, alpha=0.05):
    r = np.asarray(r, dtype=float)
    stats = np.empty(B)
    for b in range(B):
        s = r[stationary_bootstrap_indices(r.size, mean_block)]
        stats[b] = s.mean() / s.std(ddof=1) * np.sqrt(ppy)
    return np.quantile(stats, [alpha / 2, 1 - alpha / 2])
```

### 4.4 Comparing many strategies: White's Reality Check & Hansen's SPA

When selecting the best of many strategies vs a benchmark, the max performance statistic is biased. **White's Reality Check (RC)** tests H0: best strategy does not beat the benchmark, using a (block) bootstrap of the performance differential `d_{k,t} = perf_k - perf_bench`. **Hansen's SPA** improves power by studentizing and removing poor (irrelevant) strategies from the null. Use SPA over RC; both require a dependence-preserving bootstrap.

```python
def reality_check_pvalue(D, mean_block=20, B=2000):
    """D: (T, K) matrix of per-period perf differentials vs benchmark.
    H0: max_k E[d_k] <= 0 (no strategy beats the benchmark)."""
    D = np.asarray(D, dtype=float)
    T, K = D.shape
    f_bar = D.mean(0)
    V = np.sqrt(T) * f_bar.max()                  # observed max statistic
    boot_max = np.empty(B)
    for b in range(B):
        s = D[stationary_bootstrap_indices(T, mean_block)]
        # recenter each column on its own full-sample mean to impose H0
        boot_max[b] = (np.sqrt(T) * (s.mean(0) - f_bar)).max()
    return (boot_max >= V).mean()
```

RC/SPA and the DSR attack the same disease (selecting the best of many) from two angles: RC/SPA bootstrap the performance gap to a benchmark; the DSR (§1.4) analytically deflates by the expected max of `N_eff` trials. They are complementary — report both when you have selected a winner from a large search.

---

## 5. Time-series properties

### 5.1 Stationarity: ADF and KPSS

- **ADF** (`statsmodels.tsa.stattools.adfuller`): H0 = unit root (non-stationary). Reject (p<0.05) → stationary.
- **KPSS** (`kpss`): H0 = stationary. Reject → non-stationary. **Opposite null** — run both; agreement is strong evidence, disagreement suggests near-unit-root / fractional integration.

Prices are usually I(1) (non-stationary); returns are usually I(0). Model returns or spreads, not raw prices.

### 5.2 Autocorrelation

`statsmodels.acf / pacf`; Ljung-Box (`acorr_ljungbox`) for joint significance. Matters for: Sharpe SE (§4.1), choosing bootstrap block size, and detecting microstructure (bid-ask bounce gives negative lag-1 autocorr in high-freq returns).

### 5.3 Cointegration (pairs / stat-arb)

Two I(1) series are cointegrated if a linear combination is I(0) (mean-reverting spread). **Correlation ≠ cointegration** — never build a pairs trade on correlation alone.

- **Engle-Granger**: regress `y` on `x`, ADF-test the residual (the spread). Two-step; direction-dependent (regressing x on y can differ — test both orderings or use the more stationary residual).
- **Johansen** (`statsmodels.tsa.vector_ar.vecm.coint_johansen`): handles >2 assets, estimates the number of cointegrating vectors via trace/eigen statistics. Preferred for baskets.

```python
from statsmodels.tsa.stattools import coint
# Engle-Granger: tests cointegration with the FIRST arg as dependent variable.
# Direction-sensitive; swap y and x and compare.
score, pvalue, _ = coint(y, x)     # p<0.05 => reject no-cointegration
```

**Pitfall.** Cointegration relationships break (regime shifts, structural breaks). Re-estimate the hedge ratio on a rolling basis and monitor spread stationarity live; a "broken" pair that stops mean-reverting is how stat-arb books bleed.

### 5.4 Regime awareness

Volatility clusters; correlations spike toward 1 in crises (diversification fails when you need it). Consider explicit regime detection (vol thresholds, Markov-switching, HMM) and stress-test metrics within each regime, not just pooled.

---

## 6. Risk

### 6.1 Volatility estimation

- **Rolling**: `r.rolling(w).std(ddof=1)`. Simple, but equal-weighted and laggy.
- **EWMA** (RiskMetrics): `σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}`, λ≈0.94 daily. Reacts faster.
- **GARCH(1,1)**: `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}` (`arch` package). Captures vol clustering and mean reversion; use for forward vol forecasts and option/risk work.

```python
# r may be a pandas Series; convert to an ndarray so positional indexing is safe.
r = np.asarray(r, dtype=float)
sigma2 = np.empty(len(r)); sigma2[0] = r.var(ddof=1)
lam = 0.94
for t in range(1, len(r)):
    sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2  # uses r[t-1]: causal
ewma_vol = np.sqrt(sigma2)
```
(`σ²_t` depends only on `r²_{t-1}` and `σ²_{t-1}` — known at the start of bar `t`, so it is a one-sided/causal estimate with no look-ahead; this is the variance you may condition a position on at `t`.)

### 6.2 VaR and CVaR

VaR_α = the loss not exceeded with probability `1-α` (report sign convention explicitly; the functions below return VaR/CVaR as **positive loss numbers**). CVaR (Expected Shortfall) = mean loss *given* you breached VaR — coherent and tail-aware, unlike VaR. CVaR ≥ VaR by construction.

```python
def var_historical(r, alpha=0.05):
    return -np.quantile(r, alpha)                 # positive loss

def cvar_historical(r, alpha=0.05):
    q = np.quantile(r, alpha)
    return -r[r <= q].mean()                       # positive loss, >= VaR

def var_parametric(r, alpha=0.05):                # Gaussian; understates fat tails
    from scipy.stats import norm
    r = np.asarray(r, dtype=float)
    return -(r.mean() + norm.ppf(alpha) * r.std(ddof=1))
```

- **Historical**: no distributional assumption, but bounded by sample history (no scenario worse than observed).
- **Parametric (Gaussian)**: cheap, but understates tail risk; use Cornish-Fisher or Student-t for fat tails.
- **Monte Carlo**: flexible (path-dependent, options), only as good as the simulated distribution/correlations.

**Limitations.** VaR is not subadditive (can penalize diversification), says nothing about loss *beyond* the quantile, and is notoriously underestimated in crises (correlations and vol jump). Prefer CVaR; always pair with drawdown and stress tests.

### 6.3 Drawdown and tail risk

Report maxDD (§3), drawdown duration (time underwater), Calmar, and Ulcer Index (RMS of drawdowns). Tail metrics: skew, excess kurtosis, worst-day/week, and CVaR. A high Sharpe with deep, long drawdowns and negative skew is a different (worse) bet than the Sharpe alone implies.

---

## 7. Portfolio construction

### 7.1 Mean-variance and estimation-error instability

MVO maximizes `wᵀμ - (λ/2)·wᵀΣw`. **Why it's fragile.** Optimal weights are extremely sensitive to `μ` estimates (error-maximizing: it loads on assets with the largest estimation error). Sample covariance is ill-conditioned when assets ≈ observations. Fixes: shrink inputs, add constraints, or skip μ (min-variance / risk-based).

### 7.2 Covariance shrinkage (Ledoit-Wolf)

Shrinks the noisy sample covariance toward a structured target (e.g., constant-correlation or identity), with an analytically optimal shrinkage intensity that minimizes expected Frobenius distance to the true covariance. Dramatically stabilizes MVO out of sample.

```python
from sklearn.covariance import LedoitWolf
# returns_matrix: (T, N) — T observations (rows) by N assets (columns)
Sigma = LedoitWolf().fit(returns_matrix).covariance_
```

Note: sklearn's `LedoitWolf` shrinks toward a **scaled identity** (Ledoit & Wolf, *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*). For asset returns the **constant-correlation** target (Ledoit & Wolf, *Honey, I Shrunk the Sample Covariance Matrix* — a different paper, not what sklearn implements) is usually preferable: it preserves the average pairwise correlation rather than driving off-diagonals toward zero. A self-tested implementation is in `templates/validation.py:constant_correlation_shrinkage`.

### 7.3 Risk parity and Hierarchical Risk Parity (HRP)

- **Risk parity**: weights so each asset contributes equal risk; marginal risk contribution `RC_i = w_i·(Σw)_i`, solve for all `RC_i` equal. No expected returns needed.
- **HRP** (López de Prado): hierarchical clustering of the correlation matrix + recursive bisection. Avoids matrix inversion entirely — robust to ill-conditioned Σ, better OOS than MVO when N is large.

```python
# HRP sketch: cluster, quasi-diagonalize, recursive bisection.
# Distance: d_ij = sqrt(0.5*(1 - corr_ij)); linkage('single'); then
# allocate inverse-variance within clusters, split by aggregated cluster variance.
```

### 7.4 Constraints

Long-only (`w≥0`), gross/net exposure (`sum|w|≤L`, `sum w = target`), position caps, sector/factor neutrality, turnover limits. Constraints often improve OOS performance more than better point estimates — they cap the damage from estimation error.

### 7.5 Factor risk models

`Σ = B·F·Bᵀ + D` where `B` = factor exposures (loadings), `F` = factor covariance, `D` = diagonal idiosyncratic variances. Reduces parameters from `O(N²)` to `O(N·k)`, far more stable than sample Σ for large N. Basis of commercial risk models (Barra-style).

### 7.6 Beta / sector neutralization

Neutralize a factor by residualizing: regress your signal/returns on the neutralizing factor(s) and keep the residual.
```
α_neutral = α - β̂·factor,   β̂ from OLS of α on factor
```
For market-beta neutrality, size the hedge by rolling beta. For sector neutrality, demean the signal within each sector (cross-sectionally) before ranking.

---

## 8. Factor analysis

### 8.1 Information Coefficient (IC)

Cross-sectional correlation between a factor at time `t` and **forward** returns over `t..t+h`:
```
IC_t = corr( factor_{i,t}, return_{i, t→t+h} )   across assets i
```
- **Spearman (rank)** IC: robust to outliers and monotone-but-nonlinear relationships — the default for factor work.
- **Pearson** IC: assumes linearity; sensitive to outliers.

Report mean IC, IC std, and **IC IR = mean(IC)/std(IC)** (the factor's Sharpe-analog). Mean IC of 0.03-0.05 is already a solid equity factor.

```python
import pandas as pd
from scipy.stats import spearmanr

def ic_series(factor_df, fwd_ret_df):
    """factor_df, fwd_ret_df: (dates x assets), aligned.
    fwd_ret_df.loc[t] must already hold the FORWARD return over t->t+h,
    computed from prices strictly after t (no same-bar contamination)."""
    ics = {}
    for date in factor_df.index:
        f = factor_df.loc[date]
        r = fwd_ret_df.loc[date]
        mask = f.notna() & r.notna()
        if mask.sum() > 2:
            ics[date] = spearmanr(f[mask], r[mask]).correlation
    return pd.Series(ics)
```

**Detect/fix.** The classic look-ahead bug: correlating factor_t with return_t (same period) instead of forward returns. The forward-return panel must be shifted so that `fwd_ret.loc[t]` covers `t→t+h` and uses only prices after t. Concretely, `fwd_ret = close.pct_change(h).shift(-h)` makes `fwd_ret.loc[t] = close_{t+h}/close_t - 1` (uses only prices strictly after t; the last `h` rows are NaN and dropped) — never reach back to a price at or before t.

### 8.2 IC decay

Plot mean IC vs horizon `h` (1d, 5d, 20d, ...). Fast decay → high-turnover, cost-sensitive signal; slow decay → can hold longer, lower turnover. Decay shape sets the rebalance frequency.

### 8.3 Quantile / decile spread returns

Sort assets into deciles by factor each period; track each decile's forward return. A monotone Q1→Q10 pattern validates the factor. **Long-short spread = top decile − bottom decile**; its Sharpe is the headline factor performance. Monotonicity matters more than just top-vs-bottom (non-monotone = fragile).

### 8.4 Fama-MacBeth regressions

Two-pass estimate of factor risk premia:
1. **Each period** `t`, cross-sectional regression of returns on exposures → factor returns `λ_t`.
2. Average over time: premium `λ̄ = mean(λ_t)`; `t-stat = λ̄ / (std(λ_t, ddof=1)/√T)` (the std-over-time gives the SE, handling cross-sectional correlation).

```python
import statsmodels.api as sm
import numpy as np

def fama_macbeth(returns_panel, exposures_panel):
    """returns_panel: list of (n_assets,) arrays per date;
    exposures_panel: list of (n_assets, k) arrays per date."""
    lambdas = []
    for r, X in zip(returns_panel, exposures_panel):
        X1 = sm.add_constant(np.asarray(X, dtype=float))
        # .params is an ndarray for ndarray inputs (no .values); wrap in asarray.
        lambdas.append(np.asarray(sm.OLS(np.asarray(r, dtype=float), X1).fit().params))
    L = np.array(lambdas)
    mean = L.mean(0)
    tstat = mean / (L.std(0, ddof=1) / np.sqrt(L.shape[0]))
    return mean, tstat        # premia and their t-stats (consider Newey-West for autocorr)
```

Use Newey-West / Shanken corrections when `λ_t` is autocorrelated or exposures are estimated.

### 8.5 Factor crowding and decay

Published/popular factors decay (McLean-Pontiff: ~half the premium disappears post-publication). Monitor crowding (correlated positioning, valuation spreads of the factor, short interest) and IC over rolling windows; a factor whose live IC has drifted to zero is dead regardless of its backtest.

---

## 9. Most common statistical mistakes (detect → fix)

1. **Same-bar execution / look-ahead.** PnL uses `position * return_t` not `position.shift(1) * return_t`. → Always lag positions; audit every signal for information timing.
2. **Multiple-testing inflation.** Reporting the best of N configs as the Sharpe. → DSR/PSR, track honest `n_trials`, OOS via CPCV, report PBO.
3. **Deflating by the raw config count.** A correlated grid is not N independent bets — using the raw N over-deflates the DSR and can bury a real edge. → Estimate the **effective** N from the trial-return correlation matrix (participation ratio, §1.4.1) and deflate by `N_eff`.
4. **Shuffled/plain CV on time series.** → Purged + embargoed K-fold / CPCV; never `shuffle=True`.
5. **Survivorship / delisting bias.** Universe excludes dead tickers. → Use point-in-time, survivorship-free data; include delistings at the delist return.
6. **Wrong annualization.** Mixing 252/365, annualizing per-period Sharpe by anything but √PPY, ignoring autocorrelation. → Fix PPY explicitly; use Lo (2002) when autocorrelated.
7. **Ignoring costs/slippage/capacity.** Gross Sharpe ≫ net. → Subtract realistic costs = turnover × spread/impact; check capacity at target AUM.
8. **Overfitting parameters.** Tuning on the full sample. → Walk-forward, hold out a true test set, prefer fewer parameters; report DSR.
9. **Correlation mistaken for cointegration** in pairs. → Test cointegration (Engle-Granger/Johansen); re-estimate live.
10. **Gaussian VaR on fat-tailed P&L.** → Historical/CVaR or Student-t/Cornish-Fisher; stress test.
11. **iid bootstrap on dependent returns.** → Block/stationary bootstrap.
12. **Ignoring non-normality in significance.** Bare Sharpe with negative skew. → PSR, inspect skew/kurtosis and tails.
13. **Reusing the test set.** Each peek leaks. → One-shot test; tune only on validation.
14. **Positional indexing on pandas Series.** `r[t-1]` is label-based and silently wrong. → `np.asarray(r)` before integer-positional loops.
