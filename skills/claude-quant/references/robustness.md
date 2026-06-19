# Robustness & Overfitting Lab

Stress-testing whether a backtested edge is real. A single backtest produces one number; this reference is about turning that number into a *distribution* and asking how often an edge this good appears under the null hypothesis of no skill. Complements `templates/validation.py` (purged/combinatorial CV), `references/stats-risk.md` (deflated/probabilistic Sharpe, PBO), `templates/metrics.py` (Sharpe SE, PSR, DSR), and `templates/costs.py` (cost models, breakeven).

Conventions used throughout: simple returns compound multiplicatively; annualized Sharpe = `mean(excess)/std(excess, ddof=1)*sqrt(ppy)` with daily `ppy=252` (iid assumption — see autocorrelation caveat in §3); positions are lagged versus the returns they earn, `pnl_t = pos.shift(1) * ret_t`. All code below operates on NumPy arrays — coerce pandas inputs with `np.asarray(...)` first to avoid index-alignment surprises during resampling/permutation.

---

## 1. Why a single backtest path is nearly worthless

A backtest is one realized path drawn from a vast space of strategies, parameters, universes, and date ranges you *could* have tried. The reported Sharpe is, in effect, an **order statistic** of everything you searched — and the maximum of N noisy estimates is biased high even when every underlying strategy has zero true edge.

**Three failure modes that all reduce to the same thing — you optimized against noise:**

- **Data snooping / multiple testing.** Test 20 strategies at the 5% level on pure noise and you *expect* roughly 1 "significant" result (each has a 5% chance independently; the count is Binomial, not a guarantee of exactly 1). Test 1,000 configurations (the realistic count once you sweep parameters, universes, and entry rules) and spurious winners are all but certain. The reported p-value of the winner is meaningless unless adjusted for the number of trials.

- **In-sample optimization.** Choosing parameters that maximized historical PnL fits the idiosyncratic noise of that specific path. The fitted edge does not generalize because most of it was noise to begin with.

- **The garden of forking paths.** Even with *one* final configuration, the analyst made dozens of undocumented choices conditional on seeing the data: dropping an outlier, picking a winsorization level, choosing 2015 as the start because pre-2015 "had a structural break." Each fork is an implicit test. The effective number of trials is far larger than the count of formally compared strategies.

**The mental shift.** Stop asking "what Sharpe did it print?" Start asking "what is the *distribution* of Sharpe under the null, and where does my observed value fall in it?" Every technique below builds that distribution a different way:

| Technique | Null / resampling unit | Question answered |
|---|---|---|
| MCPT (permute returns) | shuffled return order | Is the equity curve more than a random ordering? |
| MCPT (permute signal alignment) | shuffled signal-to-return mapping | Does timing add value beyond average exposure? |
| Sign-flip | random ±1 on returns | Is the mean return distinguishable from zero (symmetric null)? |
| Block/stationary bootstrap | resampled blocks of the realized series | What is the sampling CI of Sharpe / CAGR / MDD? |
| White's RC / Hansen's SPA | bootstrap over the best-of-N | Does the *best* strategy beat the benchmark after snooping? |
| Deflated Sharpe / PBO | analytical / CSCV | Sharpe adjusted for N trials; P(IS-best is OOS-loser) |

---

## 2. Monte Carlo permutation tests (MCPT)

A permutation test builds the null distribution by repeatedly destroying the structure you claim is real, recomputing the metric each time, and measuring how extreme the observed value is. No distributional assumptions beyond the relevant invariance (exchangeability or sign-symmetry) — the null is generated from your own data.

**p-value (one-sided, larger-is-better), with the +1 correction so it is never exactly zero:**

```
p = (1 + #{ perm_metric >= observed_metric }) / (n_perm + 1)
```

The `+1` in numerator and denominator counts the observed sample itself as one draw from the null — without it a test with 999 permutations could report `p=0`, which overstates significance (the true p can never be proven to be exactly zero from a finite resample).

### Three flavors, three nulls — they answer different questions

**(a) Permute the returns.** Shuffle the order of the realized return series and recompute the metric (e.g. Sharpe of the strategy applied to the shuffled returns, or the Sharpe of the shuffled returns themselves). This destroys **serial dependence / autocorrelation** and any momentum/mean-reversion structure. Use it to ask: *is the equity curve's shape more than a lucky ordering of the same return pool?* Note it does **not** test exposure timing if your signal is recomputed off the shuffled prices — be explicit about what gets permuted. Note also that simple time-Sharpe is invariant to reordering of an iid return pool (mean and std are order-free), so flavor (a) is only informative for *path-dependent* metrics (MDD, Calmar, run-up/run-down) or when the signal is recomputed off the permuted path.

**(b) Permute the signal-return alignment.** Keep the returns in order; shuffle the **mapping** from signals to the returns they earn (or circularly rotate the signal series). This holds the marginal distribution of both signals and returns fixed and breaks only their *alignment in time*. Use it to ask the sharpest question in tactical strategies: *does the timing add value beyond average exposure?* A long-biased strategy in a bull market will look great under flavor (a) but should collapse under flavor (b) if the timing is noise — because the average exposure (the net long bias) is preserved while the timing is randomized. (Caveat: i.i.d. permutation of the signal also destroys signal *autocorrelation*, so under the null the permuted strategy trades far more often than the real one; if turnover/cost or signal persistence matters, prefer a circular rotation of the signal, which preserves its autocorrelation while breaking alignment.)

```python
import numpy as np

def mcpt_signal_alignment(signal, ret, metric_fn, n_perm=1000, seed=0):
    """Permute signal->return alignment. signal and ret are aligned 1-D arrays;
    metric_fn(pos, ret) -> scalar, larger=better, and must lag pos internally
    using array ops (e.g. pos[:-1]*ret[1:]), NOT pandas .shift, since inputs are
    ndarrays."""
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=float)
    ret = np.asarray(ret, dtype=float)
    obs = metric_fn(signal, ret)
    count = 0
    for _ in range(n_perm):
        perm_sig = rng.permutation(signal)          # break timing, keep exposure distribution
        if metric_fn(perm_sig, ret) >= obs:
            count += 1
    p = (1 + count) / (n_perm + 1)
    return obs, p
```

> Detect: a strategy that survives return-shuffling but dies under alignment-shuffling is earning *exposure premium*, not timing alpha. That may still be a fine product (it's beta/risk premium), but do not market it as a timing signal.

**(c) Sign-flip (for mean > 0).** When the claim is simply "mean return is positive," multiply each return by an independent random `±1` and recompute the mean (or Sharpe). This is the randomization/sign-test analog of a one-sample test of zero mean, and respects the magnitude distribution. It is exact **only under the null of symmetry about zero** (the per-period returns are distributionally symmetric); for skewed returns it tests the symmetric-null rather than the mean directly, so pair it with the bootstrap CI of the mean. Best for low-frequency strategies where the hypothesis is directional.

```python
def mcpt_sign_flip(pnl, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, dtype=float)              # coerce: broadcasting against a Series misaligns
    obs = pnl.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, pnl.size))
    null = (flips * pnl).mean(axis=1)
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, p
```

**Caveats.** (1) Permutation tests assume an *invariance* under the null — exchangeability for flavors (a)/(b), sign-symmetry for (c); flavor (a) is not exchangeable if returns are strongly autocorrelated — which is exactly why flavor (a) tests for that structure, but means you should not use plain return-shuffling to test the *mean*. (2) Use ≥1,000 permutations for a stable p around 0.05; the Monte Carlo standard error of p̂ is ≈ `sqrt(p̂(1-p̂)/n_perm)` — at p̂=0.05, n_perm=1000 gives SE≈0.0069. (3) Fix the seed and report it.

---

## 3. Bootstrap confidence intervals

Permutation tests give a p-value against a null; the bootstrap gives a **confidence interval** on the statistic itself by resampling the realized series with replacement and recomputing the metric on each resample.

**IID bootstrap.** Draw `len(returns)` observations with replacement, recompute Sharpe/CAGR/MDD, repeat B times, take percentiles. **Only valid if returns are serially independent.** Financial returns rarely are: volatility clusters, trend strategies have autocorrelated PnL, and drawdowns are inherently path-dependent. IID resampling shatters this dependence and produces CIs that are far too tight for path-dependent statistics — especially for max drawdown, which depends entirely on the *order* of returns. (For order-free statistics like the mean or per-period Sharpe, iid resampling distorts the CI less, but still ignores dependence in the variance estimate.)

**Block bootstrap.** Resample contiguous **blocks** of length `L` to preserve short-range dependence. The cost is choosing `L`: too short under-captures dependence, too long reduces effective sample size. A fixed block length also leaves a discontinuity at block joins.

**Stationary bootstrap (Politis & Romano, 1994).** Use blocks of **random geometric length** with mean `L` (each step, continue the current block with probability `1 - 1/L`, i.e. start a new block at a random index with probability `p = 1/L`, wrapping circularly). This produces a resampled series that is itself stationary (the fixed-block scheme is not), which is why it is a sound default for serially-dependent financial returns. Heuristic: set the mean block length on the order of the autocorrelation horizon of the *PnL* series (often tens of days for trend strategies; near 1 for a true zero-autocorrelation alpha — in which case the iid bootstrap is fine).

```python
import numpy as np

def stationary_bootstrap_indices(n, mean_block, rng):
    """Politis-Romano (1994) stationary bootstrap index generator."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(n)            # start new block at random point
        else:
            idx[t] = (idx[t - 1] + 1) % n       # continue block, wrap circularly
    return idx

def bootstrap_ci(series, metric_fn, mean_block=20, B=2000, alpha=0.05, seed=0):
    """Percentile CI for metric_fn(resampled_series). Use mean_block<=1 for iid
    bootstrap. NOTE: percentile CIs can be biased for skewed statistics (Sharpe,
    MDD); for a bias-corrected interval use BCa. Resample the OOS path, never the
    IS-selected one (see Pitfalls)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(series, dtype=float)
    n = len(x)
    stats = np.empty(B)
    for b in range(B):
        idx = (rng.integers(n, size=n) if mean_block <= 1
               else stationary_bootstrap_indices(n, mean_block, rng))
        stats[b] = metric_fn(x[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return stats.mean(), (lo, hi)
```

**What to bootstrap.** Sharpe, CAGR, and max drawdown are the high-value targets. A Sharpe of 1.2 with a 95% CI of `[-0.1, 2.4]` is not a strategy — it is noise with a point estimate. Max-drawdown CIs are critical for sizing: the realized MDD is one draw, and the bootstrap routinely shows the 95th-percentile MDD is materially worse than the historical worst.

> Autocorrelation caveat (also affects Sharpe annualization). The `sqrt(252)` scaling assumes iid daily returns. Positive PnL autocorrelation inflates annualized Sharpe; negative deflates it. Use `templates/metrics.py:lo_annualization_factor` for the autocorrelation-adjusted annualization. If you bootstrap Sharpe with the stationary bootstrap, the CI already reflects this dependence — prefer the bootstrap CI over the analytic standard error. The correct iid/Lo (2002) standard error of the **per-period** Sharpe is `≈ sqrt((1 + 0.5*SR_pp²)/n)` (annualized: multiply by `sqrt(252)`); this is what `templates/metrics.py:sharpe_se` returns. Note the naive `SR*sqrt(n)` t-stat omits the `(1 + 0.5*SR²)` correction — see `sharpe_tstat`.

---

## 4. Data-snooping-adjusted tests (best of N strategies)

When you select the best of N strategies/parameterizations, you must test whether that *maximum* beats a benchmark after accounting for the search. Two standard tools control the family-wise error rate via the bootstrap.

**White's Reality Check (2000).** Let `f_k,t` be the per-period performance of strategy `k` relative to a benchmark (e.g. excess return over a benchmark/zero, or a loss differential). Define `f̄_k = mean_t f_k,t`. The test statistic is the scaled best mean:

```
V = sqrt(T) * max_k f̄_k
```

The null ("no strategy beats the benchmark") distribution is built by stationary-bootstrapping the `f_k,t` matrix (resample *rows*/time jointly across all k to preserve cross-strategy correlation), recentering each strategy's resampled mean by subtracting its full-sample mean, and taking the max across k each bootstrap iteration. The Reality Check p-value is the fraction of bootstrap maxima ≥ V.

**Hansen's SPA (Superior Predictive Ability, 2005)** improves on RC in two ways: (1) it **studentizes** each strategy by its own standard error (so a single high-variance strategy can't dominate the null distribution), and (2) it down-weights strategies that are clearly inferior (RC is conservative because hopeless strategies still inflate the null max). SPA is the preferred default; RC is the simpler conceptual baseline. Both require a consistent variance estimate (`ŝ_k` should use the same dependence-robust/bootstrap variance as the resampling, not the naive iid std).

```
# RC null max per bootstrap:  max_k sqrt(T) * (f̄_k* - f̄_k)
# SPA null max per bootstrap:  max_k max(0, sqrt(T) * (f̄_k* - f̄_k) / ŝ_k)   [studentized; the truncation/recentering of clearly-inferior strategies is the "consistent" SPA_c variant]
```

> Detect: if the unadjusted best-strategy p is 0.01 but the SPA p is 0.40, the "edge" is a snooping artifact — it is the kind of result you'd expect from searching that many strategies on noise. Fix: report the SPA p-value, not the winner's naive p-value. See also the Deflated Sharpe Ratio (§6), which addresses the same problem analytically for the single best Sharpe.

---

## 5. Parameter sensitivity: plateaus vs cliffs

A genuine edge is **robust to parameter choice** — neighbors of the chosen parameters perform similarly, forming a broad **plateau**. An overfit edge sits on a sharp isolated **peak** surrounded by cliffs, because the optimizer found the one cell where noise aligned.

**Diagnose with a heatmap.** Sweep two parameters (e.g. lookback × holding period), color by Sharpe, and look at the *neighborhood* of the optimum, not the optimum itself.

```
Robust (plateau)                 Overfit (cliff/peak)
lookback ->                      lookback ->
  1.1 1.2 1.3 1.2 1.1              0.1 0.2 2.4 0.1 0.0
  1.2 1.3 1.4 1.3 1.2             -0.2 0.3 0.2 0.1 0.1
  1.1 1.3 1.3 1.3 1.1              0.0 0.1 0.2 0.0 0.2
```
The left strategy's 1.4 is trustworthy because every neighbor is ~1.2–1.3. The right strategy's 2.4 is a trap: its neighbors are near zero or negative.

**Quantify the plateau.** Don't eyeball it. Useful metrics:
- **Neighbor degradation.** `peak_sharpe - mean(neighborhood Sharpe)`. Small = plateau. A drop of >50% from peak to neighbor mean is a red flag.
- **Plateau fraction.** Fraction of the grid achieving ≥ X% (say 70%) of the peak. Higher = broader plateau.
- **Local gradient / curvature.** Large absolute Sharpe gradient around the optimum indicates a cliff.

```python
import numpy as np

def plateau_metrics(grid, peak_frac=0.7):
    """grid: 2D array of Sharpe over a parameter sweep. Edge/corner optima have
    fewer than 8 neighbors; the slice below handles that. degradation_pct can be
    negative if a neighbor exceeds the (NaN-masked) peak cell, and is undefined
    when peak==0."""
    peak = np.nanmax(grid)
    i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
    nb = grid[max(0, i-1):i+2, max(0, j-1):j+2]
    n_other = np.sum(~np.isnan(nb)) - 1            # exclude the peak cell itself
    neighbor_mean = (np.nansum(nb) - peak) / max(1, n_other)
    return {
        "peak": peak,
        "neighbor_mean": neighbor_mean,
        "degradation_pct": (100 * (peak - neighbor_mean) / abs(peak)
                            if peak != 0 else float("nan")),
        "plateau_fraction": np.nanmean(grid >= peak_frac * peak),
    }
```

> Best practice: **report the plateau-center / median-neighborhood Sharpe, not the peak.** Better still, pick parameters by walk-forward (`templates/validation.py`) so the choice is made out-of-sample, then confirm the chosen cell sits on a plateau.

---

## 6. Deflated & Probabilistic Sharpe, PBO/CSCV

These are the analytical counterparts to the resampling tests above; they are documented and implemented elsewhere — point to those, don't duplicate.

- **Probabilistic Sharpe Ratio (PSR):** P(true **per-period** Sharpe > a benchmark per-period Sharpe) given the estimate, sample length, and the higher moments (skew/kurtosis) of returns. Corrects for non-normal returns and short samples. See `templates/metrics.py:probabilistic_sharpe_ratio` (note: `benchmark_sr` is in per-period units) and `references/stats-risk.md`.

- **Deflated Sharpe Ratio (DSR):** PSR with the benchmark set to the **expected maximum Sharpe under the null given N trials** — the analytical answer to §4's multiple-testing problem for the single best Sharpe. Requires an honest count of trials `N` (the garden of forking paths means `N` is larger than you think) and the **standard deviation (across trials) of the per-period trial Sharpes** (`trial_sharpe_std` — a std, not a variance). Assumes roughly independent trials; correlated grid searches have a smaller *effective* N, making the test conservative. See `templates/metrics.py:deflated_sharpe_ratio` / `expected_max_sharpe` and `references/stats-risk.md`.

- **PBO via CSCV (Probability of Backtest Overfitting):** Combinatorially split the data, find the in-sample-best configuration in each split, and measure how often it lands in the *bottom half* out-of-sample. PBO is that fraction; >0.5 means the selection process is anti-predictive. See `references/stats-risk.md` and the combinatorial CV machinery in `templates/validation.py` (`CombinatorialPurgedKFold`).

The relationship: §2–§5 are empirical/resampling stress tests on a path; §6 are the statistically principled corrections for selection bias. Use both — they fail in different ways.

---

## 7. Synthetic-data & regime stress

Even a path that survives the above is one realization of one history. Stress it against worlds that didn't happen but could have.

- **Synthetic / simulated paths.** Re-run the *strategy logic* on bootstrapped or model-simulated price paths (block-bootstrapped returns, or a fitted GARCH/jump model) to get the distribution of outcomes under plausible alternate histories. A strategy whose median synthetic Sharpe is near zero was fit to one lucky path. Caution: if you simulate the *price* and recompute signals, ensure the simulator preserves the structure your signal exploits (e.g. a mean-reversion alpha needs a simulator with mean reversion), or you'll trivially reject everything for the wrong reason.

- **Regime partitioning.** Split history into regimes — bull/bear, high/low volatility (e.g. VIX terciles), rising/falling rates, pre/post a known structural break — and report performance per regime. Define regime boundaries point-in-time (no peeking at the full-sample distribution to set thresholds, or you leak future info into the partition). An edge concentrated in a single regime (e.g. only works when vol spikes) is a conditional bet, not a general strategy; size and market it accordingly.

- **Cost sensitivity.** Re-run across a grid of cost assumptions (e.g. 0×, 1×, 2×, 5× your base spread+slippage+impact estimate from `templates/costs.py`). Report the **breakeven cost** at which Sharpe (or net return) crosses zero — `templates/costs.py:breakeven_cost_bps` computes this directly from gross return and turnover. High-turnover strategies often have a breakeven cost barely above realistic levels — that is fragility, not edge.

- **Start-date / end-date sensitivity.** Roll the start date forward by quarters and the end date back; plot Sharpe vs window. If the edge depends on including one specific year (often a single crisis), it is not robust.

- **Universe sensitivity.** Re-run on random subsamples of the universe (drop 20% of names), on sub-sectors, and on a deliberately *adversarial* subset (e.g. excluding the top-3 PnL contributors). If removing a handful of names kills the edge, the result is name-specific, not systematic.

```python
import numpy as np

def breakeven_cost(gross_pnl, turnover, base_cost, grid=(0, 1, 2, 3, 5, 8)):
    """Annualized net Sharpe vs cost multiplier; inspect where it crosses zero.
    gross_pnl, turnover, base_cost are per-period arrays in RETURN units
    (cost charged = multiplier * base_cost * turnover, in the same units as
    gross_pnl). Returns NaN for a degenerate (zero-variance) net series."""
    gross_pnl = np.asarray(gross_pnl, dtype=float)
    out = {}
    for m in grid:
        net = gross_pnl - m * base_cost * turnover
        sd = net.std(ddof=1)
        out[m] = (net.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    return out  # inspect the multiplier at which net Sharpe -> 0
```

---

## 8. "Is this backtest trustworthy?" checklist

Run before allocating capital. Treat any "no" as a blocker until explained.

- [ ] **Out-of-sample exists and was touched once.** Final config evaluated on a holdout that played no role in design/parameter selection. (`templates/validation.py`)
- [ ] **CV purged + embargoed.** Time-series CV removes leakage around split boundaries; no future label bleeds into training. (`PurgedKFold` / `CombinatorialPurgedKFold` in `templates/validation.py`)
- [ ] **Positions lagged.** `pnl_t = pos.shift(1)*ret_t`; signal uses only information available at decision time (no look-ahead).
- [ ] **MCPT p-value reported.** At least the signal-alignment permutation (§2b); ideally also sign-flip. p < 0.05 with ≥1,000 permutations.
- [ ] **Bootstrap CIs reported** for Sharpe, CAGR, and MDD using a block/stationary bootstrap — and the Sharpe CI excludes zero.
- [ ] **Multiple testing accounted for.** Deflated Sharpe with an honest trial count, or SPA/Reality Check for best-of-N. Naive winner p-value is *not* sufficient.
- [ ] **Parameters sit on a plateau.** Neighbor degradation modest; reported metric is plateau-center, not peak. Heatmap inspected.
- [ ] **PBO < 0.5** via CSCV (`references/stats-risk.md`).
- [ ] **Costs realistic and stress-tested.** Breakeven cost comfortably above realistic levels; survives 2–5× cost grid. (`templates/costs.py`)
- [ ] **Regime-robust.** Edge not confined to one regime, one crisis year, or one window; survives start/end-date rolls.
- [ ] **Universe-robust.** Survives dropping random names and the top PnL contributors.
- [ ] **Trial count disclosed.** Number of strategies/parameters/forks actually tried is documented (garden of forking paths).
- [ ] **Seeds fixed and reported** for every stochastic test (permutation, bootstrap, synthetic).

---

## Pitfalls (detect / fix)

| Pitfall | Detect | Fix |
|---|---|---|
| **Single-path backtest** | One equity curve, one Sharpe, no CIs or p-values reported | Build the distribution: MCPT p-value (§2) + block-bootstrap CIs on Sharpe/CAGR/MDD (§3). Report the interval, not the point. |
| **Optimizing then reporting the peak** | Headline metric equals the grid maximum; neighbors much worse | Choose parameters out-of-sample (walk-forward, `templates/validation.py`); report plateau-center/median-neighborhood metric (§5). |
| **Ignoring multiple testing** | Winner's naive p<0.05 after sweeping dozens/hundreds of configs; trial count undisclosed | Deflated Sharpe with honest N (`templates/metrics.py`); SPA/Reality Check for best-of-N (§4). |
| **Narrow parameter cliff** | Sharp isolated peak; >50% degradation to neighbor mean; high local gradient | Inspect heatmap; quantify plateau width (§5); reject if no plateau or re-spec the signal to be smoother. |
| **No out-of-sample** | Entire history used for design and reporting; holdout reused | Reserve an untouched holdout; purged+embargoed CV; evaluate final config exactly once (`templates/validation.py`). |
| **In-sample bootstrap** | Bootstrap drawn from the *optimized* sample/config to "confirm" the edge | Bootstrap is a sampling-uncertainty tool, not a selection-bias tool — it cannot undo overfitting. Pair with OOS + DSR/PBO/SPA. Resample the OOS path, not the IS-selected one. |
| **IID bootstrap on dependent returns** | CIs implausibly tight; MDD CI near the point estimate | Use block/stationary bootstrap (Politis-Romano, §3) with mean block ~ PnL autocorrelation horizon. |
| **Permuting the wrong thing** | Strategy "passes" but the null didn't actually break the claimed structure | Match the permutation to the hypothesis: alignment-shuffle (or circular rotation) for timing, sign-flip for mean, return-shuffle for path structure (§2). |
| **pandas index leaking into resamples** | Permuted/bootstrapped arrays realign on the original index, silently undoing the shuffle | Coerce to NumPy with `np.asarray(...)` before permuting/resampling; index alignment quietly reverses a Series permutation. |

---

**See also:** `templates/metrics.py` (`sharpe_se`, `sharpe_tstat`, `lo_annualization_factor`, `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `expected_max_sharpe`) · `templates/validation.py` (`PurgedKFold`, `CombinatorialPurgedKFold`, walk-forward CV) · `templates/costs.py` (`breakeven_cost_bps`, slippage/impact models) · `references/stats-risk.md` (DSR/PSR/PBO detail) · `references/pitfalls.md`. Runnable MCPT / stationary-bootstrap / Reality-Check / parameter-plateau helpers are in `templates/robustness.py`.