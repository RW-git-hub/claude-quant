# Portfolio Optimization

Turning expected returns and risk into weights, *robustly*. The core tension: classical mean-variance is optimal given true inputs, but inputs are estimated with error, and the optimizer amplifies that error. Most of this file is about controlling that amplification.

Conventions used throughout:
- `w` = weight vector (N assets), `mu` = expected (excess) returns, `Sigma` = return covariance, `rf` = risk-free rate, `1` = vector of ones.
- Returns are **simple** (aggregate multiplicatively across time, additively across assets within a period — which is what `w'ret` assumes). `Sigma` is the covariance of the *same-frequency, same-units* returns you will earn; annualize consistently (`Sigma_ann ≈ Sigma_daily * 252` holds only under zero autocorrelation — if returns are serially correlated, scaling variance by the number of periods is biased; use a Newey-West / overlapping-window adjustment).
- `lambda` (risk-aversion) > 0. Larger `lambda` -> less risk taken.
- All inputs (`mu`, `Sigma`) must be **point-in-time**: estimated only from data available strictly before the rebalance timestamp. Weights set at `t` earn returns over `(t, t+1]` (`pnl_{t+1} = w_t' ret_{t+1}`), mirroring `pnl_t = pos.shift(1)*ret_t`.
- Code references: `templates/validation.py` provides `constant_correlation_shrinkage` (Ledoit-Wolf constant-correlation target) and purged/embargoed CV (`PurgedKFold`, `CombinatorialPurgedKFold`); `templates/costs.py` provides the transaction-cost models. `templates/portfolio.py` ships runnable, self-tested implementations (min-variance, max-Sharpe, mean-variance, risk parity / ERC, HRP, Black-Litterman); reuse the shrinkage in `templates/validation.py` and the cost models in `templates/costs.py`.

---

## 1. Mean-Variance Optimization (Markowitz)

### Objective

Maximize a quadratic utility trading off return against variance:

```
max_w   w'mu - (lambda/2) w'Sigma w
```

Unconstrained, the first-order condition `mu - lambda*Sigma*w = 0` gives the closed form:

```
w* = (1/lambda) * inv(Sigma) * mu                      # unconstrained, no budget constraint
```

Note this does *not* generally satisfy `1'w = 1`. With the budget constraint `1'w = 1`, the solution is a combination of the global minimum-variance and tangency portfolios (the **two-fund theorem**):

```
w* = w_mv + (1/lambda) * ( inv(Sigma)*mu - (1'inv(Sigma)*mu) * w_mv )
```

where `w_mv = inv(Sigma)1 / (1'inv(Sigma)1)`. (Check: `1'w* = 1` since `1'w_mv = 1` and the bracketed term is orthogonal to `1` after the subtraction. As `lambda -> inf`, `w* -> w_mv`.)

### The efficient frontier

The set of portfolios with minimum variance for each level of target return. Parametrically, fix target return `m`, solve:

```
min_w  w'Sigma w   s.t.  w'mu = m,  1'w = 1
```

Tracing `m` sweeps out a hyperbola in `(sigma, return)` space; the upper branch (returns above the GMV return) is the **efficient frontier**. With a risk-free asset, the line from `rf` tangent to the frontier is the capital market line; its tangency point is the max-Sharpe portfolio (Section 2).

### Why MVO is fragile: error-maximization

Markowitz is provably optimal *given true `mu` and `Sigma`*. In practice both are estimated, and the optimizer behaves as an **error-maximizer** (Michaud, 1989): it places the largest bets exactly where estimation error makes assets look most attractive.

Two compounding failure modes:

1. **Sensitivity to `mu`.** Expected returns are estimated with enormous standard error (the SNR of a Sharpe-0.5 daily series is tiny — you need decades to pin a mean down). `inv(Sigma)` multiplies `mu`, and small changes in `mu` produce wildly different, often extreme long/short weights. Best & Grauer (1991) show the optimal weights are acutely sensitive to small `mu` perturbations.
2. **Ill-conditioned `Sigma`.** The sample covariance of `N` assets from `T` observations is near-singular when `T` is not >> `N` (and *singular*, hence non-invertible, when `T <= N`). `inv(Sigma)` then has huge entries along low-variance eigen-directions; the optimizer levers into those directions because they look like "free" risk reduction. The condition number `kappa(Sigma) = lambda_max/lambda_min` measures this.

> Detect: tiny input changes -> large weight changes; weights with extreme leverage or many near-±caps; `np.linalg.cond(Sigma)` >> 1e3; eigenvalues of `Sigma` spanning many orders of magnitude.

The rest of this file (shrinkage, dropping `mu`, constraints, BL, risk-based) exists to tame this.

---

## 2. Minimum-Variance and Maximum-Sharpe (Tangency) Portfolios

These are the two reference portfolios that span the budget-constrained frontier.

### Global minimum-variance (GMV)

Uses **only `Sigma`** — no `mu`, so it sidesteps the worst estimation problem.

```
w_mv = inv(Sigma) * 1 / (1' inv(Sigma) * 1)
```

Solves `min_w w'Sigma w  s.t.  1'w = 1`. Empirically, GMV often beats MVO out of sample precisely because it doesn't trust `mu`. It still inherits `Sigma` ill-conditioning -> shrink `Sigma` (Section 3).

### Tangency (maximum-Sharpe) portfolio

The portfolio maximizing the Sharpe ratio `(w'(mu - rf))/sqrt(w'Sigma w)` with returns in *excess* of `rf`:

```
w_tan ∝ inv(Sigma) * (mu - rf)                          # then normalize so 1'w = 1
w_tan  = inv(Sigma)(mu - rf) / (1' inv(Sigma)(mu - rf))
```

(Here `mu - rf` is the vector of excess expected returns; subtract `rf` elementwise. The normalization assumes `1' inv(Sigma)(mu - rf) != 0`; if that scalar is negative the "tangency" portfolio is actually on the inefficient lower branch — a red flag that `mu` is dominated by noise.)

This is the riskiest, most `mu`-dependent point on the frontier — maximally exposed to error-maximization. Treat raw tangency weights with suspicion; they are a *theoretical* target, not a deployable allocation without shrinkage/constraints.

> Implementation: never form `inv(Sigma)` explicitly. Solve the linear system instead — `np.linalg.solve(Sigma, b)` (or `scipy.linalg.cho_solve` on a Cholesky factor when `Sigma` is SPD) is more accurate and faster than `inv(Sigma) @ b`, and `cholesky` fails loudly if `Sigma` is not positive-definite.

---

## 3. Estimation-Error Fixes

The toolkit for making MVO survive out of sample. Combine several; they are complementary.

### 3a. Covariance shrinkage

Pull the noisy sample covariance toward a structured, well-conditioned target:

```
Sigma_hat = delta * F + (1 - delta) * S
```

where `S` = sample covariance, `F` = target, `delta` in [0,1] the shrinkage intensity.

- **Constant-correlation target** (Ledoit-Wolf 2003, "Honey, I Shrunk the Sample Covariance Matrix"): keep sample variances, replace all pairwise correlations with their average. This is usually the best off-the-shelf target for *asset returns*. See `templates/validation.py:constant_correlation_shrinkage`, which computes the analytically optimal `delta`.
- **Ledoit-Wolf scaled-identity target** (Ledoit-Wolf 2004) with *analytically optimal* `delta` (minimizes expected Frobenius loss) — `sklearn.covariance.LedoitWolf`. No tuning required, but note its target is a scaled identity, which is often *worse* than the constant-correlation target for return covariances (it shrinks correlations toward zero). Do not confuse it with a single-index/market-factor target, which is a different LW estimator not implemented in sklearn.
- **Oracle Approximating Shrinkage (OAS)** — `sklearn.covariance.OAS`, derived to improve on LW's `delta` under an assumed-Gaussian model; can help when returns are near-Gaussian, but the same identity-target caveat applies.

Shrinkage lowers the condition number, pulls in extreme eigenvalues, and demonstrably improves out-of-sample GMV/MVO. It is the single highest-leverage fix.

### 3b. Drop `mu` entirely — risk-based portfolios

Since `mu` is the noisiest input, many practitioners discard it and build portfolios from `Sigma` alone: GMV (Section 2), risk parity/ERC (Section 5), HRP (Section 6), inverse-vol. You give up the *theoretical* return optimization but remove the dominant error source. Frequently wins out of sample. (This wins *only* when you have no genuinely informative return forecast; if you do have alpha, throwing away `mu` throws away the edge — shrink it instead, Section 3e/4.)

### 3c. Constraints (Section 8 expands)

Long-only and box constraints act as **implicit shrinkage** (Jagannathan & Ma, 2003): for GMV, a no-short constraint is mathematically equivalent to shrinking the covariances of the assets that would otherwise be shorted, improving out-of-sample performance even when the constraint is "wrong." Gross-leverage and position caps bound the damage from any single bad estimate.

### 3d. Resampling (Michaud)

Bootstrap or Monte-Carlo: draw many simulated return histories consistent with `(mu, Sigma)`, optimize each, average the resulting weights. Produces smoother, more diversified frontiers. Costs: computationally heavy, no clean closed form, statistically ad hoc (no agreed objective it optimizes), and it averages over the *same* biased point estimates — it addresses sampling variance around `(mu, Sigma)`, not model bias in them.

### 3e. Robust / Bayesian `mu`

Shrink `mu` too — toward the grand mean (James-Stein), toward GMV-implied or equal-weight-implied returns, or via Black-Litterman (Section 4), which is the principled way to inject *and* shrink views.

---

## 4. Black-Litterman

A Bayesian framework that fixes MVO's `mu` problem at the source: instead of feeding raw historical means, start from a sensible **equilibrium prior** and tilt it only by the views you actually hold, weighted by your confidence.

### Step 1 — reverse-optimize the market prior

Assume the market-cap portfolio `w_mkt` is the MVO solution; back out the `mu` that makes it optimal:

```
pi = lambda * Sigma * w_mkt
```

`pi` is the **implied equilibrium excess returns**. `lambda` here is the market's implied risk aversion, often set as `lambda = (E[R_mkt] - rf) / sigma_mkt^2` (Sharpe ratio of the market divided by its volatility). This prior is, by construction, well-behaved: it reproduces the diversified market portfolio.

### Step 2 — encode views

- `P` (K×N): each row a portfolio expressing one view (absolute: a single 1; relative: +1/−1 pairs, conventionally summing to 0 within the row).
- `Q` (K×1): the expected (excess) return of each view portfolio.
- `Omega` (K×K): view uncertainty (usually diagonal); larger -> less confident. Common heuristic `Omega = diag(P (tau Sigma) P')` (He-Litterman), which makes view confidence proportional to prior variance.
- `tau`: small scalar (e.g. 0.025–0.05) scaling the prior's uncertainty `tau*Sigma`.

### Step 3 — posterior expected returns

```
mu_BL = inv( inv(tau*Sigma) + P' inv(Omega) P )
        @ ( inv(tau*Sigma) @ pi + P' inv(Omega) @ Q )
```

(Equivalently the Theil mixed-estimator / GLS form.) The posterior covariance of the mean estimate is `M = inv(inv(tau*Sigma) + P' inv(Omega) P)`; some implementations use `Sigma_BL = Sigma + M` for the subsequent optimization to account for estimation uncertainty in `mu_BL`.

Feed `mu_BL` (and optionally `Sigma_BL`) into Section 1's MVO. Note: `pi`, `Q`, and `mu_BL` must all be on the same return frequency and all in *excess* terms.

### Why it stabilizes MVO

- With **no views** (`P` empty), `mu_BL = pi` and the MVO optimizer returns `w_mkt` (up to the `lambda` used) — diversified by default. Raw MVO with a noisy historical `mu` returns garbage; BL returns the market.
- Views move weights **smoothly and locally**: a relative view on A vs B tilts A and B (and their correlated neighbors), not the entire book.
- It shrinks `mu` toward equilibrium with confidence-weighted strength, directly attacking the error-maximization that plagues Section 1.

> Pitfall: `Omega` set too small (overconfident views) collapses BL back into raw-`mu` MVO with all its fragility. Calibrate `Omega` to honest confidence, and sanity-check that posterior weights stay close to `w_mkt` for weak views.

---

## 5. Risk Parity / Equal Risk Contribution (ERC)

Allocate so each asset contributes *equally to portfolio risk*, rather than equal capital.

### Risk decomposition

Portfolio volatility `sigma_p = sqrt(w'Sigma w)` is homogeneous of degree 1 in `w`, so by Euler's theorem it decomposes additively:

```
MRC_i = (Sigma w)_i / sigma_p              # marginal risk contribution
RC_i  = w_i * (Sigma w)_i / sigma_p        # total risk contribution
sum_i RC_i = sigma_p
```

Often written on the variance scale (drop the `1/sigma_p`): `RC_i = w_i (Sigma w)_i`, with `sum_i RC_i = w'Sigma w`.

### ERC condition

Find `w` (long-only, `1'w = 1`) such that all `RC_i` are equal:

```
w_i * (Sigma w)_i = w_j * (Sigma w)_j   for all i, j
```

No closed form in general (except when correlations are all equal, where ERC reduces to inverse-vol). Solve as a convex problem — e.g. the convex log-barrier formulation (Spinu 2013 / Maillard, Roncalli & Teïletche 2010):

```
min_w  0.5 * w'Sigma w  -  c * sum_i log(w_i),   w > 0      # then rescale to 1'w=1
```

which has a unique solution (for `Sigma` PSD and `c > 0`) whose risk contributions are equal after rescaling. (Caution against directly minimizing `sum_{i,j}(RC_i - RC_j)^2`: that objective is *non-convex* in `w` and can land on a local minimum — prefer the log-barrier or a fixed-point/Newton iteration.)

### vs naive inverse-vol

**Inverse-vol** (`w_i ∝ 1/sigma_i`) equals ERC *only when all pairwise correlations are equal*. It ignores the off-diagonal of `Sigma`. ERC accounts for correlations: two highly correlated assets get less *combined* weight than inverse-vol would give, because their risk contributions overlap. Inverse-vol is the cheap, robust approximation; ERC is the correlation-aware version. Both avoid `mu`; ERC needs `Sigma` (its solve uses `Sigma w`, not `inv(Sigma)`, so it sidesteps the inversion blow-up).

> Asset-class note: risk parity is the standard for multi-asset (the "all-weather" lineage) precisely because it doesn't need return forecasts across heterogeneous assets. For a single-asset-class long/short book, ERC on the *factor* exposures is often more meaningful than on raw names.

---

## 6. Hierarchical Risk Parity (HRP)

Lopez de Prado (2016). Builds a diversified allocation **without inverting `Sigma`**, making it robust to the exact ill-conditioning that breaks MVO and even GMV.

### Algorithm

1. **Correlation distance.** Convert correlation to a metric distance:
   ```
   d_ij = sqrt( 0.5 * (1 - rho_ij) )
   ```
   Then optionally compute a Euclidean distance on the columns of `d` (distance-of-distances) as the input to clustering, for ordering stability.
2. **Hierarchical clustering.** Agglomerative linkage (Lopez de Prado's original uses single linkage; ward/complete are common alternatives) on the distance matrix to build a dendrogram of asset relationships.
3. **Quasi-diagonalization.** Reorder rows/columns of `Sigma` by the dendrogram leaf order so similar assets sit adjacent — the matrix becomes approximately block-diagonal, concentrating large covariances near the diagonal.
4. **Recursive bisection.** Walk the tree top-down. At each split into clusters `L`/`R`, compute each cluster's variance using *within-cluster inverse-variance* weights, then allocate between clusters by **inverse cluster variance**:
   ```
   w_C    ∝ 1 / diag(Sigma_C)          # within-cluster inverse-variance, normalized to sum 1
   var_C  = w_C' Sigma_C w_C           # cluster variance under those weights
   alpha_L = 1 - var_L / (var_L + var_R)   # lower-variance cluster gets MORE weight
   ```
   Scale every asset in the left sub-tree by `alpha_L`, the right by `1 - alpha_L`, and recurse. (Final leaf weights are the product of the `alpha`/`1-alpha` factors along the path; they sum to 1 and are long-only.)

### Why it's robust

- **No matrix inversion** -> immune to near-singular `Sigma`; works even when `T < N`.
- Uses only variances and the *ordering* implied by correlations, not the precise (noisy) correlation values inside an inverse, so it's stable to estimation error.
- Out-of-sample studies (Lopez de Prado; subsequent replications, with mixed results) report lower realized variance and turnover than naive MVO under realistic estimation error — though it does not uniformly beat shrinkage-GMV.

> Trade-off: HRP is a heuristic, not an optimum — it can leave risk-reduction on the table vs *true*-input GMV, and results depend on the linkage method. Treat the linkage choice as a hyperparameter to validate on a holdout, not tune to the test set.

---

## 7. Mean-CVaR and Robust Objectives (brief)

Variance penalizes upside and downside symmetrically and is a sufficient risk measure only for elliptical (e.g. Gaussian) returns. Alternatives target *tail* risk or worst-case inputs:

- **Mean-CVaR.** Replace `w'Sigma w` with Conditional Value-at-Risk (expected loss beyond the `alpha`-quantile, e.g. `alpha = 0.95`). Rockafellar & Uryasev (2000) show the CVaR minimization is a **convex program, linear (an LP) when losses are linear in `w`** over a discrete scenario set:
  ```
  min_{w, eta, u}  eta + (1/((1-alpha) S)) * sum_s u_s
  s.t.  u_s >= -w' r_s - eta,   u_s >= 0   (for each scenario s=1..S),   constraints on w
  ```
  where `r_s` is the return vector in scenario `s` and `S` the number of scenarios. Better for fat-tailed / asymmetric assets (options, crypto). Needs a scenario set (historical or simulated); the tail estimate is driven by relatively few scenarios, so it is noisy at high `alpha` / small `S`.
- **Robust optimization.** Optimize against the worst `mu` in an uncertainty set (e.g. ellipsoid `(mu - mu_hat)' inv(Omega_mu) (mu - mu_hat) <= kappa^2`). Yields more conservative, stable weights; the mean-variance version reduces to a tractable second-order cone program. Conceptually a frequentist cousin of Black-Litterman.
- **Mean-CDaR / drawdown control**, **higher-moment** (mean-variance-skew-kurtosis) objectives — useful niches, much heavier estimation burden (higher moments are even noisier than the mean).

---

## 8. Constraints

Constraints are not just policy compliance — they are **regularization** that improves out-of-sample behavior (Section 3c). Most are convex and slot into a QP/SOCP solver (`cvxpy`).

| Constraint | Form | Purpose |
|---|---|---|
| Budget | `1'w = 1` (or `=0` for dollar-neutral L/S) | fully invested / neutral |
| Long-only | `w >= 0` | implicit covariance shrinkage; no shorting |
| Box | `lb <= w_i <= ub` | cap single-name concentration |
| Gross leverage | `sum_i |w_i| <= G` | bound total exposure (convex via auxiliary vars) |
| Net exposure | `lb_net <= 1'w <= ub_net` | market-neutrality band |
| Sector/group caps | `A_g w <= cap_g` (or `|A_g w| <= cap_g`) | limit factor/sector tilts |
| Turnover | `sum_i |w_i - w_i^prev| <= T_max` | limit trading per rebalance |
| Cardinality | at most `k` nonzero `w_i` | sparsity (non-convex; needs MIQP/heuristic) |

Note: the L1 constructs (`sum|w_i|`, `sum|w_i - w_i^prev|`) are convex but not differentiable; in `cvxpy` use `cp.norm1(...)`, which the solver reformulates with auxiliary variables — don't hand-code them as `dw_i^2`.

### Transaction-cost-aware optimization

Don't optimize weights then trade naively — optimize **net of expected trading cost** so the optimizer only moves when the alpha justifies the cost (cross-reference `references/transaction-costs.md`, `templates/costs.py`):

```
max_w   w'mu - (lambda/2) w'Sigma w  -  TC(w - w_prev)
```

with a per-asset cost model. Decompose total cost into a *proportional* (L1) part and a *market-impact* part:

```
TC(dw) = sum_i [ c_i*|dw_i|            # half-spread + commission, proportional (L1)
               + impact_i(dw_i) ]      # market impact, convex superlinear
```

- The **L1 (proportional) term induces a no-trade region**: small expected-return changes don't move weights at all (the cost kink dominates the gradient near `dw_i = 0`). This is the optimizer-level analog of rebalancing bands.
- For market impact, `templates/costs.py` uses the empirically standard **square-root law**: per-unit impact ∝ `sqrt(participation)`, so the *total* impact cost of a trade scales roughly like `|dw_i|^{3/2}`. A pure quadratic `dw_i^2` keeps the problem a QP and is conservative for large trades; the `|dw_i|^{3/2}` form is convex but requires an SOCP/conic solver (`cp.power(|dw|, 1.5)` is `cvxpy`-DCP-compliant). Almgren et al. (2005) is the canonical impact reference; calibrate the coefficient and the exponent to your own fills rather than assuming a value.
- Use realistic, per-asset `c_i` (crypto/small-cap wider than large-cap equities; futures via tick value and ADV).

> Pitfall: optimizing gross weights and ignoring costs produces high-turnover "paper" portfolios whose live Sharpe collapses. Always backtest the *cost-net* weights with positions lagged (`pnl_{t+1} = w_t' ret_{t+1} - costs_t`).

---

## 9. Rebalancing

When to move from current weights toward target weights. Trading is costly; *not* trading lets the book drift. The two canonical policies:

- **Calendar rebalancing.** Trade on a fixed schedule (daily/weekly/monthly). Simple, predictable, but trades even when drift is trivial -> wasted cost; and can be crowded around well-known dates.
- **Band (tolerance / no-trade-region) rebalancing.** Rebalance asset `i` only when `|w_i - w_i^target| > band_i` (absolute or relative band), then trade back to target or just to the band edge ("trade-to-edge" minimizes turnover). Adapts to volatility — quiet markets -> few trades, turbulent -> more.

Best practice is **cost-aware**: bands sized so expected alpha capture from rebalancing exceeds expected cost. This is exactly the no-trade region the L1-cost optimizer (Section 8) produces endogenously — band rebalancing is its cheap heuristic approximation.

Practical notes:
- Rebalance frequency must match signal decay: rebalancing daily on a monthly-horizon signal just pays spread for noise.
- Account for **drift between rebalances**: with no trading, realized weights evolve as `w_t,i ∝ w_{t-1,i} * (1 + ret_t,i)` (renormalize so they sum to the invested fraction); risk targets (vol/exposure caps) can be breached intra-period.
- Net flows (deposits/withdrawals) are near-free rebalancing — direct them to under-weight assets first.

---

## Pitfalls (detect / fix)

| Pitfall | Detect | Fix |
|---|---|---|
| **MVO error-maximization** | Tiny `mu` perturbation -> large weight swings; extreme long/short positions; out-of-sample Sharpe << in-sample | Shrink `mu` (Black-Litterman / James-Stein); add long-only & box constraints; or drop `mu` (GMV/ERC/HRP). Resample. |
| **Inverting a near-singular sample covariance** | `np.linalg.cond(Sigma)` >> 1e3; eigenvalues span many orders; `T` not >> `N` (or `T <= N`); unstable weights | Shrink `Sigma` (`constant_correlation_shrinkage` / Ledoit-Wolf / OAS); use HRP (no inversion); never form `inv(Sigma)` — use `solve`/Cholesky and check rank/PD. |
| **Ignoring estimation error** | Backtest uses point-estimate `mu`,`Sigma` and reports the in-sample frontier as if achievable | Treat inputs as random: shrinkage, resampling, robust/Bayesian objectives; report out-of-sample (walk-forward) weights only. |
| **Unconstrained leverage** | `sum|w_i|` blows up (10x+); one estimate drives a huge position | Impose gross/net caps and box bounds; cap single-name and sector exposure. |
| **In-sample optimization of the frontier** | "Optimal" portfolio chosen on the same data used to estimate inputs; spectacular backtest Sharpe | Walk-forward: estimate `mu`,`Sigma` on the train window, hold weights fixed out-of-sample, roll. Purge+embargo for any ML-derived inputs (`templates/validation.py`). |
| **Look-ahead in `mu`/`Sigma` estimation** | Covariance/means computed over a window that includes the rebalance date or future returns; using full-sample stats | Estimate strictly from data `< t`; lag positions vs returns (`pnl_{t+1}=w_t'ret_{t+1}`); verify the estimation window's upper bound is exclusive of `t`. |
| **Overconfident BL views (`Omega` too small)** | BL weights swing far from `w_mkt` on weak views; behaves like raw-`mu` MVO | Calibrate `Omega` to honest confidence; sanity-check weak views barely move weights. |
| **Inverse-vol mistaken for risk parity** | Correlated assets jointly over-weighted; realized RCs unequal | Use ERC (correlation-aware) when correlations are heterogeneous; reserve inverse-vol for near-equicorrelation. |
| **High turnover from naive rebalancing** | Live Sharpe << backtest; cost drag dominates; trading on noise | Cost-aware optimization (L1 term) or band/no-trade-region rebalancing; match rebalance frequency to signal decay. |
| **Realized risk != target (drift)** | Vol/exposure caps breached between rebalances due to drift | Monitor realized drifted weights `w_{t-1}*(1+ret_t)` (renormalized); rebalance on band breach, not just calendar. |

See `templates/validation.py` for covariance shrinkage (`constant_correlation_shrinkage`) and purged/embargoed walk-forward CV (`PurgedKFold`, `CombinatorialPurgedKFold`) used to estimate and validate inputs, and `templates/costs.py` for the transaction-cost models referenced in Section 8. The GMV/tangency/ERC/HRP/BL/constrained-MVO constructors described here are not yet provided as a template — implement them with `numpy`/`scipy`/`cvxpy` following the formulas above.