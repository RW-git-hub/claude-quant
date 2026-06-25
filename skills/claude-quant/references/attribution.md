# Performance Attribution

Decomposing a portfolio's return relative to a benchmark into interpretable, *additive* pieces — who/what earned the active return. Runnable, self-tested implementations live in [`templates/attribution.py`](../templates/attribution.py); this doc explains the methods and the gotchas.

Conventions used throughout:
- Returns are **simple** periodic returns (a +1% return is `0.01`). They aggregate **multiplicatively across time** (`1 + R = prod(1 + r_t)`) and **additively across holdings within a period** (`R_p = sum_i w_i r_i`).
- **Active (excess) return** is **arithmetic** by default: `A = R_p - R_b`. The arithmetic active return does *not* compound additively across periods — closing that gap is exactly what Carino linking does.
- Weights `w_p` / `w_b` are **beginning-of-period** holding weights; each sums to ~1 within a period for a fully-invested book.
- All four functions are **ex-post diagnostics over a closed period**. They produce no forward-looking signal (nothing to lag), but their outputs must **not** be fed back into sizing for the *same* period — that would use the period's realized return to set that period's weights (look-ahead). `factor_attribution` fits betas **in-sample** over the whole window, so its loadings are descriptive, not a tradeable forecast.

---

## 1. Brinson-Fachler (single-period sector attribution)

`brinson_fachler(w_p, w_b, r_p, r_b, sectors)` splits the total active return `A = R_p - R_b` into three additive effects per sector `i`:

```
allocation_i  = (w_p_i - w_b_i) * (r_b_i - R_b)     # sector bet, vs TOTAL benchmark
selection_i   =  w_b_i          * (r_p_i - r_b_i)    # stock picking within sector
interaction_i = (w_p_i - w_b_i) * (r_p_i - r_b_i)    # joint allocation x selection
```

where `R_p = sum_i w_p_i r_p_i` and `R_b = sum_i w_b_i r_b_i`. The three effects **sum exactly** to `R_p - R_b` (the function `assert`s this).

**Why Fachler, not plain Brinson-Hood-Beebower (BHB)?** The allocation term uses `(r_b_i - R_b)` — the sector benchmark return *relative to the total benchmark* — instead of BHB's `r_b_i`. So overweighting a sector is credited as allocation skill only if that sector **beat the overall benchmark**, which is the economically correct interpretation. Plain BHB uses `r_b_i` and dumps the difference into a residual.

**The exact identity** (per sector, before summing):

```
w_p r_p - w_b r_b = (w_p - w_b)(r_b - R_b)   [allocation]
                  +  w_b      (r_p - r_b)    [selection]
                  + (w_p - w_b)(r_p - r_b)   [interaction]
                  + (w_p - w_b) R_b          [cross term]
```

Summed over `i`, the cross term `sum_i (w_p_i - w_b_i) R_b = (1 - 1) R_b = 0` because both books are fully invested. So `sum_i (alloc + sel + inter) = R_p - R_b` exactly. **Gotcha:** if `w_p` and `w_b` carry a *different* cash budget (don't each sum to 1), the cross term no longer cancels and the components miss the total — `brinson_fachler` raises on this.

**Multi-period:** Brinson-Fachler is a **single-period** model. Per-period effects must be linked (Carino below), **never naively summed** — the arithmetic sum drifts from the compounded total by a cross-compounding residual.

---

## 2. Carino logarithmic linking (multi-period)

`carino_link(period_active, R_p, R_b)` links single-period arithmetic active returns into contributions that sum **exactly** to the geometrically-compounded total active return.

**The problem.** Single-period active returns `a_t = r_t - b_t` do **not** add up to the geometric multi-period active

```
A = R_p - R_b = (prod_t (1+r_t) - 1) - (prod_t (1+b_t) - 1),
```

because the portfolio and benchmark compound on different bases; `sum_t a_t` leaves a cross-product residual.

**Carino's fix (1999).** Scale each `a_t` by `k_t / k`:

```
k_t = ( ln(1+r_t) - ln(1+b_t) ) / ( r_t - b_t )     # per-period coefficient
k   = ( ln(1+R_p)  - ln(1+R_b)  ) / ( R_p - R_b  )   # total-period coefficient
```

Then

```
sum_t  (k_t / k) * a_t   ==   R_p - R_b     (exactly).
```

**Intuition:** `k_t` maps each arithmetic active into log space (where active returns *do* add); dividing by the total-period `k` maps the summed log-active back to the arithmetic total.

### The 0/0 limit (the part everyone gets wrong)

When `r_t = b_t` the active is zero and `k_t` is a raw `0/0`. The coefficient is the difference quotient of `f(x) = ln(1+x)`, whose derivative is `f'(x) = 1/(1+x)`. So:

```
k_t  ->  1 / (1 + r_t)      as  (r_t - b_t) -> 0     [= 1/(1+b_t) since r_t = b_t]
```

`templates/attribution.py::_carino_coef` returns `1/(1+r)` whenever `|r - b| < 1e-12`, making `k_t` continuous everywhere. The same guard applies to the **total** coefficient `k`: when `R_p = R_b` (zero total active), `k -> 1/(1+R_p)` (finite, non-zero), so the linking does not divide by zero — every linked contribution is simply `0`. Using the raw ratio at these points returns `NaN`; the guard is what makes the function robust.

**Two implementation traps:**
1. `k_t` depends on the return **levels** `r_t, b_t`, not just the difference `a_t`. Passing only `a_t` is the single most common Carino bug. `carino_link` therefore takes a `(T, 2)` array/DataFrame of `[r_t, b_t]`.
2. `R_p`, `R_b` are passed in explicitly so the caller controls compounding — and so the same routine can link **Brinson effects** (whose `r_t/b_t` are sub-portfolio returns) the same way.

---

## 3. Factor attribution (time-series OLS)

`factor_attribution(returns, factor_returns)` regresses the return series on factor returns via OLS (`numpy.linalg.lstsq`, no statsmodels):

```
r_t = alpha + sum_k beta_k f_{k,t} + eps_t
```

and reports average contributions `contribution_k = beta_k * mean(f_k)`, `alpha_contrib = alpha`, plus the in-sample `R^2`. Because OLS with an intercept produces zero-mean residuals, the exact identity holds:

```
mean(r) = alpha + sum_k beta_k * mean(f_k).
```

**Caveat:** betas are fit on the same window they explain — descriptive, not a forecast. For a *tradeable* exposure, estimate beta on a trailing window ending strictly **before** the period being attributed.

---

## 4. Perold implementation shortfall (execution waterfall)

`implementation_shortfall(...)` computes Perold's (1988) paper-vs-reality gap in **currency** terms: the return of a costless "paper" book that transacts the full target at the **decision price**, minus the realized return of the actual (partially) filled book — decomposed so each bucket is a **cost when positive** (for a buy):

```
delay (slippage) = filled   * (arrival_price  - decision_price)   # decision -> market
trading (impact) = filled   * (avg_exec_price - arrival_price)    # impact/timing on fills
opportunity cost = unfilled * (close_price    - decision_price)   # the MISSED trade
commission       = filled   * commission_per_share                # explicit cost
total            = delay + trading + opportunity + commission
```

with `unfilled = target_shares - filled_shares`. **Perold's key insight** is the **opportunity cost**: shares you meant to trade but never got are a real cost, marked here to the period close. For a **sell**, all price differences are negated (a price that rises after a sell decision *helps* you — a negative cost). `total_bps` expresses the shortfall as bps of the paper notional `target_shares * decision_price`.

---

## References
- Brinson, Hood & Beebower (1986); **Brinson & Fachler (1986)**, "Measuring Non-US Equity Portfolio Performance" — the allocation/selection/interaction decomposition.
- **Carino, D. (1999)**, "Combining Attribution Effects Over Time", *Journal of Performance Measurement* — the logarithmic linking coefficient and its 0/0 limit.
- **Perold, A. (1988)**, "The Implementation Shortfall: Paper versus Reality", *Journal of Portfolio Management* — the IS cost waterfall.

See [`templates/attribution.py`](../templates/attribution.py) for the runnable, self-tested code (run `python attribution.py` for the analytic-anchor self-tests).
