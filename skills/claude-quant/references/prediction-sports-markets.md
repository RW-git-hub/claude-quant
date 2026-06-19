# Prediction Markets & Sports Betting Quant

Treat every quoted price/odds as a *market-implied probability*, not the truth. The job is to (1) recover a clean probability from a vig-laden quote, (2) form your own probability, (3) bet only when your edge survives costs and is sized to avoid ruin, and (4) backtest it without peeking at information you would not have had at bet time. The single most damning leak in this domain is using the **closing line** to decide a bet — it is the strongest *ex-post* predictor of edge and almost always available in historical data, so it is trivially easy to leak.

Conventions used throughout: probabilities are in `[0,1]`; decimal odds `d` (a.k.a. European) pay `d` per 1 staked **including** stake; `b = d - 1` is net payout per unit; `p` is your model's win probability, `q = 1 - p`. Positions/bets are decided strictly on the information set available *before* the event (pre-event prices).

> **Template status (read first).** `templates/betting_markets.py` ships the core, self-tested utilities for this reference: odds conversions, devigging (multiplicative / power / Shin), expected value, Kelly sizing, Brier score, log loss, and closing-line value. `templates/calibration.py` (numpy-only, self-tested) ships the full calibration toolkit referenced in §15: reliability curves, Expected/Max Calibration Error, the Murphy Brier decomposition, and **working** Platt-scaling and isotonic (PAVA) recalibrators — no longer pseudocode, and no sklearn dependency. `templates/metrics.py` covers return/Sharpe/Sortino/drawdown/VaR-style performance metrics only (no calibration utilities). The bankroll-simulation snippet in §17 below is illustrative pseudocode to adapt. For time-series CV reuse `templates/validation.py` (`PurgedKFold` / `CombinatorialPurgedKFold`, purge + embargo) — and use it to carve the held-out calibration fold (§15).

---

## PART A — Prediction Markets (Polymarket etc.)

### 1. The instrument

A binary outcome contract pays **1 unit if it resolves YES, 0 if NO**. Its price `P ∈ [0,1]` is therefore the market-implied probability of YES directly — no conversion needed. Categorical markets (e.g. "who wins the election") are a set of mutually-exclusive YES contracts, one per outcome.

- Buying YES at price `P` costs `P`, returns `1` on YES → profit `1 - P`, loss `P` on NO. Expected profit per unit = `p_true - P`.
- YES at `P` is economically identical to NO at `1 - P`. On a CLOB, selling YES = buying NO (you post collateral for the complementary leg).
- Max loss is bounded (you can't lose more than your stake), max gain is bounded. This bounded payoff is why Kelly sizing (Part B §13) maps cleanly.

### 2. Mechanics

**CLOB vs automated market maker.** Two designs:
- **CLOB (central limit order book):** discrete bids/asks, you cross the spread as taker or post as maker. Polymarket uses an off-chain CLOB with on-chain settlement. You face a real bid-ask spread and finite depth.
- **AMM / LMSR (Hanson's Logarithmic Market Scoring Rule):** a market maker quotes a price as a deterministic function of net shares outstanding. LMSR cost function `C(q) = b · ln(Σ_i exp(q_i / b))`; price of outcome `i` is `p_i = exp(q_i/b) / Σ_j exp(q_j/b)` (a softmax — prices automatically sum to 1). The liquidity parameter `b` controls depth and the maximum subsidy/loss the maker can take (worst-case loss is `b · ln n` for `n` equally-likely outcomes). Larger `b` = deeper book, smaller price impact per share. Trading moves the price along this curve, so your *average* fill is worse than the pre-trade marginal price (integrate `C`). Note: Polymarket itself is a CLOB, not an LMSR AMM — LMSR is included here as the canonical AMM reference, not as a description of Polymarket.

**Polymarket specifics:**
- Collateral is **USDC** on Polygon; positions are ERC-1155 conditional tokens (one per outcome of a condition).
- Resolution is via **UMA's optimistic oracle**: a proposer asserts an outcome with a bond; if undisputed within the challenge window it settles; if disputed it escalates to UMA's DVM (Data Verification Mechanism) token-holder vote. This introduces *resolution latency* and *resolution risk* (§5).
- Fees: Polymarket has historically charged 0% trading fee on most markets, but **do not hardcode this** — check current terms and the specific market. You always pay **gas** for on-chain actions (deposit/withdraw, claim/redeem) and you cross the **spread** as a taker. Maker/taker distinction matters on the CLOB: makers may get rebates or zero fee, takers pay the spread.
- Settlement: winning shares redeem 1 USDC each *after* resolution finalizes — capital is locked until then.

### 3. Price as probability — and its biases

- **Conversion:** YES price `P` ↔ implied prob `P`; ↔ fair decimal odds `d = 1/P`; ↔ American odds (§8).
- **Favorite–longshot bias:** empirically, longshots (low `P`) are *overpriced* and favorites (high `P`) are slightly *underpriced* relative to realized frequency. The well-calibrated relationship is not the 45° line; a reliability diagram (§15) typically shows realized frequency *below* the diagonal at the low end (longshots resolve YES less often than priced) and *above* it at the high end. This is the same bias seen in racetrack/sportsbook markets. (It is empirically robust but not universal — some prediction markets and some sports show muted or even reversed bias, so measure it on *your* data rather than assuming it.)
- **Calibration is the core question:** does the set of contracts priced at ~0.30 actually resolve YES ~30% of the time? Build reliability diagrams from historical Polymarket resolutions; if the market is miscalibrated in a *stable, out-of-sample* way, that's a candidate signal. If your *own* model is miscalibrated, recalibrate (§15) before sizing.

### 4. Arbitrage

For a complete set of mutually-exclusive, collectively-exhaustive outcomes the **fair prices should sum to 1**; the *executable* prices will not, and that gap is what you trade.
- If you can **buy every YES leg** (pay the asks) for `Σ asks < 1`, you lock a guaranteed 1-unit payout for less than 1 → buy-side edge.
- If you can **sell every YES leg** (hit the bids) for `Σ bids > 1`, you collect more than the 1-unit liability you owe → sell-side edge.
- (Equivalently: buying all YES legs is the same as selling all NO legs at `1 − ask`. State the arb in terms of the prices you actually transact at — asks when buying, bids when selling — not a single mid sum.)
- **Cross-market:** logically-linked markets (e.g. "Candidate X wins" vs "Party of X wins") must obey probability inequalities; violations are arbitrageable.
- **Cross-venue:** Polymarket vs a sportsbook / Betfair on the same event. Devig the sportsbook side first (§9) before comparing, and compare *executable* prices (taker price + fees), not mid.

**Frictions that kill paper arbs:** gas on every leg, fees, the spread you actually cross, **capital lockup until resolution** (your USDC is dead money for weeks/months — discount the arb by the opportunity cost), and **resolution risk** (the "free" arb isn't free if one venue resolves differently or the oracle is disputed). Always size the arb net of all four.

### 5. Time value & resolution

- As the event approaches, price converges to the eventual `0` or `1`; volatility of the price collapses near resolution.
- **Hold-to-resolution vs trade-out:** holding earns `1 - P` (or `P`) but locks capital and bears full resolution risk; trading out earlier realizes mark-to-market PnL, frees capital, and offloads tail/oracle risk — at the cost of crossing the spread again.
- **Settlement / oracle-dispute risk:** UMA disputes can delay settlement and, rarely, resolve against the "obvious" outcome. Ambiguously-worded markets (the resolution *criteria*, not the event) are the main hazard — read the rubric, not the headline. Price in a haircut for dispute probability on contentious markets.

### 6. Liquidity, depth, slippage, limits

- Quote depth is thin on most markets. Compute **slippage** by walking the book: for a buy of `S` shares, average fill = `(Σ over levels price·qty) / S`, not the top-of-book. For LMSR/AMM, average fill = `(C(q+Δ) - C(q)) / Δ`.
- Position limits and self-imposed concentration limits matter; thin markets mean your own order *is* the market impact.
- Build cost into the backtest as **fill = best available price walked through real depth + fees + gas**, never at mid.

### 7. Data

- **REST/CLOB API:** current order book, midpoint, trades, market metadata, resolution status. (Verify current endpoints/schemas against Polymarket's live docs — API surfaces change.)
- **GraphQL subgraph:** historical on-chain trades, positions, market creation/resolution events — good for point-in-time reconstruction of settled trades.
- For point-in-time correctness, store **timestamped snapshots** of the book; reconstructing "the price at time `t`" from trade history alone misses resting liquidity and the spread. Record `(timestamp, best_bid, best_ask, depth, mid)` so backtests can charge the real spread. On-chain trade history captures executions but not the resting book, so snapshots are mandatory for honest cost modeling.

---

## PART B — Sports Betting

### 8. Odds formats and conversion

| Format | Example | Implied prob (raw) | To decimal `d` |
|---|---|---|---|
| Decimal | `2.50` | `1/2.50 = 0.40` | — |
| American (+) | `+150` | `100/(150+100) = 0.40` | `1 + A/100` |
| American (−) | `−200` | `(-A)/((-A)+100) = 0.667` | `1 + 100/(-A)` |
| Fractional | `3/2` | `den/(num+den) = 0.40` | `1 + num/den` |

Raw implied prob = `1/d`. Net payout per unit `b = d - 1`. These raw probs across all outcomes **sum to >1** — that excess is the vig.

### 9. The vig / overround and DEVIGGING

Let raw implied probs be `r_i = 1/d_i`. The **overround** (booker's margin) is `O = Σ_i r_i - 1 > 0`. The raw `r_i` are *not* probabilities; you must remove the vig to recover the market's fair estimate `p_i`.

**Multiplicative (normalize):** `p_i = r_i / Σ_j r_j`. Simple, but distributes the margin proportionally to each leg's raw prob — relative to methods that load more margin onto longshots, it *leaves favorite–longshot bias largely intact* (it under-deflates longshots and over-deflates favorites versus Shin/power).

**Shin's method:** models a fraction `z` of bets coming from **insiders** (informed money the book defends against). With `R = Σ_j r_j`,
`p_i = ( sqrt(z² + 4(1-z)·r_i²/R) − z ) / (2(1-z))`
where `z` is found by a 1-D root solve so that `Σ_i p_i = 1`. Shin attributes more of the margin to longshots, which matches observed favorite–longshot bias better than naive normalization.

**Power method:** find exponent `k` such that `Σ_i r_i^k = 1`, then `p_i = r_i^k`. Because each `r_i < 1`, an overround (`Σ r_i > 1`) requires `k > 1` to pull the sum down to 1. Also bends the correction non-linearly across the probability range and typically beats multiplicative on calibration.

**Additive:** subtract the margin equally, `p_i = r_i − O/n` (this sums to 1 since `Σ r_i − n·(O/n) = Σ r_i − O = 1`). Crude; can produce negatives in lopsided markets. Avoid except as a baseline.

**Wisdom-of-the-crowd / weighted:** combine devigged probabilities from *several* books (weighting sharp books more) for a better consensus estimate than any single book.

> Rule of thumb: for **two-way** markets all methods agree closely; for **multi-way** markets (e.g. outright winner of a 20-runner field) the method choice materially changes longshot probabilities. Prefer **Shin** or **power** when longshots matter; validate the choice on *your* sport with a reliability diagram (§15).

### 10. Edge and CLOSING LINE VALUE (CLV)

The closing line (final pre-event price across sharp books) is the market's best aggregate estimate. **Consistently beating the closing line is one of the strongest empirical predictors of long-run profitability** — and faster to measure than realized PnL, which is far noisier due to outcome variance.

**Measuring CLV** (devig both your bet odds and the closing odds first, on the *same* side):
- **Probability CLV:** `p_close_devig − p_your_entry_devig` (positive = the closing consensus implied a higher win prob than the price you got, i.e. you got a better price).
- **Odds CLV / "beat the close":** `d_your_entry / d_close − 1` for the same side (positive = you locked a higher payout than close).
- Track CLV per bet and aggregate; a positive mean CLV with statistical significance is your earliest evidence of edge, well before PnL confirms it. (Caveat: CLV measured against *soft* books, or against your own book if you are large enough to move it, is not the same as beating the sharp consensus close — benchmark against the sharpest available close.)

**Critical:** CLV is a *post-hoc diagnostic only*. It uses the closing line, which is **not** in your information set at bet time. Using the closing line to *select* bets is look-ahead (§16, pitfalls).

### 11. EV, bet selection, books

- **Expected value** of a unit stake at decimal `d` with your prob `p`: `EV = p·(d-1) − (1-p) = p·d − 1`. Bet only if `EV > 0`, i.e. `p > 1/d`.
- **Line shopping:** the same bet is offered at different odds across books; always take the best available `d`. A few cents of `d` is often the entire edge.
- **Sharp vs soft books:** sharp books (low margin, high limits, move on smart money) define the "true" line; soft/recreational books (high margin, low limits, slow to move, restrict winners) are where +EV bets live but get limited. Your fair-value estimate should lean on sharp/consensus devigged lines.
- **Exchanges (Betfair):** you bet *against other users*, can **back or lay**, and can **make markets** (post offers). Commission is charged on **net winnings per market** (typically a few %), not on stake. EV must net the commission: effective payout on a winning back at `d` with commission rate `c` is `1 + (d−1)(1−c)`. (Real commission is computed on net market profit, sometimes with a points/discount scheme, so the per-bet `(d−1)(1−c)` is an approximation — model your actual account's terms.)

### 12. Modeling

- **Elo / power ratings:** maintain a rating per team; expected score `E_A = 1 / (1 + 10^((R_B − R_A)/400))`; update `R_A ← R_A + K·(S_A − E_A)`. Add home-field advantage and margin-of-victory multipliers. Cheap, robust baseline.
- **Poisson** for soccer scorelines: model home/away goals as independent Poisson with means `λ_home, λ_away` from team attack/defense strengths; `P(score = (i,j)) = Pois(i;λ_home)·Pois(j;λ_away)`. Sum the score matrix to get 1X2 / over-under probabilities.
- **Dixon–Coles:** corrects the independent-Poisson model for the empirical **dependence in low scores** (0-0, 1-0, 0-1, 1-1) via a correction factor `τ`, and adds **time-decay weighting** of past matches. Better fit to draws and low-scoring games than plain Poisson.
- **Regression/ML** for win probability: logistic regression / gradient boosting on team/player features. Whatever the model, **its output probabilities must be calibrated** (§15) before they go into edge/Kelly — a sharp-but-miscalibrated classifier will systematically mis-size bets.
- **Edge per bet:** `edge = p·d − 1` using your *calibrated* `p` against the *best available* `d`.

### 13. Bankroll & Kelly

The Kelly fraction maximizes long-run log-growth. For a single binary bet at decimal odds `d` (`b = d−1`), win prob `p`, `q = 1−p`:

```
f* = (b·p − q) / b      # fraction of bankroll to stake
   = p − q/b
   = (p·d − 1) / (d − 1) # = edge / b
```

(All three forms are algebraically identical: `b·p − q = (d−1)p − (1−p) = p·d − 1`.)

- `f* ≤ 0` ⇒ **no bet** (no edge). Never bet a negative-edge position; clamp the stake to 0.
- **Fractional Kelly (½ or ¼):** standard practice. Full Kelly assumes your `p` is *exactly right*; any estimation error makes full Kelly over-bet, which sharply raises variance and drawdown and can turn a true edge growth-negative. Fractional Kelly trades a little growth for a large reduction in variance and drawdown, and is robust to `p` error. Default to ¼–½ Kelly.
- **Simultaneous / correlated bets:** Kelly for multiple simultaneous bets is solved *jointly* (allocate fractions so total expected log-growth is maximized; with correlation you solve the joint optimization, not per-bet). Treating correlated bets as independent and summing per-bet Kelly fractions **over-stakes** badly — the classic parlay trap (§ pitfalls).
- **Risk of ruin / variance:** even +EV fractional-Kelly betting has large drawdowns; simulate the bankroll path (§17) to see the drawdown distribution and ruin probability before committing real money.

### 14. In-play / live

- Prices update tick-by-tick; latency is both the edge and the risk. Your model must run faster than the market and account for stale quotes.
- Suspensions (goal/wicket/timeout) freeze the book; orders may be cancelled or filled at the *post-event* price — model this slippage.
- In-play markets have **wider margins** and thinner depth; demand a larger edge threshold than pre-match.

---

## PART C — Shared Machinery

### 15. Probability calibration

Calibration is the **bet-sizing** question, distinct from ranking. A model can rank bets perfectly (high AUC/IC) yet be badly *miscalibrated*: if the contracts it calls "70%" actually resolve YES 85% of the time, then `EV = p·d − 1` and the Kelly fraction `(p·d−1)/(d−1)` are computed off a wrong `p`, so every stake is mis-sized and the EV you think you are harvesting is corrupted even though the ordering is fine. **Calibrate `p` before it ever enters EV or Kelly (§11, §13).** All utilities below are in `templates/calibration.py` (numpy-only, self-tested) — import them; do not re-implement.

**Scoring your probability forecasts** against binary outcomes `y ∈ {0,1}`:
- **Brier score:** `BS = mean((p − y)²)`. Lower is better. `calibration.brier_score(p, y)`.
- **Log loss:** `−mean(y·ln p + (1−y)·ln(1−p))`. Punishes confident wrong calls harshly; clip `p` to `[ε, 1−ε]` to avoid `ln(0)`. `calibration.log_loss(p, y, eps=1e-15)`.
- **Murphy decomposition** of the Brier score: `BS = reliability − resolution + uncertainty`, where `reliability = (1/N)Σ_k n_k(f_k − o_k)²` (lower = better calibrated; 0 = perfectly calibrated), `resolution = (1/N)Σ_k n_k(o_k − ō)²` (higher = bins genuinely separate outcomes), and `uncertainty = ō(1 − ō)` (irreducible base-rate variance, `ō = mean(y)`). `calibration.brier_decomposition(p, y, n_bins, strategy)` returns all three plus the raw and binned Brier; the identity reconstructs the raw Brier exactly when each bin holds a single distinct forecast value. Reliability isolates the *calibration* part of a poor Brier from the *sharpness* part.

**Reliability diagram & calibration error:**
- **Reliability curve:** `calibration.reliability_curve(p, y, n_bins=10, strategy='uniform'|'quantile')` bins predictions and returns per-bin mean predicted prob vs observed frequency (and counts). Plot mean predicted (x) vs observed (y); the 45° line is perfect calibration. Realized *below* predicted = overconfident — and at the extremes this is exactly the favorite–longshot bias signature in market prices (§3). Use `strategy='quantile'` (equal-count bins) when predictions cluster (e.g. near 0.5); `'uniform'` is the classic equal-width diagram.
- **Expected Calibration Error (ECE):** `Σ_k (n_k/N)·|o_k − f_k|`, the count-weighted mean gap between confidence and accuracy. `calibration.expected_calibration_error(p, y, n_bins, strategy)`. **Max Calibration Error (MCE):** the worst single-bin gap, `calibration.max_calibration_error(...)`. Both depend on `n_bins`/`strategy` — always report the binning alongside the number.

**Recalibration** (each fits a transform you apply to new probabilities):
- **Platt scaling:** `transform = calibration.platt_scale(p_cal, y_cal)` fits `σ(a·logit(p) + b)` by hand-rolled Newton/IRLS logistic regression (no sklearn). Parametric (2 params), robust on small calibration sets, but can only apply a *monotone logistic squash* of the logit — it cannot fix non-monotone miscalibration. `transform.coef_` exposes `(a, b)`. (For an over-confident model whose reported logits are inflated, the fit recovers `a < 1`, shrinking the logit back toward calibration — the template's self-test asserts exactly this.)
- **Isotonic regression:** `transform = calibration.isotonic_fit(p_cal, y_cal)` fits a non-decreasing map via Pool-Adjacent-Violators (PAVA). Non-parametric and more flexible (can fix shape, not just slope), but needs more data and overfits small samples (steps fit noise). Prefer Platt when the calibration fold is small.
- Both return a closure `transform(p_new) → calibrated probs`. **Fit on a held-out calibration fold that is DISJOINT from both the data your model trained on and the fold you evaluate/bet on.** Recalibrating on the evaluation set makes any miscalibration vanish trivially — that is look-ahead (Iron Law: out-of-sample is sacred), and the post-calibration Brier/ECE you would quote are inflated. For time-dependent betting data, carve the calibration fold with **purge + embargo** (`templates/validation.py`) so an overlapping information window can't bleed event outcomes into the fit.

> Implementation note: import these from `templates/calibration.py` (dependency-free numpy). `templates/metrics.py` is performance metrics only (Sharpe/Sortino/drawdown/VaR — no calibration). You no longer need sklearn (`sklearn.calibration` / `sklearn.isotonic`) for any of this; the template's self-tests verify that the overconfident-generator's ECE and log-loss both drop out-of-sample after Platt and isotonic, and that the isotonic map is non-decreasing.

### 16. Leak-free bet backtesting

The discipline that makes or breaks this entire domain:

1. **Decide on pre-event information only.** Bet selection and sizing must use *only* prices/odds and features timestamped strictly before the bet is placed (which is before event start). This mirrors the house rule `pnl_t = pos.shift(1)·ret_t`: the decision is lagged relative to the outcome it earns.
2. **Never use the closing line to decide.** The closing line is computed *after* most betting; use it **only** to measure CLV *after* the fact (§10). If your backtest's bet filter references the close, it's look-ahead — delete it.
3. **Settle at the realized outcome,** paying out `b` on wins and `−1` on losses (per unit stake), netting exchange commission on wins.
4. **Charge the vig / spread / commission** on every bet at the price you would actually have transacted (best *available* pre-event odds, not the mid, not the close).
5. **Survivorship-correct the event set.** Include every event that *was scheduled and bettable at decision time*, not only completed/non-void events selected with hindsight. Voided/postponed games, cancelled markets, and de-listed contracts must be handled by the rule known at bet time (typically stake refunded), not dropped retroactively.
6. **Time-series CV must purge + embargo** around each event so leakage through overlapping information (injury news, line moves spanning the boundary) can't bleed train into test. Reuse `templates/validation.py` (`PurgedKFold` / `CombinatorialPurgedKFold`) — set `label_horizon` to cover the information window an event's features and outcome span.

Anchor your CLV diagnostic: a leak-free strategy should show **positive average CLV** *and* positive PnL; positive PnL with negative CLV is usually variance or a hidden leak — investigate before trusting it. (And implausibly high CLV/PnL with near-zero variance is the signature of a closing-line leak — audit step 2.)

### 17. Risk of ruin & bankroll simulation

Simulate forward paths to characterize tail risk rather than trusting a point EV:

```python
# Monte Carlo bankroll path under fractional Kelly
import numpy as np

def simulate_ruin(p, d, frac, n_bets, n_paths=10_000,
                  ruin_level=0.0, seed=0):
    rng = np.random.default_rng(seed)
    b = d - 1.0
    f = frac * (p * d - 1.0) / b          # fractional-Kelly stake fraction
    f = min(max(f, 0.0), 1.0)             # never bet <0 or >100% of bankroll
    bank = np.ones(n_paths)
    ruined = np.zeros(n_paths, dtype=bool)
    for _ in range(n_bets):
        live = ~ruined                    # ruined paths stop betting
        win = rng.random(n_paths) < p
        mult = np.where(win, 1.0 + f * b, 1.0 - f)
        bank = np.where(live, bank * mult, bank)
        ruined |= bank <= ruin_level + 1e-9
    log_bank = np.log(np.clip(bank, 1e-12, None))   # guard log(0)
    return {
        "p_ruin": ruined.mean(),
        "median_bank": float(np.median(bank)),
        "p05_bank": float(np.quantile(bank, 0.05)),
        "growth_rate": float(np.mean(log_bank) / n_bets),
    }
```

Report ruin probability, median terminal bankroll, the 5th-percentile path, and per-bet log-growth — full Kelly will show dramatically fatter left tails than ¼-Kelly for the same edge, which is the whole argument for fractional sizing.

> Caveats on this toy model: it assumes a *single* repeated bet with fixed `p`, `d`, and a fixed bankroll fraction `f` (no simultaneous/correlated bets, no varying odds, no minimum stake). With a hard `ruin_level=0`, multiplicative `1−f` betting never reaches exactly 0, so `p_ruin` will be 0 unless `ruin_level>0` — set `ruin_level` to a realistic give-up threshold (e.g. 0.2 of starting bankroll). For real risk-of-ruin, resample from your *actual* bet log (bootstrap historical `p`/`d`/correlation), not i.i.d. draws from one edge.

---

## Pitfalls (detect / fix)

| Pitfall | Detect | Fix |
|---|---|---|
| Treating raw odds/price as true probability | Implied probs across a market sum to >1; no devig step in code | Devig first (§9): Shin or power for multi-way / longshots; normalize for two-way; never bet on raw `1/d` |
| Ignoring the vig entirely | EV computed from `1/d` without margin removal; "edges" everywhere | Subtract overround; compute EV from devigged consensus vs best *available* price |
| Look-ahead via the closing line | Bet filter or sizing references `close`/`closing_odds`; backtest PnL implausibly smooth/high | Decide only on pre-event prices; use close *solely* to measure CLV after the fact (§10, §16) |
| Over-betting (full Kelly) | Stake = full `f*`; huge simulated drawdowns / ruin | Use ¼–½ Kelly; clamp `f*` to `[0,1]`; size on *calibrated* `p` (§13) |
| Correlated parlays priced as independent | Parlay price = product of legs; same-game / same-driver legs | Model joint probability; solve Kelly *jointly*; never multiply correlated legs (§13) |
| Ignoring Polymarket resolution/settlement risk & gas | Arb looks free; capital-lockup and gas not in PnL | Haircut for dispute probability; charge gas per leg; discount for USDC lockup until resolution (§4, §5) |
| Survivorship in event set | Only completed / non-void games in backtest; voids dropped with hindsight | Include every event bettable at decision time; apply void rule known at bet time (§16) |
| Miscalibrated model probabilities | Reliability diagram bows off the diagonal; ECE/MCE large; Brier reliability term high | Recalibrate with `calibration.platt_scale` / `calibration.isotonic_fit` on a held-out (purged) fold before feeding `p` into edge/Kelly (§15) |
| Recalibrating on the evaluation/bet fold | Post-calibration ECE/Brier implausibly low; gap reappears on a truly held-out fold | Fit Platt/isotonic on a calibration fold disjoint from training AND evaluation; purge+embargo for time series (§15) |
| Backtesting at mid, not executable price | Fills at midpoint; no spread/slippage/depth walk | Walk the book / LMSR cost integral; charge spread, fees, commission, gas (§6, §11) |
| Single-book fair value | Edge measured vs one soft book's line | Build devigged consensus across sharp books (wisdom-of-crowd) as fair value (§9, §11) |
| Importing nonexistent template code | calibration helpers assumed in `metrics.py`; `from sklearn...` assumed available | Use `templates/betting_markets.py` (devig/Kelly/CLV/Brier/log-loss) and `templates/calibration.py` (reliability/ECE/Brier-decomp/Platt/isotonic, numpy-only — no sklearn needed); reuse `validation.py` for purge/embargo |

---

**Templates that actually exist and apply here:** `templates/betting_markets.py` (odds conversions, devig multiplicative/Shin/power, expected value, Kelly & fractional Kelly, Brier/log-loss, CLV); `templates/calibration.py` (reliability curves, ECE/MCE, Murphy Brier decomposition, Platt scaling, isotonic/PAVA — numpy-only, no sklearn — §15); `templates/validation.py` (`PurgedKFold` / `CombinatorialPurgedKFold` for purge + embargo, §16, and for carving the calibration fold in §15); `templates/metrics.py` (performance metrics — Sharpe, Sortino, drawdown, VaR, deflated/probabilistic Sharpe; **not** calibration). See also `references/pitfalls.md` and `references/stats-risk.md` for the general look-ahead, survivorship, and overfitting framing that applies here verbatim.
