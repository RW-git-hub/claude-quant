---
name: walk-forward-validation
description: >-
  Use when asked to "walk-forward", "cross-validate a strategy", "purged k-fold / CPCV", "out-of-
  sample / OOS test", "is my CV leaking", "set the embargo", or "why does my out-of-sample / walk-
  forward Sharpe look too good (is the CV leaking)?" — the quick OOS-validation playbook for leak-
  free time-series CV. For whether a result is a multiple-testing artifact after many trials
  (deflated Sharpe, permutation, PBO verdict) use the overfitting-detective agent; for a leak-free
  CV/CPCV that is one stage of an ML labeling pipeline use ml-alpha-engineer; the broad claude-
  quant skill is the full router.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

Validate a strategy out-of-sample with zero leakage: pick the scheme, embargo correctly, lock the test set, and count the trials.

## Do this now
1. **Open the template.** `skills/claude-quant/templates/validation.py` ships `PurgedKFold` and `CombinatorialPurgedKFold` (CPCV) plus the `_purged_train_mask` helper. Use them as-is; never hand-roll CV or call `sklearn.KFold`.
2. **Pick the scheme.** Walk-forward = chronological, time-ordered `PurgedKFold` splits that mimic live retraining (do NOT shuffle). Want many backtest paths / a Sharpe *distribution* and a PBO estimate → `CombinatorialPurgedKFold`. There is no separate WalkForward class — walk-forward is just `PurgedKFold` consumed in time order.
3. **Set the embargo.** Pass `label_horizon` = your forward label window in bars (triple-barrier horizons in `templates/labeling.py`). The template purges `label_horizon` on BOTH sides and embargoes `embargo_pct*n` bars after each test block — size `embargo_pct` so embargo-in-bars ≥ label horizon + any feature lookback. Verify against its `__main__` self-test.
4. **Lock the final test set.** Hold out one block touched exactly once, at the very end. Any tweak after seeing it is leakage.
5. **Track the trial budget.** Log every config tried; pass the count and `trial_sharpe_std` into `deflated_sharpe_ratio` in `templates/metrics.py`.
6. **Score per fold** with `templates/metrics.py`; report the fold distribution, not just the mean.

## References
`references/robustness.md` (CV schemes, CPCV, PBO, trial budget) and `references/stats-risk.md` (deflation, multiple-testing).

## Gotchas
- `shuffle=True` / random splits silently leak on time series — always time-ordered.
- Overlapping labels need BOTH purge and embargo; embargo alone is insufficient.
- Embargo shorter than the label horizon leaks adjacent labels.
- Reusing the final test set counts as another trial — it inflates Sharpe.

## Expected output
Per-fold metrics + distribution, chosen scheme with `embargo_pct`/`label_horizon`, the trial count, and a deflated Sharpe — all from the cited templates. (Iron Laws 1, 4, 5.)
