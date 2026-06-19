"""
end_to_end.py - a small but complete cross-sectional research pipeline that wires
the claude-quant templates together and self-verifies.

Flow:  build a panel  ->  evaluate the factor (IC, quantile spread)
       ->  construct a dollar-neutral long-short book  ->  charge costs
       ->  performance & risk metrics  ->  leak-free CV + covariance shrinkage.

It uses ONLY the shipped templates (no network, no external data) so it runs
anywhere numpy/pandas are installed. Run:  python examples/end_to_end.py

This is a teaching integration example, not a production strategy: the factor's
predictive power is planted into the synthetic data so the plumbing is easy to
verify end to end.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# make the sibling templates/ importable
_TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates")
sys.path.insert(0, _TEMPLATES)

import costs                       # noqa: E402
import factor_research as fr       # noqa: E402
import metrics                     # noqa: E402
import portfolio                   # noqa: E402
import validation                  # noqa: E402


def build_panel(T: int = 500, N: int = 30, alpha: float = 0.01,
                noise: float = 0.02, seed: int = 0):
    """Synthetic panel with a planted cross-sectional signal.

    Returns (factor_df, fwd_ret_df): both indexed by date x asset, point-in-time
    aligned so fwd_ret.loc[t] is the return earned AFTER the factor is observed at t.
    The forward return is alpha * (cross-sectional z-score of the factor) + noise,
    so the factor genuinely (but noisily) ranks forward returns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    assets = [f"A{i:02d}" for i in range(N)]
    factor_df = pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=assets)
    z = fr.cross_sectional_zscore(factor_df)
    fwd = alpha * z.to_numpy() + noise * rng.standard_normal((T, N))
    fwd_ret_df = pd.DataFrame(fwd, index=dates, columns=assets)
    return factor_df, fwd_ret_df


def main() -> int:
    factor_df, fwd_ret_df = build_panel()

    # --- 1. Factor evaluation -------------------------------------------------
    ic = fr.information_coefficient(factor_df, fwd_ret_df, method="spearman")
    ic_stats = fr.ic_summary(ic)
    qr = fr.quantile_returns(factor_df, fwd_ret_df, q=5)
    qstats = fr.quantile_spread_summary(qr)
    print("Factor evaluation")
    print(f"  mean IC      : {ic_stats['mean_ic']:.3f}")
    print(f"  IC t-stat    : {ic_stats['t_stat']:.1f}")
    print(f"  Q5-Q1 spread : {qstats['mean_spread']:.4f}  monotonic={qstats['monotonic']}")

    # --- 2. Dollar-neutral long-short book from the z-scored factor -----------
    z = fr.cross_sectional_zscore(factor_df)
    gross_lev = z.abs().sum(axis=1)                       # per-date gross leverage
    weights = z.div(gross_lev, axis=0).fillna(0.0)        # sum(|w|)=1, sum(w)~0
    gross_ret = (weights * fwd_ret_df).sum(axis=1)        # w_t earns fwd_ret_t

    # --- 3. Costs: charge on turnover ----------------------------------------
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = 0.5 * weights.iloc[0].abs().sum()  # initial ramp
    net_ret = costs.apply_costs(gross_ret, turnover, cost_per_turnover=5e-4)  # 5 bps

    # --- 4. Performance & risk metrics ---------------------------------------
    g_sharpe = metrics.sharpe_ratio(gross_ret)
    n_sharpe = metrics.sharpe_ratio(net_ret)
    print("\nStrategy (long-short, after 5bps/turnover costs)")
    print(f"  gross Sharpe : {g_sharpe:.2f}")
    print(f"  net   Sharpe : {n_sharpe:.2f}")
    print(f"  net ann ret  : {metrics.annualized_return(net_ret):.1%}")
    print(f"  net max DD   : {metrics.max_drawdown(net_ret):.1%}")
    print(f"  avg turnover : {turnover.mean():.2f}/day")

    # --- 5. Leak-free CV split + covariance shrinkage ------------------------
    pk = validation.PurgedKFold(n_splits=5, embargo_pct=0.02, label_horizon=1)
    splits = list(pk.split(np.arange(len(net_ret))))
    overlap = max(np.intersect1d(tr, te).size for tr, te in splits)
    shrink = validation.constant_correlation_shrinkage(fwd_ret_df.to_numpy())
    cov = shrink["covariance"]
    w_mv = portfolio.min_variance_weights(cov)
    print("\nValidation & portfolio")
    print(f"  PurgedKFold  : {len(splits)} splits, max train/test overlap={overlap}")
    print(f"  LW shrinkage : delta={shrink['shrinkage']:.3f}")
    print(f"  min-var wts  : sum={w_mv.sum():.4f}, n={w_mv.size}")

    # --- self-checks ----------------------------------------------------------
    assert ic_stats["mean_ic"] > 0.05, ic_stats
    assert ic_stats["t_stat"] > 3.0, ic_stats
    assert qstats["mean_spread"] > 0.0, qstats
    assert np.isfinite(g_sharpe) and np.isfinite(n_sharpe)
    assert g_sharpe > n_sharpe, "costs must reduce net Sharpe"
    assert overlap == 0, "purged CV train/test must not overlap"
    assert abs(w_mv.sum() - 1.0) < 1e-8
    print("\nend_to_end.py: pipeline ran and all self-checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
