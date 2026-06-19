# claude-quant

A **Claude Code plugin** that turns Claude into a disciplined quantitative collaborator
across the full lifecycle — **data → signal/factor research → backtesting → production
code → statistics & risk** — plus specialized markets (options/derivatives, crypto/DeFi,
prediction markets & sports betting).

Its core value is **rigor**: encoding the methodology that separates a real edge from a
backtest fantasy — no look-ahead, no survivorship bias, no overfitting, realistic costs,
correct statistics. Every code template is **execution-verified** (its self-tests run
green) and every reference was adversarially fact-checked.

## Install (Claude Code)

```
/plugin marketplace add RW-git-hub/claude-quant
/plugin install claude-quant@claude-quant
```

Then invoke it by describing a quant task — "backtest this strategy", "is this overfit?",
"compute the deflated Sharpe", "build a point-in-time data pipeline", "construct a
risk-parity portfolio", "price these options / greeks", "devig this Polymarket line", etc.

## What's inside

This plugin ships the `claude-quant` skill **plus 20 specialized subagents** (listed below). The skill provides:

- **`SKILL.md`** — entry point: Iron Laws, task router, canonical conventions
- **`references/`** — 19 on-demand deep-dives (factor research, transaction costs, ML for
  alpha, derivatives, stat-arb, portfolio optimization, microstructure, regimes,
  robustness, crypto/DeFi, risk management, live trading, prediction/sports markets, …)
- **`templates/`** — 17 correct, self-testing starting points (numpy/pandas)
- **`examples/end_to_end.py`** — a full worked pipeline: data → factor → portfolio →
  costs → metrics → cross-validation

See [`skills/claude-quant/README.md`](skills/claude-quant/README.md) for the full layout
and the verification commands.

## Agents

Twenty specialized subagents you can hand a focused job. Each enforces the same Iron Laws
and cites the skill's references/templates:

| Agent | What it does |
|---|---|
| `backtest-auditor` | Read-only forensic audit of a backtest for look-ahead, survivorship, cost, and multiple-testing bugs |
| `overfitting-detective` | Deflated/Probabilistic Sharpe, permutation & bootstrap tests, plateau-vs-spike, trial-count budget |
| `factor-researcher` | Cross-sectional factor design & evaluation: IC/rank-IC, neutralization, quantile spreads, Fama-MacBeth |
| `portfolio-architect` | Portfolio construction: MVO, risk parity/ERC, HRP, Black-Litterman, shrinkage, vol targeting |
| `risk-manager` | VaR/ES, component risk, factor & scenario stress, limits, VaR backtesting |
| `execution-cost-analyst` | Cost & market-impact modeling, capacity, and TWAP/VWAP/Almgren-Chriss execution |
| `data-integrity-sentinel` | Audits a data pipeline for point-in-time, survivorship, roll, calendar, and NaN errors |
| `options-quant` | Pricing & greeks, implied-vol surfaces, hedging, scenario P&L |
| `stat-arb-strategist` | Cointegration, pairs/baskets, half-life, z-score entries, out-of-sample stability |
| `quant-code-reviewer` | Numerical correctness, hidden leakage in rolling/groupby, reproducibility, performance |
| `ml-alpha-engineer` | ML alpha pipelines: triple-barrier labeling, meta-labeling, sample weights, purged/CPCV, fractional differentiation |
| `regime-detector` | Causal vol forecasting (GARCH/HAR), HMM/Markov regimes, change points, Kalman dynamic beta |
| `crypto-defi-quant` | Perp funding/basis, AMM/LP economics, impermanent loss, liquidations, crypto data hazards |
| `prediction-market-analyst` | Devig, fair odds, Kelly staking, CLV, calibration (Polymarket/Kalshi/sports) |
| `live-trading-engineer` | OMS/EMS state machine, reconciliation, pre-trade risk, kill switches, monitoring |
| `market-microstructure-analyst` | Order book, adverse selection, spread decomposition, VPIN, Avellaneda-Stoikov quoting |
| `rates-fx-quant` | Yield curves, DV01/key-rate risk, carry & roll, FX carry, cross-currency basis |
| `volatility-strategist` | Variance/vol risk premium, VIX term structure, variance swaps, dispersion, vol targeting |
| `performance-attribution-analyst` | Brinson + factor attribution, P&L decomposition, cost/shortfall attribution |
| `alpha-research-strategist` | Front-of-funnel hypothesis generation, prioritization, trial-budget & OOS plan |

Invoke one by describing the task ("audit my backtest", "is this overfit?", "construct a
risk-parity portfolio") or by name.

## Verify

```
python skills/claude-quant/templates/run_all_tests.py   # every template's self-tests
python skills/claude-quant/examples/end_to_end.py        # full worked pipeline, self-checked
```

## License

MIT
