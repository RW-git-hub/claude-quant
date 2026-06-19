# claude-quant

A Claude Code skill that turns Claude into a disciplined quantitative collaborator
across the full lifecycle — **data → signal/factor research → backtesting →
production code → statistics & risk** — plus specialized markets (options/derivatives,
crypto/DeFi, prediction markets & sports betting).

Its core value is **rigor**: encoding the methodology that separates a real edge
from a backtest fantasy — no look-ahead, no survivorship bias, no overfitting,
realistic costs, correct statistics. Every code template is **execution-verified**
(its self-tests run green), and every reference was adversarially fact-checked.

## Layout

```
claude-quant/
├── SKILL.md                      # Entry point: Iron Laws, router, canonical conventions
├── CHANGELOG.md
├── references/                   # Read on-demand (19 files)
│   ├── playbooks.md              # "What do I actually do" — step-by-step recipes
│   ├── data.md                   # Ingestion, point-in-time, survivorship, rolls, calendars
│   ├── research-backtest.md      # Research loop, backtest mechanics, costs, sizing
│   ├── quant-dev.md              # Code structure, numerical correctness, testing, perf
│   ├── stats-risk.md             # Multiple testing, OOS/CV, metrics, risk, portfolios
│   ├── pitfalls.md               # Fast trap catalog + pre-flight checklist
│   ├── factor-research.md        # Cross-sectional factors: IC, quantiles, Fama-MacBeth
│   ├── transaction-costs.md      # Slippage, market impact, borrow/funding, capacity
│   ├── ml-for-alpha.md           # Labeling, meta-labeling, sample weights, leakage
│   ├── derivatives.md            # Options pricing, greeks, vol surface; FX/rates
│   ├── stat-arb.md               # Cointegration, pairs trading, spread mean-reversion
│   ├── portfolio-optimization.md # MVO, risk parity, HRP, Black-Litterman
│   ├── microstructure.md         # Order book, optimal execution (TWAP/VWAP/Almgren)
│   ├── time-series-regimes.md    # Vol forecasting, HMM/regimes, Kalman, change points
│   ├── robustness.md             # Permutation tests, bootstrap, Reality Check, plateaus
│   ├── crypto-defi.md            # Funding/basis, AMMs, impermanent loss, liquidations
│   ├── risk-management.md        # VaR/ES, VaR backtests, stress, limits
│   ├── prediction-sports-markets.md  # Polymarket + sports: devig, Kelly, CLV, calibration
│   └── live-trading.md           # OMS, reconciliation, kill switches, go-live checklist
├── templates/                    # Correct, self-testing starting points (numpy/pandas)
│   ├── run_all_tests.py          # Runs every template's self-tests (CI gate)
│   ├── metrics.py · validation.py · backtest_skeleton.py · data_loader.py
│   ├── factor_research.py · pairs_trading.py · portfolio.py · regime.py · labeling.py
│   ├── costs.py · execution.py · pretrade_checks.py · risk.py · robustness.py
│   └── options.py · crypto_defi.py · betting_markets.py
└── examples/
    └── end_to_end.py             # Full worked pipeline: data→factor→portfolio→costs→metrics→CV
```

## Install

Copy this directory to your personal skills folder so Claude Code activates it:

```
~/.claude/skills/claude-quant/
```

On this machine: `C:\Users\RossW\.claude\skills\claude-quant\`. After copying, the
skill is discoverable via its trigger phrases (backtest, factor research, Sharpe,
point-in-time, walk-forward, portfolio construction, pairs trading, options greeks,
VaR, Polymarket / sports betting, …).

## Verify

```
python templates/run_all_tests.py    # every template's self-tests (17/17 pass)
python examples/end_to_end.py         # full worked pipeline, self-checked
```

## Scope

All major asset classes (equities, futures/commodities, crypto, FX/rates/options)
plus prediction & sports markets; daily/swing and intraday primary (HFT noted);
modern Python stack. 19 references, 17 self-testing templates, 1 end-to-end example.
