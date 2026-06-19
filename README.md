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

This plugin ships a single skill, `claude-quant`:

- **`SKILL.md`** — entry point: Iron Laws, task router, canonical conventions
- **`references/`** — 19 on-demand deep-dives (factor research, transaction costs, ML for
  alpha, derivatives, stat-arb, portfolio optimization, microstructure, regimes,
  robustness, crypto/DeFi, risk management, live trading, prediction/sports markets, …)
- **`templates/`** — 17 correct, self-testing starting points (numpy/pandas)
- **`examples/end_to_end.py`** — a full worked pipeline: data → factor → portfolio →
  costs → metrics → cross-validation

See [`skills/claude-quant/README.md`](skills/claude-quant/README.md) for the full layout
and the verification commands.

## Verify

```
python skills/claude-quant/templates/run_all_tests.py   # every template's self-tests
python skills/claude-quant/examples/end_to_end.py        # full worked pipeline, self-checked
```

## License

MIT
