# stratval — a leakage-free validation harness for systematic trading-strategy research

A small, dependency-light Python toolkit for the part of quantitative research that is
easiest to get wrong: deciding, *honestly*, whether a strategy signal is real. Financial
decision data is small, noisy, and autocorrelated — the conditions under which spurious
"discoveries" are the default outcome. `stratval` packages the practices that separate a real
effect from an artifact, and demonstrates them end-to-end on **synthetic data with a known
ground truth**.

This is the methodology-and-engineering distillation of a long-running personal
quantitative-finance research program — one that has mostly produced **nulls**: candidate
signals that looked real in-sample and did not survive honest validation. That is the point.
The deliverable is the discipline that lets you *trust a negative result* rather than
manufacture an edge — so this repo makes **no alpha claims** and ships no real data: every
example runs on a simulator (`stratval.generate_rebalance_panel`) that plants a known signal
and a known confound so the harness can be shown to recover the one and reject the other.

## What's in the box

| Concern | Module | What it gives you |
|---|---|---|
| **Gate-first protocol** | [`gate`](src/stratval/gate.py) | pre-register the ship/reject rule as a frozen object; no goalpost moves |
| **Event-clustered bootstrap** | [`clustered_bootstrap`](src/stratval/clustered_bootstrap.py) | CIs that reflect independent *events*, not correlated rows; paired delta vs a baseline |
| **Within-event metric** | [`within_event`](src/stratval/within_event.py) | period-stripped AUC that kills regime-clock confounds |
| **Walk-forward + embargo** | [`walkforward`](src/stratval/walkforward.py) | expanding OOS folds; an embargo whose residual leakage is *measured* |
| **Null models** | [`nulls`](src/stratval/nulls.py) | within-event shuffle, anti-signal, permutation — gates that can fail |
| **Robustness suite** | [`robustness`](src/stratval/robustness.py) | leave-one-block-out, power-injection, negative controls |
| **Multiple-comparison control** | [`multiple_comparison`](src/stratval/multiple_comparison.py) | Bonferroni (FWER) and Benjamini-Hochberg (FDR) |
| **Backtest-overfitting control** | [`deflated_sharpe`](src/stratval/deflated_sharpe.py) | Probabilistic & **Deflated Sharpe Ratio** (deflate the winner for N trials) + **PBO** via CSCV |
| **Synthetic generator** | [`synth`](src/stratval/synth.py) | a stylized rebalancing process (planted signal + confound) and a trial-search matrix |

The design philosophy is written up in **[docs/methodology.md](docs/methodology.md)** —
eight lessons, each tied to the failure it prevents, with a closing map of where each piece
sits in the quant-validation literature (López de Prado purging/embargo; the Deflated Sharpe
Ratio and PBO; White's Reality Check / Hansen's SPA).

## Quickstart

```bash
pip install -r requirements.txt        # numpy, pandas, scikit-learn, scipy
python examples/demo_harvest_signal.py
```

```python
from stratval import generate_rebalance_panel, within_event_auc, clustered_auc

panel = generate_rebalance_panel()          # synthetic decisions: add/trim per name per event

# The confound trap, in two lines:
print(clustered_auc(panel, "regime_clock")["point"])   # ~0.80  pooled "skill" (an artifact)
print(within_event_auc(panel, "regime_clock")["point"])# ~0.50  collapses within-event
```

## The demo, in one screen

`examples/demo_harvest_signal.py` walks the full pipeline on synthetic data and shows:

1. **The confound trap** — a planted regime clock scores AUC ≈ 0.80 *pooled* yet ≈ 0.50
   *within-event*, while the real signal holds at ≈ 0.71 on both. The within-event metric is
   what tells the truth.
2. **A pre-registered gate** passing on the real signal (`Gate.evaluate`).
3. **A paired clustered test** showing the signal beats its collinear momentum cousin.
4. **Negative controls** — the gate correctly fails on a shuffled score and inverts on the
   negated one.
5. **Robustness** — leave-one-month-out and a power-injection curve.
6. **Walk-forward OOS folds** with an embargo.
7. **FDR control** across the whole test family.

It mirrors the worked pre-registration in
[examples/example_preregistration.md](examples/example_preregistration.md) exactly — the gates
are decided there, then evaluated in the demo.

A second demo, [examples/demo_overfitting_sharpe.py](examples/demo_overfitting_sharpe.py),
shows the same honesty one level up — at strategy *selection*. Search 50 configurations, keep
the best Sharpe, and watch the **Deflated Sharpe Ratio** reject it when it's the maximum of
pure noise (naive PSR ≈ 0.98 but DSR ≈ 0.47, PBO ≈ 0.5) yet accept a single genuine edge (DSR
≈ 0.996, PBO ≈ 0.01).

## Tests

```bash
pip install pytest
pytest
```

The suite asserts the *methodological invariants*, not just that the code runs: the planted
signal is recovered, the planted confound collapses within-event, the nulls fail, the embargo
widens the train/test gap, and FDR control behaves.

## Where this sits in the literature

The harness is a tested, single-package distillation of standard quant-validation practice, not
a new method. The map:

- **Walk-forward + measured embargo** is the practitioner form of *purging and embargoing* from
  López de Prado, *Advances in Financial Machine Learning* (2018).
- **Deflated / Probabilistic Sharpe Ratio** and **PBO (CSCV)** are Bailey & López de Prado
  (2012, 2014) and Bailey, Borwein, López de Prado & Zhu (2017).
- **Family-wise / FDR control** is Bonferroni and Benjamini-Hochberg (1995); the returns-space
  cousins of "did the best of many strategies beat chance" are White's Reality Check (2000) and
  Hansen's SPA (2005). Full mapping in [docs/methodology.md](docs/methodology.md#where-this-sits--related-work).

## Scope & honesty

- **No alpha claims.** The harness is about *validating decision logic*, not finding edge. A
  levered single-factor clone tracks a strategy's P&L while sharing none of its logic, so P&L
  correlation is treated as non-probative throughout.
- **Synthetic only.** Every number in this repo comes from a simulator with a documented data
  generating process. The point is the method.

## Companion work

The same validation discipline, in a different domain — two public ML projects built on
market-like data (noisy, autocorrelated, regime-shifting, ruthless about look-ahead bias):

- [property-price-avm](https://github.com/niles-liu/property-price-avm) — time-aware price
  forecasting with conformal uncertainty.
- [comparable-property-ranker](https://github.com/niles-liu/comparable-property-ranker) —
  cross-sectional learning-to-rank (the core mechanic of systematic-equity selection).

## License

[MIT](LICENSE).
