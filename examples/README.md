# Examples

- **[example_preregistration.md](example_preregistration.md)** — the gate-first
  pre-registration for the demo. Written before any statistic is computed; the gates here are
  the ones the demo evaluates.
- **[demo_harvest_signal.py](demo_harvest_signal.py)** — the full validation pipeline run
  end-to-end on synthetic data. No install required (it adds `../src` to the path):

  ```bash
  python examples/demo_harvest_signal.py
  ```

- **[demo_overfitting_sharpe.py](demo_overfitting_sharpe.py)** — the backtest-overfitting trap
  in Sharpe space: search 50 strategy configurations, keep the best Sharpe, and watch the
  Deflated Sharpe Ratio and PBO reject a lucky noise winner while accepting a genuine edge.

  ```bash
  python examples/demo_overfitting_sharpe.py
  ```

Both demos are deterministic (every bootstrap is seeded; the CSCV in the second is exhaustive),
so their output reproduces exactly.
