"""The backtest-overfitting trap, in Sharpe space.

Companion to demo_harvest_signal.py. That demo works at the *decision* level (does a signal
rank the right names within an event?). This one works at the *strategy-selection* level: you
search many configurations and keep the best Sharpe — is that winner real, or just the maximum
of a pile of noise?

The answer is the Deflated Sharpe Ratio (deflate the winner for how many trials you ran) and
the Probability of Backtest Overfitting (does the in-sample winner generalize out-of-sample?).

Run:  python examples/demo_overfitting_sharpe.py

Everything runs on stratval.generate_trial_returns — synthetic trials with a known number of
genuine edges, so the deflated statistics can be shown to reject luck and accept a real edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from stratval import (  # noqa: E402
    deflated_sharpe_ratio,
    generate_trial_returns,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

ANN = np.sqrt(252)  # per-observation -> annualized, for display only
N_TRIALS = 50
N_OBS = 1000
N_SPLITS = 16  # CSCV: C(16,8) = 12,870 balanced splits


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report(R) -> None:
    srs = np.array([sharpe_ratio(R[c]) for c in R.columns])
    best = int(np.nanargmax(srs))
    naive = probabilistic_sharpe_ratio(R.iloc[:, best], sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(R.iloc[:, best], n_trials=R.shape[1], trials_sr=srs)
    pbo = probability_of_backtest_overfitting(R.to_numpy(), n_splits=N_SPLITS)

    print(f"  winner = {R.columns[best]}  "
          f"(raw Sharpe {srs[best]:.4f}/obs = {srs[best] * ANN:.2f} annualized)")
    print(f"  naive PSR vs 0        = {naive['psr']:.3f}   "
          f"<- ignores that you searched {R.shape[1]} trials")
    print(f"  expected-max SR0      = {dsr['sr0_expected_max']:.4f}/obs "
          f"({dsr['sr0_expected_max'] * ANN:.2f} ann.): the luck bar for {R.shape[1]} trials")
    print(f"  Deflated Sharpe (DSR) = {dsr['dsr']:.3f}   "
          f"{'PASS' if dsr['dsr'] >= 0.95 else 'FAIL'} a 0.95 ship bar")
    print(f"  PBO (CSCV, {pbo['n_combinations']:,} splits) = {pbo['pbo']:.3f}   "
          f"IS Sharpe {pbo['median_is_sharpe']:.3f}/obs -> OOS {pbo['median_oos_sharpe']:+.3f}/obs")


def main() -> None:
    rule("1. A pure-noise search: 50 strategies, ZERO real edge")
    noise = generate_trial_returns(n_trials=N_TRIALS, n_obs=N_OBS, n_genuine=0, seed=7)
    report(noise)
    print("  -> The naive PSR is fooled: the best-of-50 Sharpe looks shippable. The DSR is not\n"
          "     -- it deflates the winner against the expected maximum of 50 noise trials and\n"
          "     rejects it. PBO ~ 0.5 confirms the in-sample winner has no out-of-sample value\n"
          "     (its healthy IS Sharpe collapses to ~0 OOS).")

    rule("2. Same search, but one configuration has a GENUINE edge")
    genuine = generate_trial_returns(n_trials=N_TRIALS, n_obs=N_OBS, n_genuine=1,
                                     genuine_sharpe=0.18, seed=7)
    report(genuine)
    print("  -> Now the DSR clears the bar even after deflating for all 50 trials, and PBO is\n"
          "     near 0: the in-sample winner is the same one that wins out-of-sample.")

    rule("VERDICT")
    print("  A high raw (or even a high naive-PSR) Sharpe means nothing without saying how many\n"
          "  configurations you searched. Deflate the winner for the trial count (DSR), and check\n"
          "  that the selection itself generalizes (PBO). Luck fails both; a real edge passes both.")


if __name__ == "__main__":
    main()
