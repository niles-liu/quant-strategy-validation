"""stratval — a leakage-free validation harness for systematic trading-strategy research.

A small, dependency-light toolkit for the part of quantitative research that is easy to get
wrong: deciding, honestly, whether a strategy signal is real. It packages the practices that
separate a real effect from an artifact on noisy, autocorrelated financial data:

  * gate-first pre-registration         (:mod:`stratval.gate`)
  * event-clustered bootstrap inference (:mod:`stratval.clustered_bootstrap`)
  * within-event / period-stripped AUC  (:mod:`stratval.within_event`)
  * walk-forward OOS splitting + embargo (:mod:`stratval.walkforward`)
  * shuffle / permutation null models    (:mod:`stratval.nulls`)
  * a robustness stress-test suite       (:mod:`stratval.robustness`)
  * multiple-comparison control          (:mod:`stratval.multiple_comparison`)
  * deflated Sharpe + backtest-overfitting (:mod:`stratval.deflated_sharpe`)
  * a synthetic data generator for demos (:mod:`stratval.synth`)

See ``docs/methodology.md`` for the philosophy and ``examples/demo_harvest_signal.py`` for an
end-to-end run on synthetic data.
"""
from __future__ import annotations

from .clustered_bootstrap import clustered_auc, paired_clustered_auc_delta, safe_auc
from .deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from .gate import Gate, GateResult
from .multiple_comparison import benjamini_hochberg, bonferroni
from .nulls import anti_signal, label_shuffle, within_event_shuffle
from .robustness import leave_one_block_out, negative_controls, power_injection
from .synth import generate_rebalance_panel, generate_trial_returns
from .walkforward import feature_autocorr_at_lag, make_expanding_folds
from .within_event import within_event_auc

__version__ = "0.1.0"

__all__ = [
    "safe_auc", "clustered_auc", "paired_clustered_auc_delta",
    "within_event_auc",
    "make_expanding_folds", "feature_autocorr_at_lag",
    "within_event_shuffle", "anti_signal", "label_shuffle",
    "leave_one_block_out", "power_injection", "negative_controls",
    "bonferroni", "benjamini_hochberg",
    "sharpe_ratio", "probabilistic_sharpe_ratio", "expected_max_sharpe",
    "deflated_sharpe_ratio", "probability_of_backtest_overfitting",
    "Gate", "GateResult",
    "generate_rebalance_panel", "generate_trial_returns",
]
