"""Null models — the falsification arm of the harness.

A gate is only as trustworthy as its ability to *fail*. Before believing a signal, you must
show that the exact same pipeline returns "no signal" when fed something that provably
carries none. These helpers manufacture such negatives:

  * ``within_event_shuffle`` permutes the score within each event, destroying the
    score<->label link while preserving every base rate, event size, and class balance. A
    correctly-built within-event metric must collapse to ~0.5 on it.

  * ``anti_signal`` negates the score. If the original ranks buys above sells, the negation
    must rank *below* 0.5 — a cheap orientation check that catches sign-flip bugs.

  * ``label_shuffle`` permutes the labels within each event — the classic permutation null
    for "is there any score->label relationship at all?"

Each returns a new column you feed back through the same metric you are gating with, so the
null is evaluated by the identical code path as the real statistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["within_event_shuffle", "anti_signal", "label_shuffle"]


def within_event_shuffle(
    df: pd.DataFrame, score: str, event_col: str = "event_id", seed: int = 17
) -> pd.Series:
    """Return ``score`` randomly permuted *within* each event (base rates preserved)."""
    rng = np.random.default_rng(seed)
    return df.groupby(event_col)[score].transform(lambda s: rng.permutation(s.to_numpy()))


def anti_signal(df: pd.DataFrame, score: str) -> pd.Series:
    """Return the negated score (orientation falsification)."""
    return -df[score]


def label_shuffle(
    df: pd.DataFrame, label: str = "label", event_col: str = "event_id", seed: int = 17
) -> pd.Series:
    """Return ``label`` randomly permuted within each event (permutation null)."""
    rng = np.random.default_rng(seed)
    return df.groupby(event_col)[label].transform(lambda s: rng.permutation(s.to_numpy()))
