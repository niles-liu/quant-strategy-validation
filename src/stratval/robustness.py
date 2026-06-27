"""Robustness suite — stress-test a verdict-bearing statistic against the choices that could
be carrying it.

Passing one gate is necessary, not sufficient. A result you would stake a claim on must also
survive: dropping any single time block (no one month is doing all the work), honest power
accounting (how close did the test run to its failure threshold?), and explicit negative
controls (the gate provably fails on shuffled / inverted signal). These checks turn "it
passed" into "it passed, and here is why I believe it is not an artifact."

Everything here is built on :func:`stratval.within_event.within_event_auc` and the null
constructors in :mod:`stratval.nulls`, so the stress tests run through the identical metric
the gate uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nulls import anti_signal, within_event_shuffle
from .within_event import within_event_auc

__all__ = ["leave_one_block_out", "power_injection", "negative_controls"]


def leave_one_block_out(
    df: pd.DataFrame,
    score: str,
    label: str = "label",
    event_col: str = "event_id",
    block_col: str = "block",
    **auc_kw,
) -> pd.DataFrame:
    """Re-run the within-event AUC with each block dropped in turn.

    Returns one row per dropped block with the recomputed point AUC, contributing-event
    count, and gate probability ``p_le_050``. If one block's removal collapses the result,
    that block was carrying it.
    """
    out = []
    for blk in sorted(df[block_col].unique()):
        sub = df[df[block_col] != blk]
        r = within_event_auc(sub, score, label=label, event_col=event_col, **auc_kw)
        out.append({"dropped_block": blk, "auc": r["point"],
                    "n_events": r["n_events"], "p_le_050": r["p_le_050"]})
    return pd.DataFrame(out)


def power_injection(
    df: pd.DataFrame,
    score: str,
    label: str = "label",
    event_col: str = "event_id",
    fractions: tuple[float, ...] = (0.3, 0.5, 0.7, 0.85, 1.0),
    n_draws: int = 40,
    seed: int = 11,
    **auc_kw,
) -> pd.DataFrame:
    """Subsample events at each fraction and report the margin to the ``p_le_050 <= 0.05`` gate.

    For each fraction we draw ``n_draws`` random subsamples of events and report the median
    gate probability and the share of subsamples that still pass. The resulting curve shows
    how much headroom the full-sample result has — a gate that only passes at 100% of the
    data is fragile.
    """
    rng = np.random.default_rng(seed)
    ev = np.array(sorted(df[event_col].unique()), dtype=object)
    rows = []
    for frac in fractions:
        k = max(1, int(round(frac * len(ev))))
        ps = []
        for _ in range(n_draws):
            pick = rng.choice(len(ev), size=k, replace=False)
            sub = df[df[event_col].isin(ev[pick])]
            r = within_event_auc(sub, score, label=label, event_col=event_col, **auc_kw)
            if r["n_events"] > 0 and np.isfinite(r["p_le_050"]):
                ps.append(r["p_le_050"])
        rows.append({
            "fraction": frac,
            "n_events_approx": k,
            "median_p_le_050": float(np.median(ps)) if ps else float("nan"),
            "share_passing": float(np.mean([p <= 0.05 for p in ps])) if ps else float("nan"),
        })
    return pd.DataFrame(rows)


def negative_controls(
    df: pd.DataFrame,
    score: str,
    label: str = "label",
    event_col: str = "event_id",
    seed: int = 17,
    **auc_kw,
) -> pd.DataFrame:
    """Falsification: the gate MUST fail on a within-event-shuffled score and MUST invert on
    the negated score.

    Returns a small table comparing the real score against ``shuffled`` (expected ~0.5, gate
    should not pass) and ``anti`` (expected < 0.5). If either control "passes", the gate has
    no teeth and the headline result is suspect.
    """
    work = df.copy()
    work["__shuffled"] = within_event_shuffle(work, score, event_col=event_col, seed=seed)
    work["__anti"] = anti_signal(work, score)

    rows = []
    for name, col, expectation in [
        ("real", score, "passes (auc>0.5, p_le_050<=0.05)"),
        ("shuffled", "__shuffled", "must NOT pass (auc~0.5)"),
        ("anti", "__anti", "must invert (auc<0.5)"),
    ]:
        r = within_event_auc(work, col, label=label, event_col=event_col, **auc_kw)
        rows.append({"variant": name, "auc": r["point"], "p_le_050": r["p_le_050"],
                     "n_events": r["n_events"], "expectation": expectation})
    return pd.DataFrame(rows)
