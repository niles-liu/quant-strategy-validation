"""Within-event (period-stripped) ranking metric.

A pooled AUC over all decisions mixes two very different questions:

  1. *Across* events: "does the strategy buy more on some days than others?" — this channel
     is dominated by time-varying base rates and regime, and is exactly where a spurious
     regime-clock variable fakes skill.
  2. *Within* an event: "given the names traded on this one occasion, does the signal rank
     the buys above the sells?" — this is the question that actually isolates the decision
     rule, with the period effect differenced out.

``within_event_auc`` answers (2). It computes one AUC per event over the names in that
event, then takes a decision-weighted average across events, with a 95% CI from an
event-clustered bootstrap. Because each event is scored on its own, any cross-event base
rate cancels — a variable that only tracks the calendar collapses to ~0.5 here even if it
looks skillful pooled.

Note the deliberate ``min_rows`` floor (default 4): single-event AUCs are only meaningful
with a few names of each class present. This is a *per-event* count, distinct from the
pooled stability floor in :mod:`stratval.clustered_bootstrap` — conflating the two silently
drops contributing events and corrupts the power accounting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["within_event_auc"]


def within_event_auc(
    df: pd.DataFrame,
    score: str,
    label: str = "label",
    event_col: str = "event_id",
    min_rows: int = 4,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict:
    """Decision-weighted within-event AUC of ``label`` ranked by ``score``.

    Only events that (a) contain both classes and (b) have at least ``min_rows`` finite
    rows contribute. Each contributing event's AUC is weighted by its finite-row count; the
    point estimate is that weighted average. The CI and the left-tail gate probability come
    from resampling the contributing events with replacement.

    Returns the point estimate, 95% CI, the number of contributing events, and
    ``P(AUC <= 0.50)`` — the under-chance probability used as the within-event gate.
    """
    from sklearn.metrics import roc_auc_score

    rows = []  # (event_auc, n_rows)
    for _, g in df.groupby(event_col):
        gg = g[[label, score]].dropna()
        if gg[label].nunique() == 2 and len(gg) >= min_rows:
            rows.append((roc_auc_score(gg[label].to_numpy(), gg[score].to_numpy()), len(gg)))

    if not rows:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_events": 0, "p_le_050": float("nan")}

    a = np.array([r[0] for r in rows])
    w = np.array([r[1] for r in rows], dtype=float)
    point = float(np.average(a, weights=w))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.choice(idx, size=len(idx), replace=True)
        boot[i] = np.average(a[p], weights=w[p])

    return {
        "point": point,
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "n_events": len(a),
        "p_le_050": float((boot <= 0.50).mean()),
    }
