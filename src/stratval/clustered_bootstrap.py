"""Event-clustered bootstrap inference for decision-level ranking metrics.

The cardinal sin in trading-strategy validation is treating correlated decisions as
independent observations. When a strategy rebalances, every name it touches in the same
event shares one signal snapshot, one regime, one set of market conditions. Bootstrapping
over individual *rows* then reports a confidence interval far tighter than the data earns —
you "discover" significance that is really just autocorrelation.

The fix is to resample the **cluster** (the rebalance event / trading day / decision batch),
not the row. Every function here takes a long-form decision frame with an ``event_col`` and
resamples whole events with replacement, so the reported precision reflects the number of
*independent* decision occasions, not the number of rows.

Schema expected (long form, one row per decision):
    event_col : the cluster id (e.g. rebalance-event id, trading date)
    label     : binary outcome being predicted (0/1)
    score     : the continuous signal under test (higher -> label==1)

Nothing here knows about any particular asset, account, or data source — it is pure
inference over a generic ``(event, label, score)`` table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["safe_auc", "clustered_auc", "paired_clustered_auc_delta"]

# A pooled AUC over very few rows or a single class is unstable; treat it as undefined
# rather than letting sklearn raise. This floor is a *pooled* guard only.
_MIN_POOLED_ROWS = 30


def safe_auc(label: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC of ``label`` ranked by ``score``, NaN-safe and degeneracy-safe.

    Returns ``nan`` if fewer than ``_MIN_POOLED_ROWS`` finite rows remain or only one
    class is present (an AUC is undefined there).
    """
    m = np.isfinite(score) & np.isfinite(label)
    if m.sum() < _MIN_POOLED_ROWS or len(np.unique(label[m])) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(label[m], score[m]))


def clustered_auc(
    df: pd.DataFrame,
    score: str,
    label: str = "label",
    event_col: str = "event_id",
    n_boot: int = 2000,
    seed: int = 17,
    bars: tuple[float, ...] = (0.55, 0.60),
) -> dict:
    """Event-clustered bootstrap of ``AUC(label | score)``.

    Resamples whole events with replacement ``n_boot`` times. A resample that happens to
    contain only one class is scored as ``0.5`` rather than dropped: dropping degenerate
    resamples silently biases the lower tail upward (it throws away exactly the unlucky
    draws that a left-tail probability is meant to count), which would overstate
    significance. Tracking the degenerate fraction makes that honesty auditable.

    Returns a dict with the point estimate, the 95% percentile CI, the degenerate-resample
    fraction, the number of independent events, and, for each value in ``bars``, the
    bootstrap probability ``P(AUC < bar)`` — the left-tail mass used as the ship/reject
    gate statistic.
    """
    rng = np.random.default_rng(seed)
    sub = df[[event_col, label, score]].dropna()
    point = safe_auc(sub[label].to_numpy(), sub[score].to_numpy())

    groups = {e: g[[label, score]].to_numpy() for e, g in sub.groupby(event_col)}
    ev_ids = np.array(list(groups.keys()), dtype=object)
    if len(ev_ids) == 0:
        out = {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
               "degenerate_frac": float("nan"), "n_events": 0}
        for b in bars:
            out[f"p_below_{b}"] = float("nan")
        return out

    aucs = np.empty(n_boot)
    n_degen = 0
    for i in range(n_boot):
        pick = rng.choice(len(ev_ids), size=len(ev_ids), replace=True)
        stacked = np.concatenate([groups[ev_ids[j]] for j in pick])
        a = safe_auc(stacked[:, 0], stacked[:, 1])
        if not np.isfinite(a):
            a = 0.5
            n_degen += 1
        aucs[i] = a

    out = {
        "point": point,
        "ci_lo": float(np.quantile(aucs, 0.025)),
        "ci_hi": float(np.quantile(aucs, 0.975)),
        "degenerate_frac": n_degen / n_boot,
        "n_events": len(ev_ids),
    }
    for b in bars:
        out[f"p_below_{b}"] = float((aucs < b).mean())
    return out


def paired_clustered_auc_delta(
    df: pd.DataFrame,
    score_a: str,
    score_b: str,
    label: str = "label",
    event_col: str = "event_id",
    n_boot: int = 2000,
    seed: int = 17,
) -> dict:
    """Paired event-clustered bootstrap of ``Delta = AUC(label|score_a) - AUC(label|score_b)``.

    The two scores are evaluated on the *same* resampled events every iteration. Pairing
    recovers the power an unpaired comparison throws away: the two AUCs move together across
    resamples, so the variance of their difference is much smaller than the difference of
    their variances. This is the right test for "does my signal beat the baseline?" — the
    model and the baseline see identical events.

    Returns both point AUCs, the point delta, its 95% CI, and ``P(Delta <= 0)`` — the
    probability the candidate fails to beat the baseline.
    """
    rng = np.random.default_rng(seed)
    sub = df[[event_col, label, score_a, score_b]].dropna()
    pa = safe_auc(sub[label].to_numpy(), sub[score_a].to_numpy())
    pb = safe_auc(sub[label].to_numpy(), sub[score_b].to_numpy())

    groups = {e: g[[label, score_a, score_b]].to_numpy() for e, g in sub.groupby(event_col)}
    ev_ids = np.array(list(groups.keys()), dtype=object)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(len(ev_ids), size=len(ev_ids), replace=True)
        stacked = np.concatenate([groups[ev_ids[j]] for j in pick])
        a = safe_auc(stacked[:, 0], stacked[:, 1])
        b = safe_auc(stacked[:, 0], stacked[:, 2])
        deltas[i] = 0.0 if not (np.isfinite(a) and np.isfinite(b)) else a - b

    point_delta = (pa - pb) if (np.isfinite(pa) and np.isfinite(pb)) else float("nan")
    return {
        "point_a": pa,
        "point_b": pb,
        "point_delta": point_delta,
        "delta_ci_lo": float(np.quantile(deltas, 0.025)),
        "delta_ci_hi": float(np.quantile(deltas, 0.975)),
        "p_delta_le_0": float((deltas <= 0).mean()),
    }
