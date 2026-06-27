"""Forward (walk-forward) out-of-sample splitting with an embargo, plus the diagnostic that
keeps the embargo honest.

Random K-fold cross-validation is invalid on time series: a fold drawn from the middle of
the sample trains on the future to predict the past, and overlapping feature windows leak
across the fold boundary. The only defensible split is **expanding-window walk-forward** —
train on everything up to a cut, test on the block after it, never the reverse.

Two refinements that matter:

  * **Cluster-aware blocks.** The test blocks are calendar periods (e.g. months), and the
    fold unit is the *event*, so a multi-decision event never straddles a boundary.

  * **Embargo.** Slow features (trailing momentum, rolling betas) computed near the train/test
    seam still share information across it. We truncate the training set a fixed number of
    periods *before* each test block begins. ``feature_autocorr_at_lag`` then *measures* the
    residual serial correlation at the embargo lag, so the leftover optimistic bias is
    quantified rather than assumed away — a fast feature should decorrelate to ~0, while a
    deliberately slow control stays high (and is treated as a control, not the signal).

Generic over any frame with a period column (``block_col``) and an integer event-date column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_expanding_folds", "feature_autocorr_at_lag"]


def make_expanding_folds(
    df: pd.DataFrame,
    test_blocks: list,
    block_col: str = "block",
    event_date_col: str = "event_date",
    label: str = "label",
    event_col: str = "event_id",
    embargo: int = 0,
    min_bothclass_events: int = 20,
    min_test_rows: int = 200,
) -> list[dict]:
    """Build expanding-window folds, one per entry in ``test_blocks``.

    For each test block, the training set is every row whose ``event_date`` is strictly
    earlier than ``embargo`` distinct event-dates before the block's first event-date. Event
    dates are treated as an ordered axis (their sorted unique values), so ``embargo`` counts
    *occasions*, not calendar days — robust to gaps.

    Each fold dict carries integer-position train/test indices plus power diagnostics: the
    number of test events that contain both classes, and an ``underpowered`` flag when a
    block is too thin to support a verdict. Surfacing power per fold stops a single
    well-populated block from silently carrying an aggregate result.
    """
    date_axis = np.array(sorted(df[event_date_col].unique()))
    folds = []
    for blk in test_blocks:
        test_mask = (df[block_col] == blk).to_numpy()
        if test_mask.sum() == 0:
            continue
        test_start = int(df.loc[test_mask, event_date_col].min())
        pos = int(np.searchsorted(date_axis, test_start))
        cut = date_axis[max(0, pos - embargo)]
        train_mask = (df[event_date_col].to_numpy() < cut)

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        te = df.iloc[test_idx]
        bothclass = int(te.groupby(event_col)[label].nunique().eq(2).sum())
        underpowered = (bothclass < min_bothclass_events) or (len(test_idx) < min_test_rows)
        folds.append({
            "block": blk,
            "test_start": test_start,
            "embargo_cut": int(cut),
            "train_idx": train_idx,
            "test_idx": test_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_test_events": int(te[event_col].nunique()),
            "n_test_bothclass_events": bothclass,
            "underpowered": bool(underpowered),
        })
    return folds


def feature_autocorr_at_lag(panel: pd.DataFrame, lag: int) -> float:
    """Mean per-series serial autocorrelation of a wide feature panel at ``lag`` rows.

    ``panel`` is wide: rows = ordered time index, columns = series (names/assets). For each
    column we correlate the series with itself shifted by ``lag`` and average the finite
    per-column correlations. Use it at the embargo length: a value near 0 means the first
    test feature vector no longer shares information with the last training one; a high value
    means the embargo should be longer (or the feature is a slow control to be labelled as
    such, not read as signal).
    """
    arr = panel.to_numpy(dtype=float)
    cors = []
    for j in range(arr.shape[1]):
        s = arr[:, j]
        a, b = s[lag:], s[:-lag]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 30 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
            cors.append(np.corrcoef(a[m], b[m])[0, 1])
    return float(np.nanmean(cors)) if cors else float("nan")
