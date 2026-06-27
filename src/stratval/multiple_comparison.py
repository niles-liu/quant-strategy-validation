"""Multiple-comparison control over a family of pre-registered tests.

Run twenty gates at alpha=0.05 on pure noise and you expect one "discovery". A research
program that probes many hypotheses must therefore control the family-wide error rate, not
just each test in isolation. Two standard procedures:

  * **Bonferroni** controls the family-wise error rate (the probability of *any* false
    positive). Conservative; use when even one false claim is costly.

  * **Benjamini-Hochberg** controls the false-discovery rate (the expected *fraction* of
    claimed discoveries that are false). More powerful; the right default when you expect
    several real effects and can tolerate a known false-discovery proportion.

Both take the raw per-test p-values and return, per test, the adjusted significance verdict
plus adjusted p-values, so a results table can show "survived FDR control" honestly.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bonferroni", "benjamini_hochberg"]


def bonferroni(pvals, alpha: float = 0.05) -> dict:
    """Bonferroni family-wise correction.

    Returns the adjusted threshold, a boolean ``reject`` array (True = significant after
    correction), and Bonferroni-adjusted p-values (``min(p * m, 1)``).
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    thresh = alpha / m if m else float("nan")
    return {
        "threshold": thresh,
        "reject": (p <= thresh),
        "p_adjusted": np.minimum(p * m, 1.0),
    }


def benjamini_hochberg(pvals, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control (independence / positive-dependence form).

    Returns a boolean ``reject`` array aligned to the input order, the BH-adjusted p-values
    (q-values, monotone-enforced and clipped to 1), and the largest passing rank ``k``.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return {"reject": np.array([], dtype=bool), "p_adjusted": np.array([]), "k": 0}

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # Largest i with p_(i) <= (i/m) * alpha.
    passing = ranked <= (ranks / m) * alpha
    k = int(np.max(np.where(passing)[0]) + 1) if passing.any() else 0

    reject_sorted = np.zeros(m, dtype=bool)
    if k > 0:
        reject_sorted[:k] = True

    # BH-adjusted p-values: enforce monotonicity from the largest rank down.
    q_sorted = ranked * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    reject = np.zeros(m, dtype=bool)
    p_adjusted = np.empty(m)
    reject[order] = reject_sorted
    p_adjusted[order] = q_sorted
    return {"reject": reject, "p_adjusted": p_adjusted, "k": k}
