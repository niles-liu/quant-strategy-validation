"""Sharpe-space backtest-overfitting control: PSR, the Deflated Sharpe Ratio, and the
Probability of Backtest Overfitting (CSCV).

The decision-level machinery elsewhere in this package answers "does this signal rank the
right names?". The complementary question every systematic-research program faces is "I tried
*many* configurations and kept the best Sharpe — is that Sharpe real, or is it the maximum of
a pile of noise?". Selecting the best of N trials inflates the winner's Sharpe even when no
strategy has any edge: the expected maximum of N independent t-like statistics grows with N.
Reporting that winning Sharpe without deflating for N is the single most common way a backtest
lies.

This module packages the standard tools for that problem:

  * :func:`probabilistic_sharpe_ratio` (PSR) — the probability the true Sharpe exceeds a
    benchmark, correcting the Sharpe's sampling error for sample length **and** non-normality
    (skew/kurtosis), after Bailey & Lopez de Prado (2012).

  * :func:`deflated_sharpe_ratio` (DSR) — PSR evaluated against a benchmark ``SR0`` that is the
    *expected maximum* Sharpe under the null of no skill given ``N`` trials and the spread of
    their Sharpes. A DSR below ~0.95 says "this winner is consistent with luck across that many
    trials." After Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".

  * :func:`probability_of_backtest_overfitting` (PBO) — Combinatorially-Symmetric
    Cross-Validation (CSCV): across every balanced in-sample/out-of-sample split of the trial
    matrix, how often does the in-sample-best configuration land below the OOS median? A high
    PBO means the selection procedure itself overfits. After Bailey, Borwein, Lopez de Prado &
    Zhu (2017).

All Sharpe ratios here are **per-observation** (not annualized); annualize for display only by
multiplying by ``sqrt(periods_per_year)``. The horizon enters the statistics through the
sample length ``T``, so mixing annualized Sharpes into these formulas would double-count it.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.stats import norm

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
]

_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns) -> float:
    """Per-observation Sharpe ratio ``mean / std`` of a return series.

    Uses the sample standard deviation (``ddof=1``). Returns ``nan`` for fewer than two finite
    observations or zero dispersion. Annualize for display by multiplying by
    ``sqrt(periods_per_year)``; leave it per-observation for every function in this module.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> dict:
    """Probabilistic Sharpe Ratio: ``P(true SR > sr_benchmark)`` given the observed series.

    The estimated Sharpe has sampling error that shrinks with sample length ``T`` and is
    *inflated by negative skew and fat tails* — exactly the return shape of most trading
    strategies. The PSR corrects for both via the third and fourth moments (Bailey &
    Lopez de Prado, 2012):

        PSR(SR0) = Phi( (SR - SR0) * sqrt(T - 1)
                        / sqrt(1 - g3*SR + ((g4 - 1)/4)*SR^2) )

    where ``g3`` is skewness and ``g4`` is (non-excess) kurtosis. Returns a dict with the
    observed Sharpe, the moments used, and ``psr`` in ``[0, 1]``. A common ship bar is
    ``psr >= 0.95``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    sr = sharpe_ratio(r)
    if T < 3 or not np.isfinite(sr):
        return {"sharpe": sr, "psr": float("nan"), "n_obs": int(T),
                "skew": float("nan"), "kurtosis": float("nan")}

    # Moments with the same population convention used throughout the DSR literature.
    z = (r - r.mean()) / r.std(ddof=0)
    g3 = float(np.mean(z ** 3))
    g4 = float(np.mean(z ** 4))  # non-excess: a normal series gives ~3

    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2))
    stat = (sr - sr_benchmark) * math.sqrt(T - 1) / denom
    return {"sharpe": sr, "psr": float(norm.cdf(stat)), "n_obs": int(T),
            "skew": g3, "kurtosis": g4}


def expected_max_sharpe(n_trials: int, var_trials_sr: float) -> float:
    """Expected maximum per-observation Sharpe under the null of *no* skill across ``n_trials``.

    The benchmark ``SR0`` a deflated Sharpe must clear. It is the expected maximum of
    ``n_trials`` independent draws whose Sharpes have variance ``var_trials_sr``, via the
    extreme-value (Gumbel) approximation used in Bailey & Lopez de Prado (2014):

        SR0 = sqrt(V) * [ (1 - gamma) * Phi^{-1}(1 - 1/N)
                          + gamma * Phi^{-1}(1 - 1/(N*e)) ]

    where ``gamma`` is the Euler-Mascheroni constant. More trials, or a wider spread of trial
    Sharpes, raise the bar a genuine result must beat. Returns ``0.0`` for a single trial.
    """
    if n_trials is None or n_trials <= 1 or not np.isfinite(var_trials_sr) or var_trials_sr <= 0:
        return 0.0
    N = float(n_trials)
    g = _EULER_MASCHERONI
    q1 = norm.ppf(1.0 - 1.0 / N)
    q2 = norm.ppf(1.0 - 1.0 / (N * math.e))
    return float(math.sqrt(var_trials_sr) * ((1.0 - g) * q1 + g * q2))


def deflated_sharpe_ratio(
    returns,
    n_trials: int,
    var_trials_sr: float | None = None,
    trials_sr=None,
) -> dict:
    """Deflated Sharpe Ratio: PSR of the selected strategy against the expected-max benchmark.

    Supply the variance of the trials' Sharpes either directly (``var_trials_sr``) or as the
    array of all trial Sharpes (``trials_sr``), from which the variance is computed. The DSR is
    ``probabilistic_sharpe_ratio(returns, sr_benchmark=SR0)`` with ``SR0 =
    expected_max_sharpe(n_trials, var_trials_sr)``.

    Interpretation: ``dsr`` is the probability the winner's true Sharpe beats what selecting the
    best of ``n_trials`` would produce by luck alone. ``dsr >= 0.95`` is the usual ship bar; a
    high *raw* Sharpe with a low DSR is the signature of an overfit search.
    """
    if var_trials_sr is None:
        if trials_sr is None:
            raise ValueError("provide var_trials_sr or trials_sr")
        s = np.asarray(trials_sr, dtype=float)
        s = s[np.isfinite(s)]
        var_trials_sr = float(s.var(ddof=1)) if s.size >= 2 else float("nan")

    sr0 = expected_max_sharpe(n_trials, var_trials_sr)
    psr = probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)
    return {
        "sharpe": psr["sharpe"],
        "sr0_expected_max": sr0,
        "n_trials": int(n_trials),
        "var_trials_sr": float(var_trials_sr),
        "dsr": psr["psr"],
        "n_obs": psr["n_obs"],
    }


def _col_sharpe(sum1: np.ndarray, sum2: np.ndarray, n: int) -> np.ndarray:
    """Per-column per-observation Sharpe from precomputed sums (sample std, ddof=1)."""
    mean = sum1 / n
    var = (sum2 - n * mean ** 2) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mean / sd, np.nan)
    return sr


def probability_of_backtest_overfitting(
    returns_matrix,
    n_splits: int = 16,
) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation.

    ``returns_matrix`` is ``(T observations, N trials)`` — one column per strategy
    configuration tried. The rows are partitioned into ``n_splits`` contiguous blocks (must be
    even); for every balanced split into in-sample (half the blocks) and out-of-sample (the
    rest), we pick the IS-best configuration by Sharpe and look up its OOS rank. The relative
    OOS rank ``omega`` gives a logit ``lambda = ln(omega / (1 - omega))``; the IS-best falling
    below the OOS median (``lambda <= 0``) is an instance of overfitting.

    PBO is the fraction of splits with ``lambda <= 0``. A PBO near 0.5 means the selection
    procedure has no out-of-sample value (the winner is as likely to be below the OOS median as
    above); near 0 means the in-sample winner reliably generalizes. Returns the PBO, the logit
    array, and the median IS/OOS Sharpe of the selected configurations (the performance-decay
    diagnostic). CSCV is exhaustive and deterministic — no seed needed.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T observations x N trials)")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    T, N = R.shape
    if n_splits > T:
        raise ValueError(f"n_splits={n_splits} exceeds T={T} observations")

    # Per-block sufficient statistics, so each split is assembled in O(N), not O(T*N).
    block_ids = np.array_split(np.arange(T), n_splits)
    n_b = np.array([len(b) for b in block_ids])
    sum1_b = np.vstack([R[b].sum(axis=0) for b in block_ids])           # (n_splits, N)
    sum2_b = np.vstack([(R[b] ** 2).sum(axis=0) for b in block_ids])    # (n_splits, N)

    all_blocks = set(range(n_splits))
    logits, is_sr_star, oos_sr_star = [], [], []
    for combo in combinations(range(n_splits), n_splits // 2):
        is_b = list(combo)
        oos_b = list(all_blocks.difference(combo))

        n_is = int(n_b[is_b].sum())
        n_oos = int(n_b[oos_b].sum())
        is_sr = _col_sharpe(sum1_b[is_b].sum(0), sum2_b[is_b].sum(0), n_is)
        oos_sr = _col_sharpe(sum1_b[oos_b].sum(0), sum2_b[oos_b].sum(0), n_oos)

        if not np.isfinite(is_sr).any():
            continue
        n_star = int(np.nanargmax(is_sr))

        # Relative OOS rank of the IS winner: 1..N ascending (high = good OOS).
        finite = np.isfinite(oos_sr)
        if not finite[n_star] or finite.sum() < 2:
            continue
        rank = int((oos_sr[finite] <= oos_sr[n_star]).sum())  # 1..(#finite)
        omega = rank / (finite.sum() + 1.0)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))
        is_sr_star.append(float(is_sr[n_star]))
        oos_sr_star.append(float(oos_sr[n_star]))

    lam = np.array(logits, dtype=float)
    pbo = float((lam <= 0).mean()) if lam.size else float("nan")
    return {
        "pbo": pbo,
        "n_combinations": int(lam.size),
        "logits": lam,
        "median_is_sharpe": float(np.median(is_sr_star)) if is_sr_star else float("nan"),
        "median_oos_sharpe": float(np.median(oos_sr_star)) if oos_sr_star else float("nan"),
    }
