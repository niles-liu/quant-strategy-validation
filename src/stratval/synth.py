"""Synthetic strategy-decision generator.

The harness is demonstrated entirely on *simulated* data with a *known* ground truth, so the
checks can be shown to recover a planted effect and reject planted confounds — no real
account, holdings, or trades anywhere.

``generate_rebalance_panel`` simulates a stylized systematic rebalancing process. On each
rebalance *event* the manager looks at a set of names and decides to add (``label==1``) or
trim (``label==0``) each. Three quantities drive the simulation:

  * ``unreal_ret`` — each name's unrealized return vs its cost basis. The planted decision
    rule is a tax-loss-harvesting-style "trim what's underwater": higher ``unreal_ret`` makes
    an *add* more likely. This is the real, within-event signal the harness should recover.

  * ``momentum`` — a recent-return feature that is *correlated* with ``unreal_ret`` but only
    weakly drives the decision. It exists so the demo can show a signal being separated from a
    collinear cousin.

  * ``regime_clock`` — a slow latent that drifts across events and is **constant within any
    one event**. It shifts each event's overall add-rate (a bull stretch adds more), so pooled
    over all rows it *looks* predictive — yet it carries no within-event information. This is
    the planted confound: a variable that fakes skill in a pooled AUC and must collapse to
    ~0.5 under a within-event metric. (Its real-world analogue is a leverage/regime clock that
    spuriously flips within-name reads.)

Returns a tidy long-form frame with exactly the schema the rest of the package consumes:
``event_id, event_date, block, name, unreal_ret, momentum, regime_clock, label``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["generate_rebalance_panel", "generate_trial_returns"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_rebalance_panel(
    n_events: int = 120,
    names_per_event: tuple[int, int] = (12, 30),
    beta_harvest: float = 0.7,
    beta_momentum: float = 0.15,
    regime_intercept: float = 1.3,
    momentum_corr: float = 0.45,
    start_date: int = 20240102,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate a panel of rebalance decisions with a known signal and a known confound.

    Parameters
    ----------
    n_events : number of rebalance occasions (clusters).
    names_per_event : (lo, hi) range; each event's name count is drawn uniformly in it.
    beta_harvest : strength of the true within-event signal (``unreal_ret`` -> add). Set 0
        for a pure-null panel (useful for tests that the gate then correctly fails).
    beta_momentum : weak direct effect of the collinear momentum feature.
    regime_intercept : how strongly the regime clock shifts each event's add-rate (the
        pooled-only confound channel).
    momentum_corr : correlation between ``momentum`` and ``unreal_ret``.
    start_date : first event date as an int ``YYYYMMDD``.
    seed : RNG seed for full reproducibility.

    Returns
    -------
    pandas.DataFrame in the package's long-form decision schema.
    """
    rng = np.random.default_rng(seed)

    # Ordered event dates on a business-day axis; group into calendar-month blocks.
    dates = pd.bdate_range(start=pd.to_datetime(str(start_date)), periods=n_events)
    date_ints = dates.strftime("%Y%m%d").astype(int).to_numpy()
    blocks = dates.strftime("%Y-%m").to_numpy()

    # Slow regime latent: a smooth sinusoidal drift plus a small random walk, normalized.
    t = np.linspace(0, 3 * np.pi, n_events)
    regime = np.sin(t) + np.cumsum(rng.normal(0, 0.15, n_events))
    regime = (regime - regime.mean()) / regime.std()

    rows = []
    rho = momentum_corr
    for e in range(n_events):
        k = int(rng.integers(names_per_event[0], names_per_event[1] + 1))

        # Unrealized return: a regime-tilted mean (bull regimes => more names in the money)
        # plus idiosyncratic spread.
        u = rng.normal(loc=0.4 * regime[e], scale=1.0, size=k)
        # Momentum: correlated with u by construction.
        noise = rng.normal(0, 1.0, size=k)
        m = rho * u + np.sqrt(max(0.0, 1.0 - rho ** 2)) * noise

        # Decision logit: the regime clock shifts the event-level add-rate (pooled-only
        # channel); unreal_ret is the genuine within-event driver.
        logit = (regime_intercept * regime[e]
                 + beta_harvest * u
                 + beta_momentum * m)
        p_add = _sigmoid(logit)
        label = (rng.random(k) < p_add).astype(int)

        rows.append(pd.DataFrame({
            "event_id": e,
            "event_date": date_ints[e],
            "block": blocks[e],
            "name": [f"SYM{e:03d}_{j:02d}" for j in range(k)],
            "unreal_ret": u,
            "momentum": m,
            "regime_clock": regime[e],   # constant within the event by construction
            "label": label,
        }))

    return pd.concat(rows, ignore_index=True)


def generate_trial_returns(
    n_trials: int = 50,
    n_obs: int = 1000,
    n_genuine: int = 0,
    genuine_sharpe: float = 0.10,
    vol: float = 0.01,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate a backtest *search*: ``n_trials`` strategy configurations, ``n_obs`` returns each.

    By default every column is pure noise (zero true edge) — the setup where the best
    in-sample Sharpe is the maximum of a pile of luck, exactly what the Deflated Sharpe Ratio
    and PBO exist to expose. Set ``n_genuine`` to plant that many columns with a real
    per-observation Sharpe of ``genuine_sharpe`` (multiply by ``sqrt(252)`` for the annualized
    figure, e.g. 0.10 -> ~1.6), so the demo can show the deflated statistics *passing* a true
    edge while rejecting the lucky winners.

    Parameters
    ----------
    n_trials : number of strategy configurations searched (the multiple-testing count N).
    n_obs : length of each return series (the sample length T that PSR/DSR correct for).
    n_genuine : how many of the trials carry a real positive edge (the rest are pure noise).
    genuine_sharpe : per-observation Sharpe of each genuine trial (mean = genuine_sharpe*vol).
    vol : per-observation return volatility shared by all trials.
    seed : RNG seed for full reproducibility.

    Returns
    -------
    pandas.DataFrame of shape ``(n_obs, n_trials)`` — one column per trial, ready to hand to
    :func:`stratval.deflated_sharpe.probability_of_backtest_overfitting`. The first
    ``n_genuine`` columns are the planted-edge ones.
    """
    rng = np.random.default_rng(seed)
    means = np.zeros(n_trials)
    means[:n_genuine] = genuine_sharpe * vol
    R = rng.normal(loc=means, scale=vol, size=(n_obs, n_trials))
    cols = [f"trial_{i:03d}" for i in range(n_trials)]
    return pd.DataFrame(R, columns=cols)
