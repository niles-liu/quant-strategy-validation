"""Sharpe-space backtest-overfitting controls: deflation must reject the best-of-N noise
winner and accept a genuine edge, and PBO must separate the two. These assert the
*methodological invariants*, not just that the code runs."""
import numpy as np

from stratval import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    generate_trial_returns,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def _best(R):
    srs = np.array([sharpe_ratio(R[c]) for c in R.columns])
    return int(np.nanargmax(srs)), srs


def test_sharpe_basic_and_degenerate():
    assert np.isnan(sharpe_ratio([1.0]))            # < 2 obs
    assert np.isnan(sharpe_ratio([2.0, 2.0, 2.0]))  # zero dispersion
    rng = np.random.default_rng(0)
    r = rng.normal(0.05, 1.0, 5000)
    assert abs(sharpe_ratio(r) - 0.05) < 0.03


def test_expected_max_sharpe_grows_with_trials():
    """More trials -> a higher luck bar; a single trial has no deflation."""
    assert expected_max_sharpe(1, 0.01) == 0.0
    bars = [expected_max_sharpe(n, var_trials_sr=0.005) for n in (2, 10, 100, 1000)]
    assert all(b2 > b1 for b1, b2 in zip(bars, bars[1:]))


def test_psr_increases_with_sample_length():
    """Same Sharpe, more data -> more confident it clears zero."""
    rng = np.random.default_rng(1)
    short = probabilistic_sharpe_ratio(rng.normal(0.05, 1.0, 200))["psr"]
    long = probabilistic_sharpe_ratio(rng.normal(0.05, 1.0, 5000))["psr"]
    assert long > short


def test_dsr_rejects_noise_winner():
    """Best of 50 pure-noise trials: naive PSR is fooled, DSR is not."""
    R = generate_trial_returns(n_trials=50, n_obs=1000, n_genuine=0, seed=7)
    best, srs = _best(R)
    naive = probabilistic_sharpe_ratio(R.iloc[:, best], sr_benchmark=0.0)["psr"]
    dsr = deflated_sharpe_ratio(R.iloc[:, best], n_trials=50, trials_sr=srs)
    assert naive > 0.5            # the lucky winner looks good on a naive read
    assert dsr["dsr"] < 0.95      # ... but fails once deflated for the 50 trials
    assert dsr["sr0_expected_max"] > 0.0


def test_dsr_accepts_genuine_edge():
    """A real edge clears the bar even after deflating for all 50 trials."""
    R = generate_trial_returns(n_trials=50, n_obs=1000, n_genuine=1,
                               genuine_sharpe=0.18, seed=7)
    best, srs = _best(R)
    assert best == 0              # the planted-edge column wins the search
    dsr = deflated_sharpe_ratio(R.iloc[:, best], n_trials=50, trials_sr=srs)
    assert dsr["dsr"] >= 0.95


def test_dsr_var_from_trials_matches_explicit():
    R = generate_trial_returns(n_trials=30, n_obs=800, n_genuine=0, seed=3)
    best, srs = _best(R)
    a = deflated_sharpe_ratio(R.iloc[:, best], n_trials=30, trials_sr=srs)
    b = deflated_sharpe_ratio(R.iloc[:, best], n_trials=30,
                              var_trials_sr=float(np.var(srs, ddof=1)))
    assert a["dsr"] == b["dsr"]


def test_pbo_high_on_noise_low_on_edge():
    """PBO ~ 0.5 when the search has no real winner; ~0 when one trial truly dominates."""
    noise = generate_trial_returns(n_trials=50, n_obs=1000, n_genuine=0, seed=7)
    edge = generate_trial_returns(n_trials=50, n_obs=1000, n_genuine=1,
                                  genuine_sharpe=0.18, seed=7)
    pbo_noise = probability_of_backtest_overfitting(noise.to_numpy(), n_splits=14)
    pbo_edge = probability_of_backtest_overfitting(edge.to_numpy(), n_splits=14)
    assert 0.35 < pbo_noise["pbo"] < 0.65       # no out-of-sample value
    assert pbo_edge["pbo"] < 0.10               # winner generalizes
    # the overfitting signature: healthy IS Sharpe collapses OOS on noise
    assert pbo_noise["median_is_sharpe"] > pbo_noise["median_oos_sharpe"]


def test_pbo_is_deterministic_and_validates_inputs():
    R = generate_trial_returns(n_trials=20, n_obs=600, seed=1)
    a = probability_of_backtest_overfitting(R.to_numpy(), n_splits=10)
    b = probability_of_backtest_overfitting(R.to_numpy(), n_splits=10)
    assert a["pbo"] == b["pbo"]                  # exhaustive CSCV, no RNG
    import pytest
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(R.to_numpy(), n_splits=7)  # must be even
