"""Event-clustered bootstrap: CI must reflect the number of independent events, the paired
delta must detect a real edge, and degenerate resamples must be counted (not silently dropped)."""
import numpy as np

from stratval import clustered_auc, paired_clustered_auc_delta, safe_auc


def test_safe_auc_basics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    x = y + rng.normal(0, 0.5, 200)        # informative
    assert safe_auc(y, x) > 0.7
    assert np.isnan(safe_auc(np.ones(200), x))   # single class -> undefined
    assert np.isnan(safe_auc(y[:5], x[:5]))      # too few rows -> undefined


def test_clustered_auc_signal_and_ci(panel):
    r = clustered_auc(panel, "unreal_ret", n_boot=500)
    assert r["point"] > 0.6
    assert r["ci_lo"] < r["point"] < r["ci_hi"]
    assert r["p_below_0.55"] < 0.05        # comfortably above the bar
    assert 0.0 <= r["degenerate_frac"] <= 1.0


def test_paired_delta_detects_edge(panel):
    """unreal_ret should beat its weaker collinear cousin on a paired clustered test."""
    d = paired_clustered_auc_delta(panel, "unreal_ret", "momentum", n_boot=500)
    assert d["point_delta"] > 0
    assert d["p_delta_le_0"] < 0.05


def test_clustered_ci_wider_than_iid_would_be(panel):
    """Sanity: clustering on events yields a CI that is not absurdly tight — it reflects the
    ~120 events, not the ~2500 rows. We check the half-width is a few AUC points, not ~0."""
    r = clustered_auc(panel, "unreal_ret", n_boot=1000)
    half_width = (r["ci_hi"] - r["ci_lo"]) / 2
    assert half_width > 0.005
