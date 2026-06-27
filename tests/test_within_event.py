"""The within-event metric is the heart of the harness: it must recover a real within-event
signal AND demote a pooled-only confound to chance."""
import numpy as np

from stratval import clustered_auc, within_event_auc


def test_recovers_planted_signal(panel):
    r = within_event_auc(panel, "unreal_ret")
    assert r["point"] > 0.6              # planted signal is recovered
    assert r["p_le_050"] <= 0.05         # distinguishable from chance
    assert r["ci_lo"] > 0.5              # CI excludes chance


def test_regime_confound_collapses_within_event(panel):
    """The regime clock looks skillful pooled but is constant within an event -> ~0.5."""
    pooled = clustered_auc(panel, "regime_clock", n_boot=500)
    we = within_event_auc(panel, "regime_clock")
    assert pooled["point"] > 0.6         # fakes skill pooled
    assert abs(we["point"] - 0.5) < 0.02  # collapses within-event
    assert we["p_le_050"] > 0.05         # gate correctly does NOT pass


def test_null_panel_fails_gate(null_panel):
    r = within_event_auc(null_panel, "unreal_ret")
    assert abs(r["point"] - 0.5) < 0.05
    assert r["p_le_050"] > 0.05


def test_empty_returns_nan():
    import pandas as pd
    empty = pd.DataFrame({"event_id": [], "label": [], "unreal_ret": []})
    r = within_event_auc(empty, "unreal_ret")
    assert r["n_events"] == 0
    assert np.isnan(r["point"])
