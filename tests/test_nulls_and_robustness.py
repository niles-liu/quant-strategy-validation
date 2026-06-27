"""Falsification machinery: shuffles must kill the signal, anti-signal must invert, and the
robustness suite's negative controls must behave."""
import numpy as np

from stratval import (
    anti_signal,
    label_shuffle,
    leave_one_block_out,
    negative_controls,
    power_injection,
    within_event_auc,
    within_event_shuffle,
)


def test_within_event_shuffle_destroys_signal(panel):
    work = panel.copy()
    work["s"] = within_event_shuffle(work, "unreal_ret")
    r = within_event_auc(work, "s")
    assert abs(r["point"] - 0.5) < 0.03
    assert r["p_le_050"] > 0.05


def test_within_event_shuffle_preserves_base_rates(panel):
    """Shuffling within event must not change per-event class balance or row counts."""
    work = panel.copy()
    work["s"] = within_event_shuffle(work, "unreal_ret")
    assert len(work) == len(panel)
    # the multiset of scores within each event is unchanged
    a = panel.groupby("event_id")["unreal_ret"].apply(lambda s: tuple(sorted(s)))
    b = work.groupby("event_id")["s"].apply(lambda s: tuple(sorted(s)))
    assert (a == b).all()


def test_anti_signal_inverts(panel):
    work = panel.copy()
    work["a"] = anti_signal(work, "unreal_ret")
    r = within_event_auc(work, "a")
    assert r["point"] < 0.5


def test_label_shuffle_is_null(panel):
    work = panel.copy()
    work["yl"] = label_shuffle(work, "label")
    r = within_event_auc(work, "unreal_ret", label="yl")
    assert abs(r["point"] - 0.5) < 0.05


def test_negative_controls_table(panel):
    nc = negative_controls(panel, "unreal_ret").set_index("variant")
    assert nc.loc["real", "p_le_050"] <= 0.05
    assert nc.loc["shuffled", "p_le_050"] > 0.05
    assert nc.loc["anti", "auc"] < 0.5


def test_leave_one_block_out_robust(panel):
    lobo = leave_one_block_out(panel, "unreal_ret")
    assert (lobo["auc"] > 0.6).all()     # no single block is carrying it
    assert (lobo["p_le_050"] <= 0.05).all()


def test_power_injection_monotone_ish(panel):
    pw = power_injection(panel, "unreal_ret", n_draws=15)
    # strong signal: passes at every fraction here
    assert (pw["share_passing"] == 1.0).all()
