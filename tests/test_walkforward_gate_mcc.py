"""Walk-forward splitting, the gate object, and multiple-comparison control."""
import numpy as np
import pandas as pd

from stratval import (
    Gate,
    benjamini_hochberg,
    bonferroni,
    feature_autocorr_at_lag,
    make_expanding_folds,
    within_event_auc,
)


def test_expanding_folds_are_strictly_prior(panel):
    blocks = sorted(panel["block"].unique())
    folds = make_expanding_folds(panel, test_blocks=blocks[2:], embargo=5,
                                 min_bothclass_events=5)
    assert len(folds) == len(blocks) - 2
    for f in folds:
        train = panel.iloc[f["train_idx"]]
        test = panel.iloc[f["test_idx"]]
        if len(train):
            # every training event-date precedes the embargo cut, which precedes the test start
            assert train["event_date"].max() < f["embargo_cut"] <= f["test_start"]
        # no row appears in both train and test
        assert set(f["train_idx"]).isdisjoint(set(f["test_idx"]))


def test_embargo_widens_the_gap(panel):
    blocks = sorted(panel["block"].unique())
    f0 = make_expanding_folds(panel, test_blocks=blocks[2:], embargo=0, min_bothclass_events=5)
    f5 = make_expanding_folds(panel, test_blocks=blocks[2:], embargo=5, min_bothclass_events=5)
    # a positive embargo never trains on more recent data than no embargo
    for a, b in zip(f0, f5):
        assert b["n_train"] <= a["n_train"]


def test_feature_autocorr_diagnostic():
    # a fast (white) panel decorrelates at lag; a slow (cumulative) one stays correlated
    rng = np.random.default_rng(0)
    white = pd.DataFrame(rng.normal(size=(300, 8)))
    slow = white.cumsum()
    assert abs(feature_autocorr_at_lag(white, 10)) < 0.2
    assert feature_autocorr_at_lag(slow, 10) > 0.7


def test_gate_pass_fail(panel):
    g = Gate(name="t", statistic="p_le_050", threshold=0.05, direction="le",
             metric=within_event_auc, pass_interp="ok", fail_interp="no")
    res = g.evaluate(panel, "unreal_ret")
    assert res.passed
    assert "PASS" in str(res)

    # a gate reading a bad statistic name must error, not silently pass
    bad = Gate(name="b", statistic="not_a_key", threshold=0.5, direction="ge",
               metric=within_event_auc)
    try:
        bad.evaluate(panel, "unreal_ret")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_gate_rejects_bad_direction():
    try:
        Gate(name="x", statistic="point", threshold=0.5, direction="bigger",
             metric=within_event_auc)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_benjamini_hochberg():
    # 3 strong signals + 2 nulls
    p = [0.001, 0.002, 0.01, 0.4, 0.9]
    bh = benjamini_hochberg(p, alpha=0.05)
    assert bh["reject"].tolist() == [True, True, True, False, False]
    assert (bh["p_adjusted"] >= np.array(p)).all()    # q >= p
    assert (bh["p_adjusted"] <= 1.0).all()


def test_bonferroni_is_more_conservative():
    p = [0.02, 0.03, 0.9]
    bonf = bonferroni(p, alpha=0.05)
    bh = benjamini_hochberg(p, alpha=0.05)
    # Bonferroni rejects no more than BH
    assert bonf["reject"].sum() <= bh["reject"].sum()
    assert np.isclose(bonf["threshold"], 0.05 / 3)
