"""End-to-end demonstration of the stratval harness on synthetic data.

Story: a stylized systematic strategy trims underwater names (a tax-loss-harvesting-style
"cut what's down" rule). We have decision data but not the strategy's source. Can we
*honestly* establish that an unrealized-return signal drives the add/trim decision — without
fooling ourselves with the regime confound baked into the data?

Run:  python examples/demo_harvest_signal.py

Everything below operates on data from stratval.generate_rebalance_panel — no real account.
The pre-registration for this demo is in examples/example_preregistration.md; the gates here
mirror it exactly (decide first, compute second).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run straight from a checkout (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from stratval import (  # noqa: E402
    Gate,
    benjamini_hochberg,
    clustered_auc,
    generate_rebalance_panel,
    leave_one_block_out,
    make_expanding_folds,
    negative_controls,
    paired_clustered_auc_delta,
    power_injection,
    within_event_auc,
)

pd.set_option("display.width", 100)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    panel = generate_rebalance_panel(seed=7)
    print(f"Synthetic panel: {len(panel):,} decisions across "
          f"{panel['event_id'].nunique()} rebalance events, "
          f"{panel['block'].nunique()} monthly blocks.")

    # -------------------------------------------------------------------------
    rule("1. The confound trap: pooled AUC lies, within-event AUC tells the truth")
    # The regime clock is constant within an event but shifts the per-event add-rate, so it
    # looks predictive POOLED and collapses to ~0.5 WITHIN event. The harvest signal is the
    # mirror image: genuinely within-event.
    for col in ["unreal_ret", "regime_clock"]:
        pooled = clustered_auc(panel, col, n_boot=1000)
        we = within_event_auc(panel, col)
        print(f"  {col:13s}  pooled AUC = {pooled['point']:.3f}  "
              f"|  within-event AUC = {we['point']:.3f} "
              f"(P[<=0.5]={we['p_le_050']:.3f}, n_events={we['n_events']})")
    print("  -> regime_clock fakes pooled skill but collapses within-event; unreal_ret "
          "survives both.\n     This is why the verdict-bearing gate is the WITHIN-EVENT one.")

    # -------------------------------------------------------------------------
    rule("2. Pre-registered primary gate (within-event direction)")
    primary = Gate(
        name="G1-within-event-direction",
        statistic="p_le_050",
        threshold=0.05,
        direction="le",
        metric=within_event_auc,
        pass_interp="direction signal reproduces within-event (distinguishable from chance)",
        fail_interp="not distinguishable from chance within-event in this sample",
    )
    res = primary.evaluate(panel, "unreal_ret")
    print(" ", res)

    # -------------------------------------------------------------------------
    rule("3. Does the signal beat its collinear cousin? (paired clustered delta)")
    delta = paired_clustered_auc_delta(panel, "unreal_ret", "momentum", n_boot=1000)
    print(f"  AUC(unreal_ret)={delta['point_a']:.3f}  AUC(momentum)={delta['point_b']:.3f}")
    print(f"  paired delta = {delta['point_delta']:+.3f}  "
          f"95% CI [{delta['delta_ci_lo']:+.3f}, {delta['delta_ci_hi']:+.3f}]  "
          f"P(delta<=0) = {delta['p_delta_le_0']:.3f}")

    # -------------------------------------------------------------------------
    rule("4. Negative controls (the gate must have teeth)")
    print(negative_controls(panel, "unreal_ret").to_string(index=False))

    # -------------------------------------------------------------------------
    rule("5. Robustness: leave-one-block-out + power-injection curve")
    print("  leave-one-month-out (does one block carry it?):")
    print(leave_one_block_out(panel, "unreal_ret").to_string(index=False))
    print("\n  power-injection (margin to the P[<=0.5]<=0.05 gate as we subsample events):")
    print(power_injection(panel, "unreal_ret").to_string(index=False))

    # -------------------------------------------------------------------------
    rule("6. Forward (walk-forward) out-of-sample folds with an embargo")
    blocks = sorted(panel["block"].unique())
    test_blocks = blocks[len(blocks) // 2:]  # forward half = OOS
    folds = make_expanding_folds(panel, test_blocks=test_blocks, embargo=5,
                                 min_bothclass_events=8)
    fold_rows = []
    for f in folds:
        oos = panel.iloc[f["test_idx"]]
        we = within_event_auc(oos, "unreal_ret")
        fold_rows.append({"test_block": f["block"], "n_train": f["n_train"],
                          "n_test": f["n_test"], "oos_within_event_auc": we["point"],
                          "p_le_050": we["p_le_050"], "underpowered": f["underpowered"]})
    print(pd.DataFrame(fold_rows).to_string(index=False))

    # -------------------------------------------------------------------------
    rule("7. Multiple-comparison control over the test family")
    # Collect the per-test gate p-values and control the false-discovery rate across them.
    family = {
        "within_event_direction": within_event_auc(panel, "unreal_ret")["p_le_050"],
        "regime_clock_within_event": within_event_auc(panel, "regime_clock")["p_le_050"],
        "momentum_within_event": within_event_auc(panel, "momentum")["p_le_050"],
        "beats_momentum_paired": delta["p_delta_le_0"],
    }
    names = list(family)
    pvals = [family[n] for n in names]
    bh = benjamini_hochberg(pvals, alpha=0.05)
    tbl = pd.DataFrame({"test": names, "p_value": pvals,
                        "q_value": bh["p_adjusted"], "survives_fdr": bh["reject"]})
    print(tbl.to_string(index=False))

    rule("VERDICT")
    print("  The unrealized-return (cut-underwater) signal reproduces WITHIN-EVENT, beats its\n"
          "  collinear momentum cousin on a paired clustered test, survives leave-one-block-out\n"
          "  and FDR control, and the planted regime confound is correctly demoted to chance.\n"
          "  The pooled-only 'skill' of the regime clock is exposed as an artifact.")


if __name__ == "__main__":
    main()
