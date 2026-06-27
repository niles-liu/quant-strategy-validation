# The gate-first protocol

A short, opinionated checklist for running a hypothesis test on strategy-decision data
without fooling yourself. The whole point is that every item is decided **before** the
verdict-bearing statistic is computed.

## The template

Write a pre-registration document (markdown is fine — see the
[worked example](../examples/example_preregistration.md)) with these sections, in order:

1. **Question.** One sentence. What rule or signal are you testing, on what decision?
2. **Data & grain.** The unit of analysis, and — critically — the **cluster unit** (the
   shared-signal occasion: a rebalance event, a trading day). Inference clusters on this.
3. **Out-of-sample policy.** Which slice is the forward read; the embargo length; an explicit
   ban on random K-fold.
4. **Baselines & nulls.** The "must beat" baseline and the nulls that must fail (within-event
   shuffle ≈ 0.5, anti-signal < 0.5).
5. **Gates.** For each gate: the exact statistic, the threshold, the comparison direction, and
   the pre-committed reading of **both** pass and fail. Name which one or two gates carry
   verdict authority; everything else is descriptive.
6. **Family-wise control.** How the family of tests is corrected (BH-FDR / Bonferroni).
7. **Decision tree.** The verdict for every combination of gate outcomes — written in advance.

## Turning a gate into code

```python
from stratval import Gate, within_event_auc, generate_rebalance_panel

panel = generate_rebalance_panel()

# Locked BEFORE looking at the statistic.
g1 = Gate(
    name="G1-within-event-direction",
    statistic="p_le_050",      # a key the metric returns
    threshold=0.05,
    direction="le",            # observed <= threshold to PASS
    metric=within_event_auc,
    pass_interp="direction signal reproduces within-event",
    fail_interp="not distinguishable from chance in this sample",
)

result = g1.evaluate(panel, "unreal_ret")
print(result)   # [PASS] G1-within-event-direction: p_le_050 = 0.0000 (gate: <= 0.05) -> ...
```

The `Gate` is frozen: there is no method to change the threshold after `evaluate`. If you
find yourself wanting to, that is the signal to stop and write a *new* pre-registration — not
to edit this one.

## The rules that keep it honest

- **No goalpost moves once results land.** Revising the *design* after an adversarial review,
  before any statistic touches the test window, is fine. Re-interpreting a failure as a
  success after seeing the number is the exact thing this protocol forbids.
- **Lead on the cluster-correct, period-stripped statistic.** The pooled AUC is a reference;
  it contains the cross-event base-rate channel and has no verdict authority.
- **Report power next to every number.** Print the contributing-event count beside each
  within-event AUC; flag a gate as under-powered rather than reading a thin result as a null.
- **Reproduce bit-for-bit.** Seed every bootstrap; a result you can't reproduce exactly isn't
  a result yet.
- **State what you carried, not just what you tested.** If participation or sizing is taken as
  given rather than reconstructed, say so next to the verdict — a conditional result is honest
  only when the condition is visible.
