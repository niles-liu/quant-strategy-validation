# Methodology — how to not fool yourself validating a trading strategy

Financial decision data is small, noisy, and autocorrelated, and the person analyzing it
usually *wants* to find a signal. That combination makes spurious "discoveries" the default
outcome, not the exception. This library packages the handful of practices that, in my
experience, do most of the work of telling a real effect from an artifact. Each one exists
because skipping it produces a specific, recurring failure.

The running example throughout is a **direction signal**: at each rebalance *event* a
strategy decides to add or trim each held name, and we want to know whether an
unrealized-return-vs-basis signal (a tax-loss-harvesting-style "cut what's underwater" rule)
drives that decision. The lessons generalize to any per-decision ranking question.

---

## 1. Gate-first: decide the verdict rule before you see the number

The cheapest and most powerful guard is to write down, *before computing anything*, (a) the
exact statistic, (b) the threshold, and (c) what each outcome will mean — and then refuse to
move the goalposts. Researcher degrees of freedom (which window, which subset, which metric,
which cutoff) are what manufacture false positives; pre-registration spends them up front,
in the open.

In code this is [`Gate`](../src/stratval/gate.py): a frozen object carrying a metric, a
threshold, and the pre-committed interpretation of pass *and* fail. You lock it, then
evaluate it; there is deliberately no API to change the threshold after the fact.

> Legitimate vs illegitimate revision: hardening a design with an adversarial review *before*
> any statistic touches the test window is good practice. Re-reading a disappointing number
> as a success afterward is the thing the protocol exists to prevent. The line is "did the
> data move the goalpost?"

See [gate-first-protocol.md](gate-first-protocol.md) and the worked
[example pre-registration](../examples/example_preregistration.md).

---

## 2. Cluster the bootstrap on the shared-signal unit

When a strategy rebalances, every name it touches that day shares one signal snapshot, one
regime, one market state. They are **not** independent observations. Bootstrapping over rows
then reports a confidence interval far tighter than the data earns — you "find" significance
that is really autocorrelation.

The fix is to resample the **cluster** (the event), not the row.
[`clustered_auc`](../src/stratval/clustered_bootstrap.py) resamples whole events with
replacement, so the reported precision reflects the number of independent decision
*occasions*. A subtle but important detail: degenerate resamples (a bootstrap draw that
happens to contain only one class) are scored as 0.5, **not dropped** — dropping them biases
the left tail upward and overstates significance, since it discards exactly the unlucky draws
a left-tail probability is meant to count.

For "does my signal beat the baseline?", use the **paired** version
([`paired_clustered_auc_delta`](../src/stratval/clustered_bootstrap.py)): evaluate both scores
on the *same* resampled events. Pairing recovers the power an unpaired comparison throws away,
because the two AUCs move together across resamples.

---

## 3. Strip the period: within-event beats pooled

This is the lesson that, once seen, you can't unsee. Consider a slow **regime clock** — a
variable that drifts across events and shifts each event's overall buy-rate, but is constant
*within* any one event. Pooled over all decisions it looks predictive (high-regime events buy
more, so the variable "predicts" buys). Yet it carries **zero** within-event information: it
cannot rank the names traded on a single occasion.

The demo makes this concrete on synthetic data where the regime clock is planted as a pure
confound:

| signal | pooled AUC | within-event AUC |
|---|---|---|
| `unreal_ret` (real signal) | 0.75 | **0.71** |
| `regime_clock` (planted confound) | **0.80** | **0.50** |

The confound out-scores the real signal *pooled*, and collapses to exactly chance
*within-event*. So the verdict-bearing metric is the within-event one
([`within_event_auc`](../src/stratval/within_event.py)): one AUC per event, decision-weighted
across events, with an event-clustered CI. The pooled number is kept only as a labelled
reference — never as evidence.

> Real-world origin: a leverage/regime clock that spuriously flips within-name reads is a
> classic way to get a false positive in this kind of work. Banning such book-level scalars
> from the signal set, and leading on the period-stripped statistic, is what kills it.

---

## 4. Forward only: walk-forward with an embargo

Random K-fold cross-validation is invalid on time series — a fold from the middle trains on
the future to predict the past. The only defensible split is **expanding-window walk-forward**:
train up to a cut, test on the block after it, never the reverse
([`make_expanding_folds`](../src/stratval/walkforward.py)).

Two refinements: the fold unit is the *event* (a multi-decision event never straddles a
boundary), and an **embargo** truncates training a fixed number of occasions before each test
block, so slow features computed near the seam don't leak across it. Crucially, the embargo's
residual leakage is *measured*, not assumed: [`feature_autocorr_at_lag`](../src/stratval/walkforward.py)
reports the serial correlation of a feature panel at the embargo lag, so a fast feature can be
shown to decorrelate while a deliberately slow control is flagged as a control, not signal.

---

## 5. Build nulls that can fail — and check they do

A gate you trust is one you've watched *fail* on something that provably has no signal. The
[`nulls`](../src/stratval/nulls.py) module manufactures these:

- **within-event shuffle** — permute the score within each event, destroying the
  score↔label link while preserving every base rate and class balance. A correct within-event
  metric must collapse to ~0.5.
- **anti-signal** — negate the score; the metric must invert below 0.5 (a cheap sign-flip
  check).
- **label shuffle** — the classic permutation null.

The [`robustness`](../src/stratval/robustness.py) suite runs these as explicit **negative
controls** alongside two more stress tests: **leave-one-block-out** (no single month is
carrying the result) and a **power-injection curve** (how much margin the result has to its
gate threshold as you subsample events). A result worth claiming survives all of them.

---

## 6. Control the family-wide error rate

Run twenty gates at α = 0.05 on noise and you expect one "discovery". A research program that
probes many hypotheses must control the family-wide error rate, not each test in isolation.
[`multiple_comparison`](../src/stratval/multiple_comparison.py) provides Bonferroni (controls
the chance of *any* false positive; conservative) and Benjamini-Hochberg (controls the
*fraction* of claimed discoveries that are false; the better default when several real effects
are expected). A claim should be reported as "survived FDR control", not just "p < 0.05".

---

## 7. Validate the mechanism, not the P&L

A levered, single-factor clone of a strategy will track its returns almost perfectly while
sharing none of its decision logic — so a high return correlation is *not* evidence you
recovered the mechanism. Whenever the question is "did I recover how it decides?", the
evidence has to live in the **decision-level** metrics above (within-event direction, the
harvest separation, the negative controls), and any P&L tracking is reported as explicitly
non-probative, with a no-mechanism null shown next to it.

---

## 8. Deflate the Sharpe for the number of trials

Lessons 1–7 protect a *single* decision-level test. The complementary failure happens one
level up, at strategy *selection*: you search many configurations, keep the best Sharpe, and
report it as if it were the only one you ran. The expected maximum of N independent Sharpes
grows with N even when no strategy has any edge, so the winner of a wide search is inflated by
construction. A 1.1-annualized-Sharpe "discovery" picked from fifty noise trials is exactly
what luck produces.

Two standard tools, in [`deflated_sharpe`](../src/stratval/deflated_sharpe.py), handle this in
Sharpe space rather than the p-value space of lesson 6:

- **Probabilistic Sharpe Ratio (PSR)** corrects the Sharpe's sampling error for sample length
  *and* for skew/fat tails — the return shape of most real strategies inflates a naive Sharpe's
  confidence. But PSR alone still ignores how many trials you ran.
- **Deflated Sharpe Ratio (DSR)** is the PSR evaluated against a benchmark `SR0` that is the
  *expected maximum* Sharpe under the null given N trials and the spread of their Sharpes. The
  demo makes the trap concrete: the best of 50 pure-noise trials shows a naive PSR of ~0.98
  (looks shippable) yet a DSR of ~0.47 (correctly rejected), while a single genuine edge clears
  DSR ≥ 0.95 even after deflating for all 50 trials.
- **Probability of Backtest Overfitting (PBO)**, via combinatorially-symmetric
  cross-validation, asks the selection procedure itself a question: across every balanced
  in-sample/out-of-sample split, how often does the in-sample-best configuration land below the
  OOS median? PBO ≈ 0.5 means the search has no out-of-sample value (the lucky winner's healthy
  in-sample Sharpe collapses to ~0 OOS); PBO ≈ 0 means the winner generalizes.

The discipline is the same one as lesson 6, in the currency quant research actually reports: a
Sharpe means nothing without stating how many configurations were searched to find it.
See [`examples/demo_overfitting_sharpe.py`](../examples/demo_overfitting_sharpe.py).

---

## Where this sits — related work

None of this is invented here; the value is in having a single, tested harness that applies all
of it. The lineage worth knowing:

- **Walk-forward + embargo (lesson 4)** is the practitioner form of the *purging and embargoing*
  in Marcos López de Prado, *Advances in Financial Machine Learning* (2018) — the standard
  reference for leakage-free cross-validation of trading strategies.
- **DSR / PSR (lesson 8)** are from Bailey & López de Prado, "The Sharpe Ratio Efficient
  Frontier" (2012) and "The Deflated Sharpe Ratio" (2014). **PBO/CSCV** is Bailey, Borwein,
  López de Prado & Zhu, "The Probability of Backtest Overfitting" (2017).
- **Family-wise / FDR control (lesson 6)** is the classical multiple-testing literature
  (Bonferroni; Benjamini & Hochberg, 1995). For the returns-space analogue of "did the best of
  many strategies beat chance," see White's Reality Check (2000) and Hansen's SPA test (2005) —
  not implemented here, but the same question DSR answers via extreme-value theory.

---

## The discipline in one paragraph

Pre-commit the decision rule. Cluster the bootstrap on the shared-signal unit. Lead on the
period-stripped (within-event) statistic and treat the pooled number as a reference only. Test
forward with an embargo whose leakage you measure. Build nulls that can fail and confirm they
do. Control the family-wide error rate — and, in Sharpe space, deflate the winner for the
number of trials you ran (DSR) and check that the selection generalizes (PBO). And remember
that tracking the P&L is not the same as recovering the mechanism. None of it is exotic; the
value is in applying all of it, every time, *before* the number can talk you out of it.
