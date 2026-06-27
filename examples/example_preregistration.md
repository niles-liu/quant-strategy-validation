# Pre-registration — synthetic "cut-underwater" direction signal

> A worked example of the gate-first protocol the harness enforces. Written **before** any
> statistic is computed; the gates below are locked and `examples/demo_harvest_signal.py`
> evaluates exactly these — no goalpost moves once the numbers land.
>
> Data: `stratval.generate_rebalance_panel` (synthetic). No real account or holdings.

## 1. Question

A stylized systematic strategy makes an add/trim decision for each held name at each
rebalance event. Does an **unrealized-return-vs-basis** signal (`unreal_ret`) drive the
direction of that decision — i.e. is the strategy cutting underwater names (a
tax-loss-harvesting-style rule) — and can we establish this *without* being fooled by the
slow regime drift baked into the data?

## 2. Data & grain

- **Grain:** one row per (rebalance event × held name). The **cluster unit is the event** —
  all names in an event share one signal snapshot, so inference must cluster on `event_id`.
- **Label:** `label = 1` for add, `0` for trim.
- **Signal under test:** `unreal_ret` (higher → add).
- **Confound (planted):** `regime_clock` — constant within an event, drifts across events,
  shifts each event's overall add-rate. Predictive *pooled*, carries no *within-event*
  information.
- **Collinear cousin:** `momentum` — correlated with `unreal_ret`, weak direct effect.

## 3. Out-of-sample policy

Walk-forward only. Expanding training window; the **forward (later) half of the monthly
blocks is the OOS read**. A 5-event **embargo** truncates training before each test block to
break the slow-feature seam. Random K-fold is disallowed (it trains on the future).

## 4. Baselines & nulls (decided in advance)

- **Within-event shuffle** of `unreal_ret` → AUC must be ≈ 0.50 (the gate must not pass).
- **Anti-signal** (negated `unreal_ret`) → AUC must invert (< 0.50).
- **Collinear baseline:** `momentum` — the signal must *beat* it on a paired clustered test.

## 5. Gates (locked)

| Gate | Statistic | Rule | Pass interpretation |
|---|---|---|---|
| **G1 (primary)** | within-event AUC of `unreal_ret`, `P(AUC ≤ 0.50)` | `≤ 0.05` | direction signal reproduces within-event |
| **G2** | paired clustered Δ = AUC(`unreal_ret`) − AUC(`momentum`), `P(Δ ≤ 0)` | `≤ 0.05` | signal beats its collinear cousin |
| **G3 (falsification)** | within-event AUC of shuffled signal | `≈ 0.50`, must **not** pass G1's rule | gate has teeth |
| **G4 (confound)** | within-event AUC of `regime_clock` | `≈ 0.50` | pooled-only "skill" is an artifact |

**Family-wise control:** the family of within-event tests is subjected to Benjamini-Hochberg
FDR control at α = 0.05; a claim must survive it.

**Robustness (no verdict authority, but required before claiming):** leave-one-block-out (no
single month carries G1) and a power-injection curve (margin to the G1 threshold).

## 6. Decision tree (pre-committed)

- **G1 passes AND survives FDR AND G3/G4 behave** ⇒ "the cut-underwater direction signal
  reproduces within-event; the regime drift is correctly demoted." Report the within-event
  AUC with its event count and CI; note it is modest and single-process.
- **G1 fails / under-powered** ⇒ report honestly as not established in this sample; do **not**
  fall back on the (confounded) pooled AUC.
- The pooled AUC is a **reference only** throughout — it contains the regime base-rate channel
  and has no verdict authority.
