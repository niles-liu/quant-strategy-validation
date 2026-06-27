"""Gate-first protocol — make the ship/reject decision a committed object, not a story told
after the numbers land.

The single most effective guard against fooling yourself on noisy financial data is to write
the decision rule down *before* you compute the statistic: name the metric, the threshold,
and what each outcome means, then refuse to move the goalposts once results arrive. This
module turns that discipline into code. You construct a :class:`Gate` with a pre-committed
metric function and threshold, lock it, and only then evaluate it. The evaluation records the
observed value and a pass/fail verdict against the *pre-registered* rule — there is no API to
change the threshold after evaluation, which is exactly the point.

A gate is intentionally tiny and self-contained; the value is the workflow it enforces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

__all__ = ["Gate", "GateResult"]


@dataclass(frozen=True)
class GateResult:
    """The outcome of evaluating a pre-registered gate."""

    name: str
    statistic: str
    observed: float
    threshold: float
    direction: str
    passed: bool
    interpretation: str

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        rel = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<"}[self.direction]
        return (f"[{verdict}] {self.name}: {self.statistic} = {self.observed:.4f} "
                f"(gate: {rel} {self.threshold}) -> {self.interpretation}")


@dataclass(frozen=True)
class Gate:
    """A pre-registered ship/reject rule.

    Parameters
    ----------
    name : human-readable gate id (e.g. "G1-within-event-direction").
    statistic : which scalar from the metric output the gate reads (a key into the dict the
        metric returns, e.g. "point" or "p_le_050").
    threshold : the pre-committed bar.
    direction : one of "ge" / "le" / "gt" / "lt" — how ``observed`` must compare to
        ``threshold`` to PASS.
    metric : a callable returning a dict of named scalars (e.g.
        :func:`stratval.within_event.within_event_auc`).
    pass_interp / fail_interp : the pre-committed reading of each outcome. Writing both down
        in advance is what stops a post-hoc reinterpretation of a failure as a success.
    """

    name: str
    statistic: str
    threshold: float
    direction: str
    metric: Callable[..., dict]
    pass_interp: str = "gate passed"
    fail_interp: str = "gate failed"
    _cmp: dict = field(default_factory=lambda: {
        "ge": lambda o, t: o >= t,
        "le": lambda o, t: o <= t,
        "gt": lambda o, t: o > t,
        "lt": lambda o, t: o < t,
    }, repr=False)

    def __post_init__(self):
        if self.direction not in ("ge", "le", "gt", "lt"):
            raise ValueError(f"direction must be ge/le/gt/lt, got {self.direction!r}")

    def evaluate(self, *args, **kwargs) -> GateResult:
        """Run the pre-committed metric and judge it against the locked threshold."""
        out = self.metric(*args, **kwargs)
        if self.statistic not in out:
            raise KeyError(
                f"metric returned keys {list(out)}; gate reads {self.statistic!r}")
        observed = float(out[self.statistic])
        passed = bool(self._cmp[self.direction](observed, self.threshold))
        return GateResult(
            name=self.name,
            statistic=self.statistic,
            observed=observed,
            threshold=self.threshold,
            direction=self.direction,
            passed=passed,
            interpretation=self.pass_interp if passed else self.fail_interp,
        )
