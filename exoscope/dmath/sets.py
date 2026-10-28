"""Set theory for comparing detections.

The two detectors produce different kinds of objects: the threshold method
returns individual *dips*; BLS returns one *periodic signal*. To compare like
with like, the BLS signal is expanded into its individual predicted transit
*epochs*, and each epoch is an event. Events from the two methods are then
matched one-to-one when their mid-times differ by at most a tolerance τ.
Each matched pair (or unmatched event) becomes one element of the universe.

Definitions used throughout ExoScope:

    A = { events found by the threshold detector }
    B = { BLS-predicted transit epochs that contain data }
    G = { BLS-predicted epochs that fall in data gaps (unobservable) }
    U = A ∪ B ∪ G          (the universe of evaluated epochs)

So the complement (A ∪ B)ᶜ = U − (A ∪ B) = G: epochs where a transit was
expected but nothing could be observed, and Aᶜ ∩ B = B − A: transits that
BLS predicts and the data cover, but the threshold detector missed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_events(a_times: Sequence[float], b_times: Sequence[float], tol: float) -> list[tuple[int, int, float]]:
    """Optimal one-to-one matching of two event lists within tolerance ``tol``.

    Solves an assignment problem (Hungarian algorithm) minimising the total
    |Δt| over pairs with |Δt| ≤ tol. Returns (i, j, Δt) triples.
    Each event is used at most once, so a match is a partial bijection.
    """
    a = np.asarray(a_times, dtype=float)
    b = np.asarray(b_times, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return []
    cost = np.abs(a[:, None] - b[None, :])
    big = 1e6 + cost.max()
    masked = np.where(cost <= tol, cost, big)
    rows, cols = linear_sum_assignment(masked)
    return [(int(i), int(j), float(b[j] - a[i])) for i, j in zip(rows, cols) if cost[i, j] <= tol]


@dataclass
class SetElement:
    element_id: str
    t_center: float
    threshold_id: str | None = None
    bls_id: str | None = None
    in_gap: bool = False
    dt_hours: float | None = None  # BLS − threshold time difference for matched pairs

    @property
    def label(self) -> str:
        if self.threshold_id and self.bls_id:
            return f"{self.threshold_id}≈{self.bls_id}"
        return self.threshold_id or self.bls_id or self.element_id


@dataclass
class EventSetComparison:
    elements: dict[str, SetElement]
    U: frozenset[str]
    A: frozenset[str]
    B: frozenset[str]
    tolerance_hours: float
    bls_available: bool

    # --- the operations -------------------------------------------------
    def union(self) -> frozenset[str]:
        return self.A | self.B

    def intersection(self) -> frozenset[str]:
        return self.A & self.B

    def a_minus_b(self) -> frozenset[str]:
        return self.A - self.B

    def b_minus_a(self) -> frozenset[str]:
        return self.B - self.A

    def complement(self, s: Iterable[str]) -> frozenset[str]:
        return self.U - frozenset(s)

    def symmetric_difference(self) -> frozenset[str]:
        return self.A ^ self.B

    def jaccard(self) -> float:
        """|A ∩ B| / |A ∪ B| — 1 = perfect agreement, 0 = none."""
        u = self.union()
        return len(self.intersection()) / len(u) if u else float("nan")

    def inclusion_exclusion(self) -> dict[str, int | bool]:
        """Check |A ∪ B| = |A| + |B| − |A ∩ B|."""
        lhs = len(self.union())
        rhs = len(self.A) + len(self.B) - len(self.intersection())
        return {"union": lhs, "A": len(self.A), "B": len(self.B), "intersection": len(self.intersection()),
                "holds": lhs == rhs}

    def sorted_ids(self, s: Iterable[str]) -> list[str]:
        return sorted(s, key=lambda k: self.elements[k].t_center)

    def summary(self) -> dict[str, Any]:
        n_union = len(self.union())

        def pct(x: int) -> float:
            return 100 * x / n_union if n_union else 0.0

        return {
            "|U|": len(self.U), "|A|": len(self.A), "|B|": len(self.B),
            "|A∪B|": n_union, "|A∩B|": len(self.intersection()),
            "|A−B|": len(self.a_minus_b()), "|B−A|": len(self.b_minus_a()),
            "|(A∪B)ᶜ|": len(self.complement(self.union())),
            "shared_%": pct(len(self.intersection())),
            "only_threshold_%": pct(len(self.a_minus_b())),
            "only_bls_%": pct(len(self.b_minus_a())),
            "jaccard": self.jaccard(),
        }

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for k in self.sorted_ids(self.U):
            e = self.elements[k]
            region = (
                "A ∩ B" if k in self.A and k in self.B else
                "A − B" if k in self.A else
                "B − A" if k in self.B else
                "(A ∪ B)ᶜ (data gap)"
            )
            rows.append({"element": k, "t_center": e.t_center, "threshold_event": e.threshold_id,
                         "bls_epoch": e.bls_id, "region": region,
                         "dt_hours": e.dt_hours})
        return rows


def compare_detections(threshold_events: Sequence, bls_events: Sequence | None,
                       bls_gap_times: Sequence[float] = (), tolerance_hours: float = 2.0) -> EventSetComparison:
    """Build U, A, B from detector outputs (objects with ``event_id`` and ``t_center``)."""
    if tolerance_hours <= 0:
        raise ValueError("matching tolerance must be positive")
    tol = tolerance_hours / 24
    bls_events = list(bls_events or [])
    a_times = [e.t_center for e in threshold_events]
    b_times = [e.t_center for e in bls_events]
    pairs = match_events(a_times, b_times, tol)
    matched_a = {i for i, _, _ in pairs}
    matched_b = {j for _, j, _ in pairs}

    raw: list[SetElement] = []
    for i, j, dt in pairs:
        ta, tb = threshold_events[i], bls_events[j]
        raw.append(SetElement("", (ta.t_center + tb.t_center) / 2, ta.event_id, tb.event_id, dt_hours=dt * 24))
    for i, e in enumerate(threshold_events):
        if i not in matched_a:
            raw.append(SetElement("", e.t_center, threshold_id=e.event_id))
    for j, e in enumerate(bls_events):
        if j not in matched_b:
            raw.append(SetElement("", e.t_center, bls_id=e.event_id))
    for t in bls_gap_times:
        raw.append(SetElement("", float(t), in_gap=True))

    raw.sort(key=lambda e: e.t_center)
    elements = {}
    for k, e in enumerate(raw, start=1):
        e.element_id = f"e{k}"
        elements[e.element_id] = e
    A = frozenset(k for k, e in elements.items() if e.threshold_id)
    B = frozenset(k for k, e in elements.items() if e.bls_id)
    return EventSetComparison(elements, frozenset(elements), A, B, tolerance_hours, bls_available=bool(bls_events))


def detection_metrics(detected_times: Sequence[float], true_times: Sequence[float], tol: float) -> dict[str, float]:
    """Precision / recall / F1 from set operations against ground truth.

    With D = detections and T = true transits (matched within tol):
    TP = |D ∩ T|, FP = |D − T|, FN = |T − D|.
    """
    pairs = match_events(detected_times, true_times, tol)
    tp = len(pairs)
    fp = len(detected_times) - tp
    fn = len(true_times) - tp
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def roster(ids: Iterable[str], comparison: EventSetComparison | None = None, limit: int = 12) -> str:
    """Roster notation, e.g. {T1≈B1, T2, B4, …}."""
    ids = list(ids)
    if comparison is not None:
        ids = comparison.sorted_ids(ids)
        names = [comparison.elements[k].label for k in ids]
    else:
        names = ids
    if not names:
        return "∅"
    shown = ", ".join(names[:limit])
    return "{" + shown + (f", … (+{len(names) - limit})" if len(names) > limit else "") + "}"
