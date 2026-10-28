"""The common event record produced by every detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TransitEvent:
    """One transit-like dip (a *potential* transit, not a confirmed one).

    Attributes:
        event_id: short label, e.g. "T3" (threshold) or "B7" (BLS epoch).
        method: "threshold" or "bls".
        t_center: mid-event time (days, same time system as the light curve).
        t_start / t_end: first / last in-event times (days).
        depth: fractional flux decrease (0.01 = 1%).
        duration: event length in days.
        snr: depth / noise × sqrt(number of in-event points).
        n_points: number of measurements inside the event.
        epoch: integer cycle number when a period is known.
    """

    event_id: str
    method: str
    t_center: float
    t_start: float
    t_end: float
    depth: float
    duration: float
    snr: float
    n_points: int
    epoch: int | None = None

    @property
    def depth_ppm(self) -> float:
        return self.depth * 1e6

    def to_dict(self) -> dict:
        d = asdict(self)
        d["depth_ppm"] = self.depth_ppm
        d["duration_hours"] = self.duration * 24
        return d
