"""Improved threshold-based dip detection.

The original ExoScope flagged every *point* below ``mean − 3·std`` and took
the median spacing of flagged points as the period. That approach has three
problems, each fixed here:

* mean/std are pulled down by the transits themselves → use the median and a
  robust (MAD-based) noise estimate instead;
* single noisy points become "events" → group consecutive flagged points into
  runs and require ``min_points`` per event;
* one missed transit doubles a spacing and corrupts a median → estimate the
  period with the approximate-GCD / congruence method from number theory,
  which tolerates missed and spurious events.
"""

from __future__ import annotations

import time as _time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..dmath.number_theory import PeriodEstimate, estimate_period
from ..processing import ProcessedLightCurve, robust_std
from .events import TransitEvent

ALGORITHM_VERSION = "threshold-2.0"


@dataclass
class ThresholdSettings:
    sigma_threshold: float = 3.0  # flag points below median − k·σ
    min_points: int = 2  # minimum consecutive flagged points for an event
    max_gap_points: int = 1  # unflagged points allowed inside one event
    period_tolerance_hours: float = 2.0  # ε for the approximate-GCD period estimate

    def validate(self) -> None:
        errors = []
        if not 1.0 <= self.sigma_threshold <= 10.0:
            errors.append("threshold must be between 1 and 10 sigma")
        if not 1 <= self.min_points <= 50:
            errors.append("minimum points per event must be between 1 and 50")
        if not 0 <= self.max_gap_points <= 10:
            errors.append("allowed gap must be between 0 and 10 points")
        if not 0.1 <= self.period_tolerance_hours <= 24:
            errors.append("period tolerance must be between 0.1 and 24 hours")
        if errors:
            raise ValueError("Invalid threshold settings: " + "; ".join(errors))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThresholdResult:
    events: list[TransitEvent]
    threshold_level: float  # relative-flux level below which points are flagged
    noise: float
    n_flagged_points: int
    period: PeriodEstimate | None
    runtime_s: float
    settings: ThresholdSettings
    rejected_runs: int = 0  # flagged runs shorter than min_points
    version: str = ALGORITHM_VERSION
    flagged_mask: np.ndarray = field(default=None, repr=False)

    @property
    def event_times(self) -> np.ndarray:
        return np.array([e.t_center for e in self.events])


def _group_runs(idx: np.ndarray, time: np.ndarray, max_gap_points: int, cadence: float) -> list[np.ndarray]:
    """Split sorted flagged indices into runs of (nearly) consecutive cadences."""
    if len(idx) == 0:
        return []
    groups, current = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        same_run = (b - a) <= max_gap_points + 1 and (time[b] - time[a]) <= (max_gap_points + 1.5) * cadence
        if same_run:
            current.append(b)
        else:
            groups.append(np.array(current))
            current = [b]
    groups.append(np.array(current))
    return groups


def detect_threshold(lc: ProcessedLightCurve, settings: ThresholdSettings | None = None) -> ThresholdResult:
    s = settings or ThresholdSettings()
    s.validate()
    start = _time.perf_counter()
    t, f = lc.time, lc.flux
    baseline = float(np.median(f))
    noise = robust_std(f)
    level = baseline - s.sigma_threshold * noise
    flagged = f < level
    groups = _group_runs(np.where(flagged)[0], t, s.max_gap_points, lc.cadence_days)

    events: list[TransitEvent] = []
    rejected = 0
    for g in groups:
        if len(g) < s.min_points:
            rejected += 1
            continue
        seg = slice(g[0], g[-1] + 1)  # include unflagged points inside the run
        tt, ff = t[seg], f[seg]
        deficit = np.clip(baseline - ff, 0, None)
        center = float(np.sum(tt * deficit) / deficit.sum()) if deficit.sum() > 0 else float(tt.mean())
        depth = float(baseline - ff.mean())
        n = len(ff)
        events.append(
            TransitEvent(
                event_id=f"T{len(events) + 1}",
                method="threshold",
                t_center=center,
                t_start=float(tt[0]),
                t_end=float(tt[-1]),
                depth=depth,
                duration=float(tt[-1] - tt[0] + lc.cadence_days),
                snr=float(depth / noise * np.sqrt(n)) if noise > 0 else float("inf"),
                n_points=n,
            )
        )

    period = None
    if len(events) >= 2:
        period = estimate_period(np.array([e.t_center for e in events]), tol=s.period_tolerance_hours / 24)
        if period is not None and period.members and not period.ambiguous:
            for k, e_idx in enumerate(period.members):
                events[e_idx].epoch = period.epochs[k]

    return ThresholdResult(
        events=events,
        threshold_level=float(level),
        noise=float(noise),
        n_flagged_points=int(flagged.sum()),
        period=period,
        runtime_s=_time.perf_counter() - start,
        settings=s,
        rejected_runs=rejected,
        flagged_mask=flagged,
    )
