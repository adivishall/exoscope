"""Synthetic light curves with known injected transits (teaching and testing).

Every dataset produced here carries ``meta["source"] = "synthetic"`` and a
``truth`` dictionary with the injected parameters and transit epochs, which
lets ExoScope compute precision, recall and period error exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model import LightCurveData


@dataclass
class SyntheticParams:
    period_days: float = 3.0
    depth_ppm: float = 5000.0
    duration_hours: float = 3.0
    t0_days: float = 1.2
    baseline_days: float = 27.0
    cadence_minutes: float = 30.0
    noise_ppm: float = 600.0
    variability_amp_ppm: float = 2000.0  # slow stellar variability that detrending must remove
    variability_period_days: float = 6.0
    n_gaps: int = 1  # data gaps, e.g. spacecraft downlinks
    gap_length_days: float = 1.5
    missing_fraction: float = 0.01  # random NaN cadences
    outlier_fraction: float = 0.002  # positive outliers (cosmic rays / flares)
    ingress_fraction: float = 0.15  # trapezoid shape: share of duration spent in ingress+egress
    odd_depth_ratio: float = 1.0  # ≠ 1 mimics an eclipsing binary (odd eclipses deeper/shallower)
    seed: int = 42

    def validate(self) -> None:
        checks = [
            (0.2 <= self.period_days <= 100, "period must be between 0.2 and 100 days"),
            (0 < self.depth_ppm <= 100_000, "depth must be between 0 and 100,000 ppm"),
            (0.25 <= self.duration_hours <= 24, "duration must be between 0.25 and 24 hours"),
            (self.duration_hours / 24 < self.period_days / 2, "duration must be shorter than half the period"),
            (1 <= self.baseline_days <= 400, "baseline must be between 1 and 400 days"),
            (1 <= self.cadence_minutes <= 60, "cadence must be between 1 and 60 minutes"),
            (0 <= self.noise_ppm <= 50_000, "noise must be between 0 and 50,000 ppm"),
            (self.baseline_days * 1440 / self.cadence_minutes <= 400_000, "too many points; increase cadence"),
            (0 <= self.missing_fraction < 0.5, "missing fraction must be in [0, 0.5)"),
            (0 <= self.outlier_fraction < 0.1, "outlier fraction must be in [0, 0.1)"),
            (0 <= self.ingress_fraction <= 0.5, "ingress fraction must be in [0, 0.5]"),
            (0 <= self.n_gaps <= 5, "between 0 and 5 gaps"),
            (0.1 <= self.odd_depth_ratio <= 10, "odd/even depth ratio must be between 0.1 and 10"),
        ]
        errors = [msg for ok, msg in checks if not ok]
        if errors:
            raise ValueError("Invalid synthetic parameters: " + "; ".join(errors))


def transit_model(time: np.ndarray, period: float, t0: float, duration: float, depth: float,
                  ingress_fraction: float = 0.15, odd_depth_ratio: float = 1.0) -> np.ndarray:
    """Relative flux of a trapezoidal transit (1 out of transit, 1-depth at the bottom).

    ``odd_depth_ratio`` scales the depth of odd-numbered epochs, imitating an
    eclipsing binary whose primary and secondary eclipses alternate."""
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period  # distance from nearest mid-transit
    x = np.abs(phase)
    half = duration / 2
    tau = max(ingress_fraction * duration / 2, 1e-9)  # ingress duration
    epoch = np.round((time - t0) / period).astype(int)
    d = np.where(epoch % 2 == 1, depth * odd_depth_ratio, depth)
    flux = np.ones_like(time)
    full = x <= half - tau
    ramp = (x > half - tau) & (x < half)
    flux[full] -= d[full]
    flux[ramp] -= d[ramp] * (half - x[ramp]) / tau
    return flux


def generate_synthetic(params: SyntheticParams | None = None) -> LightCurveData:
    p = params or SyntheticParams()
    p.validate()
    rng = np.random.default_rng(p.seed)
    cadence = p.cadence_minutes / 1440
    time = np.arange(0.0, p.baseline_days, cadence)

    # Data gaps placed deterministically (seeded) away from the edges.
    keep = np.ones(len(time), dtype=bool)
    gaps = []
    for _ in range(p.n_gaps):
        start = rng.uniform(0.15, 0.85) * p.baseline_days
        gaps.append((start, start + p.gap_length_days))
        keep &= ~((time >= start) & (time < start + p.gap_length_days))
    time = time[keep]

    depth = p.depth_ppm * 1e-6
    duration = p.duration_hours / 24
    transit = transit_model(time, p.period_days, p.t0_days, duration, depth, p.ingress_fraction, p.odd_depth_ratio)
    variability = 1 + p.variability_amp_ppm * 1e-6 * np.sin(2 * np.pi * time / p.variability_period_days + 0.7)
    noise = rng.normal(0, p.noise_ppm * 1e-6, len(time))
    flux = transit * variability + noise

    n_out = int(round(p.outlier_fraction * len(time)))
    if n_out:
        idx = rng.choice(len(time), n_out, replace=False)
        flux[idx] += rng.uniform(5, 15, n_out) * max(p.noise_ppm, 100) * 1e-6
    n_nan = int(round(p.missing_fraction * len(time)))
    if n_nan:
        flux[rng.choice(len(time), n_nan, replace=False)] = np.nan

    flux *= 12_000.0  # arbitrary "electrons per second" scale, like real photometry
    flux_err = np.full(len(time), max(p.noise_ppm, 1) * 1e-6 * 12_000.0)

    # Ground truth: every mid-transit time inside the baseline, and whether it was observed.
    n_first = int(np.ceil((time.min() - p.t0_days) / p.period_days))
    n_last = int(np.floor((time.max() - p.t0_days) / p.period_days))
    epochs = []
    for n in range(n_first, n_last + 1):
        tc = p.t0_days + n * p.period_days
        in_tr = np.abs(time - tc) < duration / 2
        epochs.append({"epoch": n, "t_center": tc, "observed": bool(np.isfinite(flux[in_tr]).sum() >= 1)})

    truth = {**asdict(p), "duration_days": duration, "depth": depth, "epochs": epochs, "gaps": gaps}
    meta = {
        "target": "Synthetic star",
        "mission": "Synthetic",
        "dataset": f"P={p.period_days:g} d, depth={p.depth_ppm:g} ppm, noise={p.noise_ppm:g} ppm, seed={p.seed}",
        "source": "synthetic",
        "time_format": "days",
        "flux_unit": "arbitrary",
        "exptime_s": p.cadence_minutes * 60,
        "synthetic_params": asdict(p),  # enough to regenerate the identical dataset
    }
    return LightCurveData(time=time, flux=flux, flux_err=flux_err, meta=meta, truth=truth)
