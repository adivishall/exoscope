"""Injection–recovery: add simulated transits to a light curve and test whether they are found.

This is the standard way to measure a pipeline's real detection limits
(see e.g. Giacobbe et al. 2012; Hippke & Heller 2019): injecting into *real*
data keeps the star's correlated (red) noise and the instrument's systematics,
which synthetic white noise does not have.

Data with injected transits are labelled ``source = "injected"`` — they are no
longer a pure real observation and must never be presented as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.model import LightCurveData
from .data.synthetic import transit_model
from .processing import robust_std


def mask_transits(data: LightCurveData, period: float, t0: float, duration: float, factor: float = 2.0) -> LightCurveData:
    """Remove the cadences within ``factor × duration`` of a known signal's transits."""
    if period <= 0 or duration <= 0:
        raise ValueError("period and duration must be positive")
    phase = np.abs(np.mod(data.time - t0 + 0.5 * period, period) - 0.5 * period)
    keep = phase > factor * duration / 2
    meta = {**data.meta, "masked_signal": {"period": period, "t0": t0, "duration": duration, "factor": factor}}
    return LightCurveData(data.time[keep], data.flux[keep],
                          None if data.flux_err is None else data.flux_err[keep], data.quality_ok[keep], meta)


@dataclass
class Injection:
    period: float
    t0: float
    depth: float  # fractional
    duration: float  # days


def inject(data: LightCurveData, inj: Injection) -> LightCurveData:
    """Multiply the flux by a trapezoidal transit model (keeps the original noise)."""
    model = transit_model(data.time, inj.period, inj.t0, inj.duration, inj.depth)
    meta = {**data.meta, "source": "injected", "host_source": data.meta.get("source"),
            "injected": {"period": inj.period, "t0": inj.t0, "depth": inj.depth, "duration": inj.duration}}
    return LightCurveData(data.time, data.flux * model, data.flux_err, data.quality_ok, meta)


def white_noise_twin(data: LightCurveData, seed: int = 0) -> LightCurveData:
    """Same timestamps and the same robust scatter as ``data``, but Gaussian white noise
    and no stellar signal — the idealised case the real data are compared against."""
    good = np.isfinite(data.flux) & data.quality_ok
    rel = data.flux[good] / np.nanmedian(data.flux[good])
    sigma = 1.4826 * np.median(np.abs(np.diff(rel))) / np.sqrt(2)  # point-to-point scatter
    rng = np.random.default_rng(seed)
    t = data.time[good]
    meta = {"target": f"white-noise twin of {data.target}", "source": "synthetic", "time_format": data.time_format,
            "dataset": data.meta.get("dataset", ""), "twin_sigma": float(sigma)}
    return LightCurveData(t, 1 + rng.normal(0, sigma, len(t)), np.full(len(t), sigma), None, meta)


def residual_scatter(flux: np.ndarray) -> float:
    return robust_std(flux)
