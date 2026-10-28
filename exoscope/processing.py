"""Light-curve preprocessing: cleaning, normalisation, detrending, outlier removal.

Order of operations (each step is logged so the UI and report can explain it):

1. Drop cadences with missing (NaN) time or flux.
2. Drop cadences flagged by the mission's quality flags (optional).
3. Normalise: divide by the median so the out-of-transit level is 1.0.
4. Detrend ("flatten"): divide by a Savitzky–Golay trend fitted with
   iterative sigma-masking, so slow stellar variability is removed while
   short transit dips survive. The window must be several times longer than
   a transit, otherwise the trend follows the dip and erases it.
5. Sigma-clip outliers. By default only *upward* outliers (cosmic rays,
   flares) are removed — clipping downward points would delete transits.
6. Optional binning to a coarser cadence (speeds up BLS on 2-minute data).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from .data.model import DataError, LightCurveData

MIN_POINTS = 20


def robust_std(x: np.ndarray) -> float:
    """1.4826 × median absolute deviation: a standard-deviation estimate that
    is barely affected by transits and outliers."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


@dataclass
class PreprocessSettings:
    remove_quality_flagged: bool = True
    flatten: bool = True
    flatten_window_days: float = 1.0
    flatten_polyorder: int = 2
    clip_upper_sigma: float | None = 4.0
    clip_lower_sigma: float | None = None  # off by default: would remove transits
    bin_minutes: float | None = None

    def validate(self) -> None:
        errors = []
        if not 0.1 <= self.flatten_window_days <= 10:
            errors.append("flatten window must be between 0.1 and 10 days")
        if self.flatten_polyorder not in (1, 2, 3):
            errors.append("polynomial order must be 1, 2 or 3")
        for name, v in (("upper clip", self.clip_upper_sigma), ("lower clip", self.clip_lower_sigma)):
            if v is not None and not 2 <= v <= 20:
                errors.append(f"{name} must be between 2 and 20 sigma (or off)")
        if self.bin_minutes is not None and not 1 <= self.bin_minutes <= 120:
            errors.append("bin size must be between 1 and 120 minutes (or off)")
        if errors:
            raise ValueError("Invalid preprocessing settings: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedLightCurve:
    time: np.ndarray
    flux: np.ndarray  # normalised + detrended (+ clipped, + binned)
    flux_err: np.ndarray
    noise: float  # robust std of the final flux
    cadence_days: float
    # Intermediate stage for plotting: normalised flux and the fitted trend.
    norm_time: np.ndarray = field(repr=False, default=None)
    norm_flux: np.ndarray = field(repr=False, default=None)
    trend: np.ndarray = field(repr=False, default=None)
    clipped_time: np.ndarray = field(repr=False, default=None)
    clipped_flux: np.ndarray = field(repr=False, default=None)
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def baseline_days(self) -> float:
        return float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0

    @property
    def n_points(self) -> int:
        return int(len(self.time))


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def savgol_trend(time: np.ndarray, flux: np.ndarray, window_points: int, polyorder: int,
                 break_tolerance: float = 5.0, sigma: float = 3.0, niters: int = 3) -> np.ndarray:
    """Fit a smooth trend segment by segment (segments split at data gaps).

    Points deviating by more than ``sigma`` robust standard deviations are
    masked and replaced by interpolation before refitting, so transits and
    outliers do not drag the trend downwards.
    """
    trend = np.empty_like(flux)
    cadence = np.median(np.diff(time))
    breaks = np.where(np.diff(time) > break_tolerance * cadence)[0] + 1
    for seg in np.split(np.arange(len(time)), breaks):
        t, f = time[seg], flux[seg]
        w = min(_odd(window_points), len(seg) if len(seg) % 2 else len(seg) - 1)
        if w <= polyorder + 1:
            trend[seg] = np.median(f)
            continue
        mask = np.ones(len(seg), dtype=bool)
        tr = np.full(len(seg), np.median(f))
        for _ in range(niters):
            ff = f.copy()
            if (~mask).any() and mask.sum() >= 2:
                ff[~mask] = np.interp(t[~mask], t[mask], f[mask])
            tr = savgol_filter(ff, w, polyorder, mode="interp")
            resid = f - tr
            s = robust_std(resid[mask])
            if not np.isfinite(s) or s == 0:
                break
            new_mask = np.abs(resid) < sigma * s
            if new_mask.sum() < polyorder + 2 or np.array_equal(new_mask, mask):
                break
            mask = new_mask
        trend[seg] = tr
    return trend


def bin_lightcurve(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, bin_days: float):
    edges = np.arange(time[0], time[-1] + bin_days, bin_days)
    idx = np.digitize(time, edges) - 1
    counts = np.bincount(idx, minlength=len(edges))
    ok = counts > 0
    sums = np.bincount(idx, weights=time, minlength=len(edges))
    fsum = np.bincount(idx, weights=flux, minlength=len(edges))
    esum = np.bincount(idx, weights=flux_err**2, minlength=len(edges))
    n = counts[ok]
    return sums[ok] / n, fsum[ok] / n, np.sqrt(esum[ok]) / n


def preprocess(data: LightCurveData, settings: PreprocessSettings | None = None) -> ProcessedLightCurve:
    s = settings or PreprocessSettings()
    s.validate()
    steps: list[dict[str, Any]] = []

    def log(step: str, detail: str, before: int, after: int) -> None:
        steps.append({"step": step, "detail": detail, "points_before": before, "points_after": after})

    n0 = len(data.time)
    if n0 == 0:
        raise DataError("The light curve is empty.")
    order = np.argsort(data.time)
    t, f = data.time[order], data.flux[order]
    e = data.flux_err[order] if data.flux_err is not None else np.full(n0, np.nan)
    q = data.quality_ok[order]

    finite = np.isfinite(t) & np.isfinite(f)
    t, f, e, q = t[finite], f[finite], e[finite], q[finite]
    log("Remove missing values", f"dropped {n0 - len(t)} cadences with NaN time or flux", n0, len(t))

    if s.remove_quality_flagged:
        n1 = len(t)
        t, f, e = t[q], f[q], e[q]
        log("Quality mask", f"dropped {n1 - len(t)} cadences flagged by the mission pipeline", n1, len(t))

    if len(t) < MIN_POINTS:
        raise DataError(f"Only {len(t)} usable points remain; at least {MIN_POINTS} are needed.")

    # Remove duplicate timestamps so time -> flux is a function (see Relations module).
    t, uniq = np.unique(t, return_index=True)
    f, e = f[uniq], e[uniq]

    med = np.median(f)
    if not np.isfinite(med) or med == 0:
        raise DataError("Median flux is zero or undefined; cannot normalise.")
    f = f / med
    e = np.where(np.isfinite(e), e / abs(med), robust_std(f))
    log("Normalise", f"divided by median flux {med:.6g} so the baseline is 1.0", len(t), len(t))

    cadence = float(np.median(np.diff(t)))
    norm_t, norm_f = t.copy(), f.copy()
    if s.flatten:
        window_pts = max(int(round(s.flatten_window_days / cadence)), s.flatten_polyorder + 2)
        trend = savgol_trend(t, f, _odd(window_pts), s.flatten_polyorder)
        f = f / trend
        e = e / trend
        log("Detrend (flatten)", f"Savitzky–Golay trend, window {s.flatten_window_days:g} d "
            f"({_odd(window_pts)} points), polynomial order {s.flatten_polyorder}", len(t), len(t))
    else:
        trend = np.ones_like(f)

    clipped = np.zeros(len(t), dtype=bool)
    sigma = robust_std(f)
    center = np.median(f)
    if s.clip_upper_sigma is not None:
        clipped |= f > center + s.clip_upper_sigma * sigma
    if s.clip_lower_sigma is not None:
        clipped |= f < center - s.clip_lower_sigma * sigma
    if s.clip_upper_sigma is not None or s.clip_lower_sigma is not None:
        desc = []
        if s.clip_upper_sigma is not None:
            desc.append(f"above +{s.clip_upper_sigma:g}σ")
        if s.clip_lower_sigma is not None:
            desc.append(f"below −{s.clip_lower_sigma:g}σ (warning: can remove transit points)")
        log("Sigma-clip outliers", f"removed {int(clipped.sum())} points {' and '.join(desc)}",
            len(t), int((~clipped).sum()))
    clipped_t, clipped_f = t[clipped], f[clipped]
    t, f, e = t[~clipped], f[~clipped], e[~clipped]

    if s.bin_minutes is not None and s.bin_minutes / 1440 > cadence * 1.01:
        n1 = len(t)
        t, f, e = bin_lightcurve(t, f, e, s.bin_minutes / 1440)
        cadence = float(np.median(np.diff(t)))
        log("Bin", f"averaged into {s.bin_minutes:g}-minute bins", n1, len(t))

    if len(t) < MIN_POINTS:
        raise DataError(f"Only {len(t)} points remain after preprocessing; relax the settings.")

    return ProcessedLightCurve(
        time=t, flux=f, flux_err=e, noise=robust_std(f), cadence_days=cadence,
        norm_time=norm_t, norm_flux=norm_f, trend=trend,
        clipped_time=clipped_t, clipped_flux=clipped_f, steps=steps,
    )
