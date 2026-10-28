"""Box Least Squares (BLS) periodic transit search using Astropy.

BLS fits a periodic box-shaped dip (period P, mid-time t0, duration D,
depth δ) to the light curve over a grid of trial periods and reports the
best fit at each period (the *periodogram*). Because it combines every
transit coherently, it finds signals far below the per-point noise.

After the strongest signal is found, its in-transit points can be masked and
the search repeated to look for additional periodic signals.
"""

from __future__ import annotations

import time as _time
import warnings
from dataclasses import asdict, dataclass, field

import astropy
import numpy as np
from astropy.timeseries import BoxLeastSquares

from ..processing import ProcessedLightCurve
from .events import TransitEvent

ALGORITHM_VERSION = f"bls-astropy-{astropy.__version__}"


@dataclass
class BLSSettings:
    period_min: float = 0.5  # days
    period_max: float | None = None  # days; default = baseline / 2 (≥ 2 transits)
    durations_hours: tuple[float, ...] = (1.0, 2.0, 3.0, 4.5, 6.0)
    frequency_factor: float = 1.0  # < 1 = finer grid, > 1 = coarser
    max_periods: int = 30_000  # cap on grid size to keep runtime bounded
    n_signals: int = 1  # how many signals to search for (iterative masking)
    objective: str = "likelihood"  # or "snr"
    min_points_in_transit: int = 2  # an epoch counts as observed with this many in-transit points

    def validate(self, baseline_days: float | None = None) -> None:
        errors = []
        if not 0.1 <= self.period_min <= 100:
            errors.append("minimum period must be between 0.1 and 100 days")
        if self.period_max is not None and self.period_max <= self.period_min:
            errors.append("maximum period must exceed the minimum period")
        if not self.durations_hours:
            errors.append("at least one trial duration is required")
        elif min(self.durations_hours) <= 0 or max(self.durations_hours) / 24 >= self.period_min:
            errors.append("trial durations must be positive and shorter than the minimum period")
        if not 1_000 <= self.max_periods <= 200_000:
            errors.append("period-grid cap must be between 1,000 and 200,000")
        if not 1 <= self.n_signals <= 3:
            errors.append("number of signals must be 1, 2 or 3")
        if self.objective not in ("likelihood", "snr"):
            errors.append("objective must be 'likelihood' or 'snr'")
        if not 0.2 <= self.frequency_factor <= 5:
            errors.append("frequency factor must be between 0.2 and 5")
        if baseline_days is not None and self.period_min * 2 > baseline_days:
            errors.append(f"minimum period must be at most half the baseline ({baseline_days / 2:.2f} d)")
        if errors:
            raise ValueError("Invalid BLS settings: " + "; ".join(errors))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["durations_hours"] = list(self.durations_hours)
        return d


@dataclass
class BLSCandidate:
    rank: int
    period: float
    t0: float
    duration: float  # days
    depth: float
    depth_err: float
    depth_snr: float
    power: float
    sde: float  # signal detection efficiency: (peak − mean) / std of the periodogram
    depth_odd: float
    depth_odd_err: float
    depth_even: float
    depth_even_err: float
    harmonic_delta_log_likelihood: float  # log L(sine) − log L(box): < 0 means the box (transit) fits better
    n_transits_expected: int
    n_transits_observed: int
    events: list[TransitEvent]
    unobserved_times: list[float]  # predicted mid-transit times that fall in data gaps
    secondary_depth: float  # mean dip at phase 0.5 (half-way between transits)
    secondary_snr: float
    periods: np.ndarray = field(repr=False)
    power_spectrum: np.ndarray = field(repr=False)

    @property
    def odd_even_sigma(self) -> float:
        """Difference between odd- and even-numbered transit depths, in σ."""
        err = np.hypot(self.depth_odd_err, self.depth_even_err)
        return float(abs(self.depth_odd - self.depth_even) / err) if err > 0 else float("nan")

    def to_dict(self) -> dict:
        return {
            "rank": self.rank, "period_days": self.period, "t0": self.t0,
            "duration_hours": self.duration * 24, "depth": self.depth, "depth_ppm": self.depth * 1e6,
            "depth_err_ppm": self.depth_err * 1e6, "depth_snr": self.depth_snr, "power": self.power,
            "sde": self.sde, "odd_even_sigma": self.odd_even_sigma,
            "harmonic_delta_log_likelihood": self.harmonic_delta_log_likelihood,
            "secondary_depth_ppm": self.secondary_depth * 1e6, "secondary_snr": self.secondary_snr,
            "n_transits_expected": self.n_transits_expected, "n_transits_observed": self.n_transits_observed,
        }


@dataclass
class BLSResult:
    candidates: list[BLSCandidate]
    runtime_s: float
    n_periods: int
    grid_capped: bool
    settings: BLSSettings
    period_max_used: float
    error_scale: float = 1.0  # factor applied to per-point errors so they match the measured scatter
    version: str = ALGORITHM_VERSION

    @property
    def best(self) -> BLSCandidate | None:
        return self.candidates[0] if self.candidates else None


def period_grid(baseline: float, pmin: float, pmax: float, min_duration: float,
                frequency_factor: float, max_periods: int) -> tuple[np.ndarray, bool]:
    """Uniform grid in frequency. A transit of duration D drifts by Δf·T² in
    phase-time over a baseline T, so Δf = factor·D/T² keeps the drift below D."""
    fmin, fmax = 1 / pmax, 1 / pmin
    df = frequency_factor * min_duration / baseline**2
    n = int(np.ceil((fmax - fmin) / df)) + 1
    capped = n > max_periods
    n = min(n, max_periods)
    freqs = np.linspace(fmin, fmax, n)
    return np.sort(1 / freqs), capped


def _epoch_events(t: np.ndarray, f: np.ndarray, noise: float, period: float, t0: float, duration: float,
                  min_points: int, rank: int) -> tuple[list[TransitEvent], list[float]]:
    """Turn the BLS ephemeris into per-epoch events (epochs with data) and the
    list of predicted times that have no data (fall in gaps)."""
    n_first = int(np.ceil((t.min() - t0 - duration / 2) / period))
    n_last = int(np.floor((t.max() - t0 + duration / 2) / period))
    events, unobserved = [], []
    baseline = np.median(f)
    for n in range(n_first, n_last + 1):
        tc = t0 + n * period
        inside = np.abs(t - tc) <= duration / 2
        k = int(inside.sum())
        if k < min_points:
            unobserved.append(float(tc))
            continue
        depth = float(baseline - f[inside].mean())
        events.append(
            TransitEvent(
                event_id=f"B{rank}.{len(events) + 1}" if rank > 1 else f"B{len(events) + 1}",
                method="bls", t_center=float(tc), t_start=float(tc - duration / 2), t_end=float(tc + duration / 2),
                depth=depth, duration=duration, snr=float(depth / noise * np.sqrt(k)) if noise > 0 else float("inf"),
                n_points=k, epoch=n,
            )
        )
    return events, unobserved


def run_bls(lc: ProcessedLightCurve, settings: BLSSettings | None = None) -> BLSResult:
    s = settings or BLSSettings()
    baseline = lc.baseline_days
    s.validate(baseline)
    start = _time.perf_counter()
    pmax = min(s.period_max or baseline / 2, baseline)
    durations = np.array(sorted(s.durations_hours)) / 24

    t, f, e = lc.time.copy(), lc.flux.copy(), lc.flux_err.copy()
    e = np.where(np.isfinite(e) & (e > 0), e, lc.noise)
    # Archived per-point errors often underestimate the real scatter (residual stellar variability, red noise):
    # 1.3–1.9× for the bundled Kepler/TESS samples. Inflate them to the measured robust scatter (never
    # shrink), otherwise every SNR — and the 7.1σ detection gate — would be optimistic on real data.
    error_scale = max(1.0, float(lc.noise / np.median(e))) if np.median(e) > 0 else 1.0
    e = e * error_scale
    periods, capped = period_grid(baseline, s.period_min, pmax, durations.min(), s.frequency_factor, s.max_periods)

    candidates: list[BLSCandidate] = []
    keep = np.ones(len(t), dtype=bool)
    for rank in range(1, s.n_signals + 1):
        tt, ff, ee = t[keep], f[keep], e[keep]
        if len(tt) < 20:
            break
        model = BoxLeastSquares(tt, ff, dy=ee)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.power(periods, durations, objective=s.objective)
        power = np.asarray(res.power)
        i = int(np.nanargmax(power))
        P, t0, D = float(res.period[i]), float(res.transit_time[i]), float(res.duration[i])
        stats = model.compute_stats(P, D, t0)
        oe = odd_even_test(tt, ff, P, t0, D)
        sec_depth, sec_snr = secondary_test(tt, ff, lc.noise, P, t0, D)
        std = float(np.nanstd(power))
        events, unobserved = _epoch_events(tt, ff, lc.noise, P, t0, D, s.min_points_in_transit, rank)
        candidates.append(
            BLSCandidate(
                rank=rank, period=P, t0=t0, duration=D,
                depth=float(res.depth[i]), depth_err=float(res.depth_err[i]), depth_snr=float(res.depth_snr[i]),
                power=float(power[i]), sde=float((power[i] - np.nanmean(power)) / std) if std > 0 else 0.0,
                depth_odd=oe["depth_odd"], depth_odd_err=oe["depth_odd_err"],
                depth_even=oe["depth_even"], depth_even_err=oe["depth_even_err"],
                harmonic_delta_log_likelihood=float(stats["harmonic_delta_log_likelihood"]),
                n_transits_expected=len(events) + len(unobserved), n_transits_observed=len(events),
                events=events, unobserved_times=unobserved, secondary_depth=sec_depth, secondary_snr=sec_snr,
                periods=np.asarray(res.period), power_spectrum=power,
            )
        )
        # Mask this signal (1.5 × duration) before searching for the next one.
        keep &= ~_mask(t, P, 1.5 * D, t0)
    return BLSResult(candidates=candidates, runtime_s=_time.perf_counter() - start, n_periods=len(periods),
                     grid_capped=capped, settings=s, period_max_used=pmax, error_scale=error_scale)


def odd_even_test(t: np.ndarray, f: np.ndarray, period: float, t0: float, duration: float) -> dict[str, float]:
    """Compare mean depths of odd- and even-numbered transits.

    Each observed epoch's depth is measured in the transit *core* (central
    half of the duration, avoiding ingress/egress), and the uncertainty of
    each group mean comes from the scatter between epochs. This is robust to
    an imperfect box model and to underestimated per-point errors; an
    eclipsing binary at twice the period shows a clear odd/even difference.
    """
    baseline = np.median(f)
    epochs = np.round((t - t0) / period).astype(int)
    core = np.abs(t - (t0 + epochs * period)) <= duration / 4
    depths = {0: [], 1: []}
    for n in np.unique(epochs[core]):
        sel = core & (epochs == n)
        if sel.sum() >= 1:
            depths[int(n) % 2].append(baseline - f[sel].mean())
    out = {"depth_odd": np.nan, "depth_odd_err": np.nan, "depth_even": np.nan, "depth_even_err": np.nan,
           "n_odd": len(depths[1]), "n_even": len(depths[0])}
    if len(depths[0]) >= 2 and len(depths[1]) >= 2:
        for parity, name in ((1, "odd"), (0, "even")):
            d = np.asarray(depths[parity])
            out[f"depth_{name}"] = float(d.mean())
            out[f"depth_{name}_err"] = float(d.std(ddof=1) / np.sqrt(len(d)))
    return out


def secondary_test(t: np.ndarray, f: np.ndarray, noise: float, period: float, t0: float,
                   duration: float) -> tuple[float, float]:
    """Depth and SNR of a dip at phase 0.5 (an eclipse of the *companion*).

    A deep secondary eclipse means the "transit" is probably an eclipsing
    binary (or the true period is half the BLS period). Hot Jupiters can
    show very shallow secondary eclipses too, so depth matters, not only SNR.
    """
    phase_dist = np.abs(np.mod(t - t0, period) - 0.5 * period)
    inside = phase_dist < duration / 2
    if inside.sum() < 3:
        return float("nan"), float("nan")
    depth = float(np.median(f) - f[inside].mean())
    return depth, float(depth / noise * np.sqrt(inside.sum())) if noise > 0 else float("nan")


def _mask(t: np.ndarray, period: float, duration: float, t0: float) -> np.ndarray:
    return np.abs(np.mod(t - t0 + 0.5 * period, period) - 0.5 * period) < 0.5 * duration


def bls_model_flux(time: np.ndarray, cand: BLSCandidate) -> np.ndarray:
    """Box model for plotting: 1 − depth inside transit, 1 outside."""
    return np.where(_mask(time, cand.period, cand.duration, cand.t0), 1 - cand.depth, 1.0)
