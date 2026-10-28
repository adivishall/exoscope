"""The original ExoScope v1 pipeline, preserved for comparison.

v1 fetched one hard-coded star with Lightkurve, flattened the flux, flagged
every point below ``mean − 3·std`` as a "dip", and estimated the period as
the median spacing between flagged points. It is kept (unchanged in spirit)
so the Methodology page can show, on the same data, why each improvement in
v2 was made:

* mean/std are biased by the transits themselves;
* every flagged *point* counted, so one transit produced several "dips" and
  single noisy points produced false ones;
* the median spacing collapses to the cadence (points inside one transit)
  or doubles when a transit is missed.

``run_legacy_original`` reproduces the original script end-to-end (network
required); ``legacy_detect`` applies the v1 logic to any already-flattened
light curve so it can be compared offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEGACY_DEFAULT_TARGET = "Kepler-10"


@dataclass
class LegacyResult:
    dip_times: np.ndarray  # every flagged point
    threshold: float
    period_estimate: float | None


def legacy_detect(time: np.ndarray, flux: np.ndarray, k: float = 3.0) -> LegacyResult:
    flux = np.asarray(flux, dtype=float)
    threshold = float(np.nanmean(flux) - k * np.nanstd(flux))
    dips = np.asarray(time)[flux < threshold]
    period = float(np.median(np.diff(dips))) if len(dips) >= 2 else None
    return LegacyResult(dip_times=dips, threshold=threshold, period_estimate=period)


def run_legacy_original(target: str = LEGACY_DEFAULT_TARGET) -> LegacyResult:  # pragma: no cover - network
    """The v1 script: download, flatten, threshold, median spacing."""
    import lightkurve as lk

    lc = lk.search_lightcurve(target, mission="Kepler", author="Kepler", cadence="long")[0].download()
    flat = lc.remove_nans().flatten()
    return legacy_detect(flat.time.value, np.asarray(flat.flux.value, dtype=float))
