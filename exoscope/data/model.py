"""The light-curve container shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Where a dataset came from. Anything other than "mast" or "bundled" is not a
# real astronomical observation and must be labelled as such in the UI.
SOURCE_LABELS = {
    "mast": "Real observation (downloaded from MAST via Lightkurve)",
    "bundled": "Real observation (offline copy bundled with ExoScope)",
    "synthetic": "SYNTHETIC data (generated for teaching/testing, not a real star)",
    "injected": "Real observation with SIMULATED transits injected (injection–recovery test)",
}


class DataError(ValueError):
    """Raised when a dataset is empty, malformed or otherwise unusable."""


@dataclass
class LightCurveData:
    """A single light curve plus provenance.

    Attributes:
        time: observation times in days (BKJD for Kepler, BTJD for TESS,
            plain days for synthetic data — see ``meta["time_format"]``).
        flux: flux values (any unit; normalised during preprocessing).
        flux_err: per-point uncertainties, or ``None``.
        quality_ok: True where the mission quality flags mark the cadence as
            good. All True when the mission provides no flags.
        meta: provenance and header information (target, mission, source...).
        truth: for synthetic data only — the injected transit parameters and
            epochs, used as ground truth for precision/recall.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None
    quality_ok: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    truth: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.time.ndim != 1 or self.flux.ndim != 1:
            raise DataError("time and flux must be one-dimensional arrays")
        if len(self.time) != len(self.flux):
            raise DataError(f"time ({len(self.time)}) and flux ({len(self.flux)}) lengths differ")
        if self.flux_err is not None:
            self.flux_err = np.asarray(self.flux_err, dtype=float)
            if len(self.flux_err) != len(self.flux):
                raise DataError("flux_err length differs from flux length")
        if self.quality_ok is None:
            self.quality_ok = np.ones(len(self.time), dtype=bool)
        else:
            self.quality_ok = np.asarray(self.quality_ok, dtype=bool)

    # -- provenance -------------------------------------------------------
    @property
    def source(self) -> str:
        return str(self.meta.get("source", "unknown"))

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    @property
    def is_real(self) -> bool:
        return self.source in ("mast", "bundled")

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, f"Unknown source ({self.source})")

    @property
    def target(self) -> str:
        return str(self.meta.get("target", "unknown"))

    @property
    def label(self) -> str:
        """Short human-readable dataset name, e.g. 'Kepler-8 · Quarter 2'."""
        parts = [self.target]
        if self.meta.get("dataset"):
            parts.append(str(self.meta["dataset"]))
        if self.is_synthetic:
            parts.append("SYNTHETIC")
        return " · ".join(parts)

    @property
    def time_format(self) -> str:
        return str(self.meta.get("time_format", "days"))

    # -- quality summary --------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Data-quality statistics computed from the raw arrays."""
        n = len(self.time)
        finite = np.isfinite(self.time) & np.isfinite(self.flux)
        good = finite & self.quality_ok
        out: dict[str, Any] = {
            "n_points": n,
            "n_nan": int(n - finite.sum()),
            "n_quality_flagged": int((finite & ~self.quality_ok).sum()),
            "n_usable": int(good.sum()),
        }
        t = np.sort(self.time[good])
        if len(t) >= 2:
            dt = np.diff(t)
            cadence = float(np.median(dt))
            baseline = float(t[-1] - t[0])
            out.update(
                {
                    "t_start": float(t[0]),
                    "t_end": float(t[-1]),
                    "baseline_days": baseline,
                    "cadence_minutes": cadence * 24 * 60,
                    "largest_gap_days": float(dt.max()),
                    # fraction of the baseline actually covered by usable cadences
                    "duty_cycle": float(min(1.0, len(t) * cadence / baseline)) if baseline > 0 else 0.0,
                }
            )
            f = self.flux[good]
            med = np.median(f)
            if med != 0:
                rel = f / med
                # point-to-point scatter is insensitive to slow trends and transits
                p2p = 1.4826 * np.median(np.abs(np.diff(rel) - np.median(np.diff(rel)))) / np.sqrt(2)
                out["p2p_scatter_ppm"] = float(p2p * 1e6)
        out["completeness"] = out["n_usable"] / n if n else 0.0
        return out

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"time": self.time, "flux": self.flux})
        if self.flux_err is not None:
            df["flux_err"] = self.flux_err
        df["quality_ok"] = self.quality_ok
        return df

    @classmethod
    def from_frame(cls, df: pd.DataFrame, meta: dict[str, Any], truth: dict | None = None) -> "LightCurveData":
        missing = {"time", "flux"} - set(df.columns)
        if missing:
            raise DataError(f"missing required columns: {sorted(missing)}")
        return cls(
            time=df["time"].to_numpy(float),
            flux=df["flux"].to_numpy(float),
            flux_err=df["flux_err"].to_numpy(float) if "flux_err" in df else None,
            quality_ok=df["quality_ok"].to_numpy(bool) if "quality_ok" in df else None,
            meta=dict(meta),
            truth=truth,
        )
