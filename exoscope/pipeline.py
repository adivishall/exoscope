"""End-to-end analysis: preprocess → detect (threshold, BLS) → compare → evaluate."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from . import __version__
from .data.model import LightCurveData
from .data.targets import ReferencePlanet, reference_planets
from .detection.bls import BLSResult, BLSSettings, run_bls
from .detection.threshold import ThresholdResult, ThresholdSettings, detect_threshold
from .dmath.number_theory import alias_relation
from .dmath.sets import EventSetComparison, compare_detections, detection_metrics
from .evaluation import CandidateEvaluation, evaluate_candidate
from .processing import PreprocessSettings, ProcessedLightCurve, preprocess


@dataclass
class AnalysisSettings:
    preprocess: PreprocessSettings = field(default_factory=PreprocessSettings)
    threshold: ThresholdSettings = field(default_factory=ThresholdSettings)
    bls: BLSSettings = field(default_factory=BLSSettings)
    run_threshold: bool = True
    run_bls: bool = True
    match_tolerance_hours: float = 2.0

    def validate(self) -> None:
        if not (self.run_threshold or self.run_bls):
            raise ValueError("Select at least one detection method.")
        if not 0.1 <= self.match_tolerance_hours <= 24:
            raise ValueError("Matching tolerance must be between 0.1 and 24 hours.")
        self.preprocess.validate()
        self.threshold.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess": self.preprocess.to_dict(),
            "threshold": self.threshold.to_dict(),
            "bls": self.bls.to_dict(),
            "run_threshold": self.run_threshold,
            "run_bls": self.run_bls,
            "match_tolerance_hours": self.match_tolerance_hours,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisSettings":
        bls = dict(d.get("bls", {}))
        if "durations_hours" in bls:
            bls["durations_hours"] = tuple(bls["durations_hours"])

        def pick(klass, raw):  # ignore unknown keys from older versions
            names = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in raw.items() if k in names})

        return cls(
            preprocess=pick(PreprocessSettings, d.get("preprocess", {})),
            threshold=pick(ThresholdSettings, d.get("threshold", {})),
            bls=pick(BLSSettings, bls),
            run_threshold=d.get("run_threshold", True),
            run_bls=d.get("run_bls", True),
            match_tolerance_hours=d.get("match_tolerance_hours", 2.0),
        )

    def flat(self) -> dict[str, Any]:
        """Flattened 'section.key' → value view, used for experiment diffs."""
        out = {}
        for section, value in self.to_dict().items():
            if isinstance(value, dict):
                for k, v in value.items():
                    out[f"{section}.{k}"] = tuple(v) if isinstance(v, list) else v
            else:
                out[section] = value
        return out

    def with_overrides(self, overrides: dict[str, Any]) -> "AnalysisSettings":
        """Copy with 'section.key' overrides, e.g. {'threshold.sigma_threshold': 3.5}."""
        new = replace(self, preprocess=replace(self.preprocess), threshold=replace(self.threshold),
                      bls=replace(self.bls))
        for key, value in overrides.items():
            if "." in key:
                section, attr = key.split(".", 1)
                setattr(getattr(new, section), attr, value)
            else:
                setattr(new, key, value)
        return new


def default_settings_for(data: LightCurveData) -> AnalysisSettings:
    """Sensible defaults adapted to the dataset (cadence and baseline)."""
    s = AnalysisSettings()
    summ = data.summary()
    cadence_min = summ.get("cadence_minutes", 30)
    baseline = summ.get("baseline_days", 30)
    if cadence_min < 5:
        s.preprocess.bin_minutes = 10.0  # 2-minute TESS data: bin for speed
    s.bls.period_min = max(0.3, min(0.5, baseline / 4))
    s.bls.durations_hours = tuple(d for d in (1.0, 2.0, 3.0, 4.5, 6.0) if d / 24 < s.bls.period_min)
    return s


@dataclass
class AnalysisResult:
    dataset_meta: dict[str, Any]
    dataset_summary: dict[str, Any]
    settings: AnalysisSettings
    processed: ProcessedLightCurve
    threshold: ThresholdResult | None
    bls: BLSResult | None
    comparison: EventSetComparison
    evaluation: CandidateEvaluation
    timings: dict[str, float]
    created_at: str
    truth_metrics: dict[str, Any] | None = None
    reference_check: dict[str, Any] | None = None
    app_version: str = __version__
    analysis_id: int | None = None

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_meta.get("source") == "synthetic"

    @property
    def best(self):
        return self.bls.best if self.bls else None

    def headline(self) -> dict[str, Any]:
        """Compact numbers for history, experiment diffs and reports."""
        h: dict[str, Any] = {
            "points_analysed": self.processed.n_points,
            "noise_ppm": self.processed.noise * 1e6,
            "threshold_events": len(self.threshold.events) if self.threshold else None,
            "threshold_period_days": (self.threshold.period.period
                                      if self.threshold and self.threshold.period and self.threshold.period.members
                                      else None),
        }
        if self.best:
            b = self.best
            h.update({"bls_period_days": b.period, "bls_depth_ppm": b.depth * 1e6, "bls_duration_hours": b.duration * 24,
                      "bls_snr": b.depth_snr, "bls_sde": b.sde, "bls_transits_observed": b.n_transits_observed})
        h["jaccard"] = self.comparison.jaccard() if self.comparison.bls_available else None
        h["criteria_passed"] = f"{self.evaluation.n_pass}/{len(self.evaluation.applicable)}"
        return h


def check_against_reference(period: float, refs: list[ReferencePlanet] | tuple[ReferencePlanet, ...]) -> dict | None:
    """Compare a detected period with known planets (equal / harmonic / unrelated)."""
    if not refs or not np.isfinite(period):
        return None
    best = None
    for p in refs:
        if not np.isfinite(p.period_days):
            continue
        rel = alias_relation(p.period_days, period)
        score = (0 if rel["kind"] == "equal" else 1 if rel["kind"] == "harmonic" else 2, rel["relative_error"])
        if best is None or score < best[0]:
            best = (score, p, rel)
    if best is None:
        return None
    _, p, rel = best
    return {"planet": p.name, "reference_period": p.period_days, "detected_period": period,
            "ratio": rel["ratio"], "fraction": str(rel["fraction"]), "kind": rel["kind"],
            "period_error_percent": 100 * abs(period - p.period_days) / p.period_days,
            "reference_depth_ppm": p.depth_percent * 1e4, "reference_duration_hours": p.duration_hours}


def truth_metrics(data: LightCurveData, threshold: ThresholdResult | None, bls: BLSResult | None,
                  tol_hours: float) -> dict[str, Any] | None:
    """Precision/recall/F1 and period error against injected (synthetic) truth."""
    if not data.truth:
        return None
    true_times = [e["t_center"] for e in data.truth["epochs"] if e["observed"]]
    out: dict[str, Any] = {"true_period": data.truth["period_days"], "n_true_observed": len(true_times)}
    if threshold is not None:
        out["threshold"] = detection_metrics([e.t_center for e in threshold.events], true_times, tol_hours / 24)
        if threshold.period and threshold.period.members:
            out["threshold"]["period_error_percent"] = (
                100 * abs(threshold.period.period - data.truth["period_days"]) / data.truth["period_days"])
    if bls is not None and bls.best:
        out["bls"] = detection_metrics([e.t_center for e in bls.best.events], true_times, tol_hours / 24)
        out["bls"]["period_error_percent"] = 100 * abs(bls.best.period - data.truth["period_days"]) / data.truth["period_days"]
    return out


def run_analysis(data: LightCurveData, settings: AnalysisSettings | None = None,
                 processed: ProcessedLightCurve | None = None,
                 references: list[ReferencePlanet] | None = None) -> AnalysisResult:
    s = settings or default_settings_for(data)
    s.validate()
    timings: dict[str, float] = {}

    t = _time.perf_counter()
    lc = processed if processed is not None else preprocess(data, s.preprocess)
    timings["preprocess_s"] = _time.perf_counter() - t

    thr = detect_threshold(lc, s.threshold) if s.run_threshold else None
    if thr:
        timings["threshold_s"] = thr.runtime_s
    bls = run_bls(lc, s.bls) if s.run_bls else None
    if bls:
        timings["bls_s"] = bls.runtime_s

    best = bls.best if bls else None
    comparison = compare_detections(
        thr.events if thr else [], best.events if best else [],
        best.unobserved_times if best else [], s.match_tolerance_hours,
    )
    summary = data.summary()
    star_radius = data.meta.get("RADIUS")
    refs = references if references is not None else list(reference_planets(data.target))
    if star_radius is None and refs:
        star_radius = refs[0].star_radius_solar
    evaluation = evaluate_candidate(best, thr, comparison, summary, s.match_tolerance_hours,
                                    float(star_radius) if star_radius else None)
    ref_check = None
    if data.is_real:
        period = best.period if best else (thr.period.period if thr and thr.period and thr.period.members else np.nan)
        ref_check = check_against_reference(period, refs)
    timings["total_s"] = sum(v for k, v in timings.items())
    return AnalysisResult(
        dataset_meta=dict(data.meta), dataset_summary=summary, settings=s, processed=lc, threshold=thr, bls=bls,
        comparison=comparison, evaluation=evaluation, timings=timings,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        truth_metrics=truth_metrics(data, thr, bls, s.match_tolerance_hours), reference_check=ref_check,
    )


def diff_settings(old: AnalysisSettings, new: AnalysisSettings) -> list[dict[str, Any]]:
    a, b = old.flat(), new.flat()
    return [{"setting": k, "before": a.get(k), "after": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def diff_results(old: AnalysisResult, new: AnalysisResult) -> list[dict[str, Any]]:
    a, b = old.headline(), new.headline()
    rows = []
    for k in a.keys() | b.keys():
        va, vb = a.get(k), b.get(k)
        change = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va is not None and vb is not None:
            change = vb - va
        rows.append({"result": k, "before": va, "after": vb, "change": change, "changed": va != vb})
    order = list(a.keys())
    return sorted(rows, key=lambda r: order.index(r["result"]) if r["result"] in order else 99)


def settings_dict_for_storage(result: AnalysisResult) -> dict[str, Any]:
    d = result.settings.to_dict()
    d["algorithm_versions"] = {
        "threshold": result.threshold.version if result.threshold else None,
        "bls": result.bls.version if result.bls else None,
    }
    return d


__all__ = ["AnalysisSettings", "AnalysisResult", "run_analysis", "default_settings_for", "diff_settings",
           "diff_results", "check_against_reference"]
