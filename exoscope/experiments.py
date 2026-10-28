"""Run a set of configurations (from the Cartesian product) on one dataset.

Expensive stages are memoised by the parameters they depend on:
preprocessing by the preprocessing parameters, BLS by (preprocessing, BLS)
parameters. The counts of distinct keys are the sizes of projections of the
configuration set — combinatorics deciding how much work is really needed.
"""

from __future__ import annotations

import time as _time
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data.model import LightCurveData
from .detection.bls import run_bls
from .detection.threshold import detect_threshold
from .dmath.combinatorics import Constraint, Parameter, ParameterSpace
from .dmath.sets import compare_detections
from .pipeline import AnalysisSettings, check_against_reference
from .processing import preprocess

# Parameters a student can vary, with the values offered by default.
PARAMETER_CATALOG: dict[str, Parameter] = {
    "threshold.sigma_threshold": Parameter("threshold.sigma_threshold", "Threshold k (σ)", [2.5, 3.0, 3.5, 4.0], "threshold", "σ"),
    "threshold.min_points": Parameter("threshold.min_points", "Min points / event", [1, 2, 3], "threshold"),
    "preprocess.flatten_window_days": Parameter("preprocess.flatten_window_days", "Flatten window", [0.25, 0.5, 1.0, 2.0], "preprocess", " d"),
    "preprocess.clip_upper_sigma": Parameter("preprocess.clip_upper_sigma", "Upper clip", [3.0, 4.0, None], "preprocess", "σ"),
    "bls.durations_hours": Parameter("bls.durations_hours", "BLS durations", [(1.0, 2.0, 3.0), (2.0, 4.0, 6.0)], "bls"),
    "bls.period_min": Parameter("bls.period_min", "BLS min period", [0.5, 1.0], "bls", " d"),
    "match_tolerance_hours": Parameter("match_tolerance_hours", "Match tolerance", [1.0, 2.0, 3.0], "matching", " h"),
}

DEFAULT_SELECTION = {
    "threshold.sigma_threshold": [2.5, 3.0, 3.5, 4.0],
    "threshold.min_points": [1, 2, 3],
    "preprocess.flatten_window_days": [0.5, 1.0],
}

MAX_RUNS = 60


def _flatten_window_ok(cfg: dict[str, Any], base: AnalysisSettings) -> bool:
    window = cfg.get("preprocess.flatten_window_days", base.preprocess.flatten_window_days)
    durations = cfg.get("bls.durations_hours", base.bls.durations_hours)
    return window * 24 >= 3 * max(durations)


def _durations_ok(cfg: dict[str, Any], base: AnalysisSettings) -> bool:
    pmin = cfg.get("bls.period_min", base.bls.period_min)
    durations = cfg.get("bls.durations_hours", base.bls.durations_hours)
    return max(durations) / 24 < pmin


def build_space(selection: dict[str, list[Any]], base: AnalysisSettings) -> ParameterSpace:
    params = []
    for key, values in selection.items():
        if key not in PARAMETER_CATALOG:
            raise ValueError(f"unknown parameter {key}")
        if values:
            p = PARAMETER_CATALOG[key]
            params.append(Parameter(p.key, p.label, list(values), p.group, p.unit))
    if not params:
        raise ValueError("choose at least one value for at least one parameter")
    constraints = [
        Constraint("flatten window ≥ 3 × longest BLS trial duration (else detrending can erase transits)",
                   lambda c: _flatten_window_ok(c, base)),
        Constraint("longest BLS duration < minimum period", lambda c: _durations_ok(c, base)),
    ]
    space = ParameterSpace(params, constraints)
    space.validate()
    return space


def run_configurations(data: LightCurveData, base: AnalysisSettings, space: ParameterSpace,
                       configs: list[dict[str, Any]], references=None,
                       progress: Callable[[int, int, str], None] | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    if len(configs) > MAX_RUNS:
        raise ValueError(f"at most {MAX_RUNS} configurations can be executed at once")
    pre_cache: dict[tuple, Any] = {}
    bls_cache: dict[tuple, Any] = {}
    pre_keys = [p.key for p in space.parameters if p.group == "preprocess"]
    bls_keys = [p.key for p in space.parameters if p.group == "bls"]
    rows = []
    for i, cfg in enumerate(configs):
        if progress:
            progress(i, len(configs), space.short_label(cfg))
        s = base.with_overrides(cfg)
        start = _time.perf_counter()
        row: dict[str, Any] = {"config": space.short_label(cfg), **{p.label: p.fmt(cfg[p.key]) for p in space.parameters}}
        try:
            pk = tuple(cfg[k] for k in pre_keys)
            if pk not in pre_cache:
                pre_cache[pk] = preprocess(data, s.preprocess)
            lc = pre_cache[pk]
            thr = detect_threshold(lc, s.threshold) if s.run_threshold else None
            bk = pk + tuple(cfg[k] for k in bls_keys)
            if s.run_bls and bk not in bls_cache:
                bls_cache[bk] = run_bls(lc, s.bls)
            bls = bls_cache.get(bk) if s.run_bls else None
            best = bls.best if bls else None
            comp = compare_detections(thr.events if thr else [], best.events if best else [],
                                      best.unobserved_times if best else [], s.match_tolerance_hours)
            thr_p = thr.period.period if thr and thr.period and thr.period.members and not thr.period.ambiguous else np.nan
            row.update({
                "threshold_events": len(thr.events) if thr else np.nan,
                "threshold_period_d": thr_p,
                "bls_period_d": best.period if best else np.nan,
                "bls_snr": best.depth_snr if best else np.nan,
                "bls_transits": best.n_transits_observed if best else np.nan,
                "|A∩B|": len(comp.intersection()),
                "jaccard": comp.jaccard() if comp.bls_available else np.nan,
            })
            truth_p = data.truth["period_days"] if data.truth else None
            if truth_p:
                row["bls_period_error_%"] = 100 * abs(row["bls_period_d"] - truth_p) / truth_p if best else np.nan
                row["thr_period_error_%"] = 100 * abs(thr_p - truth_p) / truth_p if np.isfinite(thr_p) else np.nan
            elif references and best:
                ref = check_against_reference(best.period, references)
                row["matches_reference"] = ref["kind"] if ref else "n/a"
            row["status"] = "ok"
        except ValueError as exc:
            row["status"] = f"error: {exc}"
        row["runtime_s"] = _time.perf_counter() - start
        rows.append(row)
    if progress:
        progress(len(configs), len(configs), "done")
    stats = {"configs_run": len(configs), "preprocessing_runs": len(pre_cache), "bls_runs": len(bls_cache)}
    return pd.DataFrame(rows), stats
