import numpy as np
import pytest

from exoscope.data.model import DataError, LightCurveData
from exoscope.data.synthetic import SyntheticParams, generate_synthetic
from exoscope.detection.bls import BLSSettings, run_bls
from exoscope.detection.threshold import ThresholdSettings, detect_threshold
from exoscope.legacy import legacy_detect
from exoscope.processing import PreprocessSettings, preprocess, robust_std


def flat_lc(n=2000, noise=1e-4, seed=0, cadence=0.02):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * cadence
    return LightCurveData(t, 1000 * (1 + rng.normal(0, noise, n)), meta={"source": "synthetic"})


def test_empty_and_tiny_inputs_rejected():
    with pytest.raises(DataError):
        preprocess(LightCurveData(np.array([]), np.array([]), meta={}))
    with pytest.raises(DataError):
        preprocess(LightCurveData(np.arange(5.0), np.ones(5), meta={}))
    with pytest.raises(DataError):
        LightCurveData(np.arange(3.0), np.ones(4))
    with pytest.raises(DataError):
        preprocess(LightCurveData(np.arange(100.0), np.full(100, np.nan), meta={}))


def test_missing_values_quality_flags_and_duplicates():
    lc = flat_lc(500)
    lc.flux[[10, 20]] = np.nan
    lc.quality_ok[30:40] = False
    lc.time[50] = lc.time[49]  # duplicate timestamp
    p = preprocess(lc, PreprocessSettings(flatten=False, clip_upper_sigma=None))
    assert p.n_points == 500 - 2 - 10 - 1
    assert np.all(np.diff(p.time) > 0)  # time -> flux is a function
    assert np.median(p.flux) == pytest.approx(1.0, abs=1e-6)


def test_upper_clipping_keeps_transits_and_removes_spikes():
    lc = flat_lc(1000, noise=1e-4)
    lc.flux[100] *= 1.01  # cosmic ray
    lc.flux[500:505] *= 0.99  # transit
    p = preprocess(lc, PreprocessSettings(flatten=False))
    assert len(p.clipped_time) == 1
    assert p.flux.min() < 0.992


def test_flatten_removes_trend_but_not_transit():
    d = generate_synthetic(SyntheticParams(variability_amp_ppm=10_000, noise_ppm=200, n_gaps=0, outlier_fraction=0))
    p = preprocess(d, PreprocessSettings(flatten_window_days=1.0))
    out = np.abs(((p.time - d.truth["t0_days"] + 1.5) % 3.0) - 1.5) > 0.2
    assert robust_std(p.flux[out]) < 400e-6
    assert 1 - p.flux.min() > 0.004


def test_invalid_settings():
    with pytest.raises(ValueError):
        PreprocessSettings(flatten_window_days=0).validate()
    with pytest.raises(ValueError):
        ThresholdSettings(sigma_threshold=0.5).validate()
    with pytest.raises(ValueError):
        BLSSettings(period_min=0.2, durations_hours=(6.0,)).validate()
    with pytest.raises(ValueError):
        BLSSettings(period_min=10).validate(baseline_days=15)


def test_no_transit_gives_no_threshold_events_and_weak_bls():
    p = preprocess(flat_lc(3000, noise=1e-4, cadence=0.02), PreprocessSettings(flatten=False))
    thr = detect_threshold(p, ThresholdSettings(sigma_threshold=4.0, min_points=2))
    assert thr.events == []
    bls = run_bls(p, BLSSettings(period_min=0.5, durations_hours=(1.0, 2.0)))
    assert bls.best.depth_snr < 7.1


def test_single_outlier_is_not_an_event():
    lc = flat_lc(1000)
    lc.flux[400] *= 0.99
    thr = detect_threshold(preprocess(lc, PreprocessSettings(flatten=False)), ThresholdSettings(min_points=2))
    assert thr.events == [] and thr.rejected_runs >= 1


def test_threshold_recovers_known_period(synth_processed, synth):
    thr = detect_threshold(synth_processed)
    assert len(thr.events) >= 7
    assert thr.period.period == pytest.approx(synth.truth["period_days"], rel=1e-3)


def test_bls_recovers_known_period_and_depth(synth_processed, synth):
    b = run_bls(synth_processed).best
    assert b.period == pytest.approx(3.0, rel=2e-3)
    assert b.depth * 1e6 == pytest.approx(synth.truth["depth_ppm"], rel=0.25)
    assert b.depth_snr > 20 and b.n_transits_observed >= 7
    assert b.odd_even_sigma < 3
    assert b.harmonic_delta_log_likelihood < 0  # box preferred over sinusoid


def test_bls_flags_eclipsing_binary():
    d = generate_synthetic(SyntheticParams(odd_depth_ratio=0.4, depth_ppm=8000, n_gaps=0))
    b = run_bls(preprocess(d)).best
    alt = b.period == pytest.approx(6.0, rel=0.01)
    assert (alt and b.secondary_snr > 3) or (not alt and b.odd_even_sigma > 3)


def test_bls_multiple_signals():
    d = generate_synthetic(SyntheticParams(depth_ppm=6000, noise_ppm=300, n_gaps=0))
    from exoscope.data.synthetic import transit_model
    d.flux = d.flux * transit_model(d.time, 4.7, 2.0, 0.1, 0.003)
    res = run_bls(preprocess(d), BLSSettings(n_signals=2))
    periods = sorted(round(c.period, 1) for c in res.candidates)
    assert periods == [3.0, 4.7]


def test_noisy_missed_events_still_periodic():
    d = generate_synthetic(SyntheticParams(depth_ppm=2500, noise_ppm=900, seed=7))
    thr = detect_threshold(preprocess(d), ThresholdSettings(sigma_threshold=3.0))
    if thr.period and thr.period.members and not thr.period.ambiguous:
        assert thr.period.period == pytest.approx(3.0, rel=5e-3)


def test_legacy_detector_counts_points_not_events(synth_processed):
    leg = legacy_detect(synth_processed.time, synth_processed.flux)
    thr = detect_threshold(synth_processed)
    assert len(leg.dip_times) > len(thr.events)  # several points per transit
    assert leg.period_estimate < 0.5  # median spacing collapses to the cadence


def test_injection_recovery_helpers():
    from exoscope.data.samples import load_sample
    from exoscope.injection import Injection, inject, mask_transits, white_noise_twin
    from exoscope.pipeline import run_analysis

    host = load_sample("kepler8_q2")
    known = run_analysis(host).best
    masked = mask_transits(host, known.period, known.t0, known.duration, factor=2.5)
    assert len(masked.time) < len(host.time)
    assert run_analysis(masked).evaluation.level == "noise"  # the planet is gone
    d = inject(masked, Injection(4.3, masked.time.min() + 1.0, 0.002, 0.125))
    assert d.meta["source"] == "injected" and not d.is_real and "SIMULATED" in d.source_label
    r = run_analysis(d)
    assert r.best.period == pytest.approx(4.3, rel=0.01) and r.reference_check is None
    twin = white_noise_twin(masked)
    assert twin.is_synthetic and len(twin.time) <= len(masked.time)
    with pytest.raises(ValueError):
        mask_transits(host, 0, 0, 0.1)


def test_bls_inflates_underestimated_errors():
    from exoscope.data.samples import load_sample
    from exoscope.pipeline import default_settings_for

    d = load_sample("kepler10_q3")
    p = preprocess(d, default_settings_for(d).preprocess)
    res = run_bls(p, default_settings_for(d).bls)
    assert res.error_scale == pytest.approx(p.noise / np.median(p.flux_err), rel=1e-6) and res.error_scale > 1.3
    # never shrink: exact synthetic errors give a scale of ~1
    s = generate_synthetic(SyntheticParams(noise_ppm=500, outlier_fraction=0, variability_amp_ppm=0))
    assert run_bls(preprocess(s)).error_scale < 1.1
