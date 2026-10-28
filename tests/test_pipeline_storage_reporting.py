import json

import numpy as np
import pytest

from exoscope import reporting, storage
from exoscope.data.reload import reload_dataset
from exoscope.data.samples import list_samples, load_sample
from exoscope.data.synthetic import SyntheticParams, generate_synthetic
from exoscope.data.targets import validate_target_name
from exoscope.evaluation import evaluate_candidate
from exoscope.learning.quiz import build_quiz, score
from exoscope.pipeline import AnalysisSettings, default_settings_for, diff_results, diff_settings, run_analysis


@pytest.fixture(scope="module")
def result():
    d = generate_synthetic(SyntheticParams())
    return run_analysis(d, default_settings_for(d))


def test_pipeline_truth_metrics(result):
    tm = result.truth_metrics
    assert tm["bls"]["recall"] == 1.0 and tm["bls"]["precision"] == 1.0
    assert tm["bls"]["period_error_percent"] < 0.2
    assert tm["threshold"]["recall"] >= 0.8
    assert result.evaluation.level == "strong"


def test_settings_roundtrip_and_diff(result):
    s = result.settings
    s2 = AnalysisSettings.from_dict(json.loads(json.dumps(s.to_dict())))
    assert s2.to_dict() == s.to_dict()
    s3 = s.with_overrides({"threshold.sigma_threshold": 4.0})
    assert diff_settings(s, s3) == [{"setting": "threshold.sigma_threshold", "before": 3.0, "after": 4.0}]
    assert s.threshold.sigma_threshold == 3.0  # original untouched
    with pytest.raises(ValueError):
        AnalysisSettings(run_threshold=False, run_bls=False).validate()


def test_experiment_diff(result):
    d = generate_synthetic(SyntheticParams())
    other = run_analysis(d, default_settings_for(d).with_overrides({"threshold.sigma_threshold": 5.0}))
    rows = {r["result"]: r for r in diff_results(result, other)}
    assert rows["threshold_events"]["changed"]


def test_storage_roundtrip(result):
    aid = storage.save_analysis(result, note="test")
    rows = storage.list_analyses()
    assert rows[0]["id"] == aid and rows[0]["source"] == "synthetic"
    rec = storage.load_analysis(aid)
    assert rec["settings"]["threshold"]["sigma_threshold"] == 3.0
    assert len(rec["events"]) == len(result.threshold.events) + len(result.best.events)
    assert rec["candidates"][0]["period_days"] == pytest.approx(result.best.period)
    storage.delete_analysis(aid)
    assert storage.load_analysis(aid) is None
    eid = storage.save_experiment("x", "y", {"a": [1]}, [{"k": np.float64(1.5)}], {"n": 1}, "2026-01-01")
    assert storage.list_experiments()[0]["id"] == eid


def test_reload_synthetic_is_identical(result):
    d = reload_dataset(result.dataset_meta)
    orig = generate_synthetic(SyntheticParams())
    assert np.array_equal(np.nan_to_num(d.flux), np.nan_to_num(orig.flux))


def test_reports(result):
    md = reporting.markdown_report(result)
    assert "SYNTHETIC DATA" in md and "Discrete Mathematics" in md and "inclusion–exclusion" in md
    meta = json.loads(reporting.metadata_json(result))
    assert meta["settings"]["algorithm_versions"]["bls"].startswith("bls-astropy")
    ev = reporting.events_frame(result)
    assert set(ev["method"]) == {"threshold", "bls"}
    html = reporting.html_report(result, [])
    assert html.startswith("<!doctype html>") and "<table>" in html


def test_bundled_samples_are_real_and_recover_known_periods():
    samples = {s["sample_id"]: s for s in list_samples()}
    assert {"kepler8_q2", "kepler10_q3", "wasp18_s2"} <= set(samples)
    for sid, ref in (("kepler8_q2", 3.5224991), ("kepler10_q3", 0.8374907), ("wasp18_s2", 0.94145223)):
        d = load_sample(sid)
        assert d.is_real and not d.is_synthetic
        r = run_analysis(d)
        assert r.reference_check["kind"] == "equal"
        assert abs(r.best.period - ref) / ref < 1e-3


def test_evaluation_without_bls():
    ev = evaluate_candidate(None, None, None, {}, 2.0)
    assert ev.verdict == "No periodic signal to evaluate"


@pytest.mark.parametrize("seed", range(100, 110))
def test_noise_only_is_not_reported_as_a_signal(seed):
    # Regression: the strongest BLS peak in pure noise reaches SNR ~4–6 (look-elsewhere effect);
    # it must be graded "no significant signal", not "possible".
    d = generate_synthetic(SyntheticParams(depth_ppm=1, noise_ppm=800, seed=seed))
    assert run_analysis(d).evaluation.level == "noise"


def test_eclipsing_binary_verdict():
    d = generate_synthetic(SyntheticParams(odd_depth_ratio=0.5))
    assert run_analysis(d).evaluation.level == "non-planetary"


def test_target_validation():
    assert validate_target_name("  Kepler-8 ") == "Kepler-8"
    assert validate_target_name("KIC 11904151") == "KIC 11904151"
    for bad in ("", "x'; drop table", "a" * 50, "<script>"):
        with pytest.raises(ValueError):
            validate_target_name(bad)


def test_synthetic_validation_and_labels():
    with pytest.raises(ValueError):
        generate_synthetic(SyntheticParams(duration_hours=40))
    d = generate_synthetic(SyntheticParams(seed=5))
    assert d.is_synthetic and "SYNTHETIC" in d.label and "SYNTHETIC" in d.source_label


def test_quiz_answers_and_dynamic_questions(result):
    qs = build_quiz(result)
    assert any(q.dynamic for q in qs)
    correct = {q.qid: q.answer for q in qs}
    assert score(qs, correct) == (len(qs), len(qs))
    union_q = next(q for q in qs if q.qid == "d_union")
    assert union_q.options[union_q.answer] == str(result.comparison.summary()["|A∪B|"])
    assert [q.answer for q in build_quiz(result, seed=1)] != [0] * len(qs)
