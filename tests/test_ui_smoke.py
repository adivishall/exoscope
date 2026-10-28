"""Headless smoke test: every page renders without exceptions, before and after an analysis."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from exoscope.ui.common import PAGES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("EXOSCOPE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.chdir(ROOT)


def _errors(at: AppTest) -> list[str]:
    return [str(e.value) for e in at.exception]


@pytest.mark.parametrize("page", list(PAGES.values()))
def test_pages_render_without_data(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.switch_page(page)
    at.run()
    assert not _errors(at), _errors(at)


@pytest.mark.parametrize("sample", ["kepler8_q2", "kepler10_q3"])
def test_pages_render_with_analysis(sample):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    from exoscope.data.samples import load_sample
    from exoscope.pipeline import default_settings_for, run_analysis

    data = load_sample(sample)
    at.session_state["dataset"] = data
    at.session_state["settings"] = default_settings_for(data)
    at.session_state["analysis"] = run_analysis(data)
    for page in PAGES.values():
        at.switch_page(page)
        at.run()
        assert not _errors(at), f"{page}: {_errors(at)}"


def test_synthetic_analysis_pages():
    from dataclasses import asdict

    from exoscope.data.synthetic import SyntheticParams, generate_synthetic
    from exoscope.pipeline import default_settings_for, run_analysis

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    data = generate_synthetic(SyntheticParams(**asdict(SyntheticParams())))
    at.session_state["dataset"] = data
    at.session_state["settings"] = default_settings_for(data)
    at.session_state["analysis"] = run_analysis(data)
    for page in ("results", "sets", "numbers", "about"):
        at.switch_page(PAGES[page])
        at.run()
        assert not _errors(at), f"{page}: {_errors(at)}"


def _analysed_app(sample="kepler8_q2"):
    from exoscope.data.samples import load_sample
    from exoscope.pipeline import default_settings_for, run_analysis

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    data = load_sample(sample)
    at.session_state["dataset"] = data
    at.session_state["settings"] = default_settings_for(data)
    at.session_state["analysis"] = run_analysis(data)
    return at


def test_combinatorics_run_button_executes_experiment():
    at = _analysed_app()
    at.switch_page(PAGES["combinatorics"])
    at.run()
    run = next(b for b in at.button if b.label.startswith("Run "))
    run.click()
    at.run()
    assert not _errors(at), _errors(at)
    exp = at.session_state["experiment"]
    assert exp["stats"]["configs_run"] >= 1 and all(r["status"] == "ok" for r in exp["results"])
    at.switch_page(PAGES["relations"])  # the configuration → count function becomes available
    at.run()
    assert not _errors(at)


def test_explore_synthetic_form_generates_labelled_data():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.switch_page(PAGES["explore"])
    at.run()
    submit = next(b for b in at.button if "Generate synthetic" in b.label)
    submit.click()
    at.run()
    assert not _errors(at)
    assert at.session_state["dataset"].is_synthetic


def test_analyze_form_runs_and_experiment_mode_diffs():
    at = _analysed_app("kepler10_q3")
    at.switch_page(PAGES["analyze"])
    at.run()
    at.slider[next(i for i, s in enumerate(at.slider) if s.label.startswith("Threshold k"))].set_value(4.0)
    next(b for b in at.button if b.label == "Run analysis").click()
    at.run()
    assert not _errors(at), _errors(at)
    assert at.session_state["analysis"].settings.threshold.sigma_threshold == 4.0
    assert at.session_state["previous_analysis"] is not None


def test_quiz_submit_scores():
    at = _analysed_app()
    at.switch_page(PAGES["learn"])
    at.run()
    next(b for b in at.button if b.label == "Submit answers").click()
    at.run()
    assert not _errors(at)
    assert any(m.label == "Score" for m in at.metric)


@pytest.mark.parametrize("source", ["bundled", "synthetic"])
def test_history_reload_and_rerun(source):
    from exoscope import storage
    from exoscope.data.samples import load_sample
    from exoscope.data.synthetic import SyntheticParams, generate_synthetic
    from exoscope.pipeline import default_settings_for, run_analysis

    data = load_sample("kepler8_q2") if source == "bundled" else generate_synthetic(SyntheticParams(seed=11))
    s = default_settings_for(data).with_overrides({"threshold.sigma_threshold": 3.7})
    original = run_analysis(data, s)
    storage.save_analysis(original)

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page(PAGES["history"])
    at.run()
    next(b for b in at.button if b.label.startswith("Reload this dataset")).click()
    at.run()
    assert not _errors(at), _errors(at)
    rerun = at.session_state["analysis"]
    assert rerun.settings.threshold.sigma_threshold == 3.7  # stored settings were restored
    assert rerun.best.period == pytest.approx(original.best.period)  # same data → same result
    assert len(storage.list_analyses()) == 2
