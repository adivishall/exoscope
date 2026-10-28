"""Shared Streamlit helpers: session state, cached data access, layout blocks."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from ..data.archive import lookup_known_planets
from ..data.model import DataError, LightCurveData
from ..data.retrieval import download_lightcurve, search_target
from ..data.samples import list_samples, load_sample
from ..data.synthetic import SyntheticParams, generate_synthetic
from ..pipeline import AnalysisResult, AnalysisSettings, default_settings_for, run_analysis
from .. import storage

PAGES: dict[str, str] = {
    "home": "views/home.py",
    "explore": "views/explore.py",
    "analyze": "views/analyze.py",
    "results": "views/results.py",
    "lab": "views/lab_overview.py",
    "sets": "views/lab_sets.py",
    "numbers": "views/lab_number_theory.py",
    "graphs": "views/lab_graphs.py",
    "relations": "views/lab_relations.py",
    "combinatorics": "views/lab_combinatorics.py",
    "learn": "views/learn.py",
    "history": "views/history.py",
    "about": "views/about.py",
}

STATUS_ICON = {"pass": "✅", "caution": "⚠️", "fail": "❌", "n/a": "➖"}


# ------------------------------------------------------------------ state
def init_state() -> None:
    defaults = {"dataset": None, "settings": None, "analysis": None, "previous_analysis": None,
                "experiment": None, "guided": True, "references": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def dataset() -> LightCurveData | None:
    return st.session_state.get("dataset")


def analysis() -> AnalysisResult | None:
    return st.session_state.get("analysis")


def settings() -> AnalysisSettings:
    s = st.session_state.get("settings")
    if s is None:
        d = dataset()
        s = default_settings_for(d) if d is not None else AnalysisSettings()
        st.session_state.settings = s
    return s


def set_dataset(data: LightCurveData) -> None:
    st.session_state.dataset = data
    st.session_state.settings = default_settings_for(data)
    st.session_state.analysis = None
    st.session_state.previous_analysis = None
    st.session_state.experiment = None
    st.session_state.references = None


def references() -> tuple[list, str]:
    """Known planets for the current real target (cached per session)."""
    d = dataset()
    if d is None or not d.is_real:
        return [], ""
    if st.session_state.references is None:
        st.session_state.references = cached_archive(d.target)
    return st.session_state.references


def run_current(s: AnalysisSettings, save: bool = True) -> AnalysisResult | None:
    """Run the pipeline on the current dataset; keep the previous run for comparison."""
    d = dataset()
    if d is None:
        st.error("Load a dataset first.")
        return None
    refs, _ = references()
    try:
        with st.spinner("Preprocessing, detecting and evaluating…"):
            result = run_analysis(d, s, references=refs or None)
    except (ValueError, DataError) as exc:
        st.error(f"Analysis could not run: {exc}")
        return None
    prev = analysis()
    if prev is not None and prev.dataset_meta.get("dataset") == result.dataset_meta.get("dataset"):
        st.session_state.previous_analysis = prev
    st.session_state.analysis = result
    st.session_state.settings = s
    if save:
        try:
            storage.save_analysis(result)
        except Exception as exc:  # storage problems must not break the analysis
            st.toast(f"Analysis not saved to history: {exc}", icon="⚠️")
    return result


def quick_start(sample_id: str = "kepler8_q2") -> None:
    """Load a bundled real sample and analyse it with default settings."""
    try:
        data = cached_sample(sample_id)
    except DataError as exc:
        st.error(str(exc))
        return
    set_dataset(data)
    run_current(settings())


# ------------------------------------------------------------------ cached data access
@st.cache_data(show_spinner=False, ttl=3600)
def cached_search(target: str, mission: str) -> list[dict[str, Any]]:
    return search_target(target, mission)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_download(target: str, mission: str, index: int, products: list[dict[str, Any]]) -> LightCurveData:
    return download_lightcurve(target, mission, index, products=products)


@st.cache_data(show_spinner=False)
def cached_sample(sample_id: str) -> LightCurveData:
    return load_sample(sample_id)


@st.cache_data(show_spinner=False)
def cached_samples() -> list[dict]:
    return list_samples()


@st.cache_data(show_spinner=False, max_entries=30)
def cached_synthetic(params: dict[str, Any]) -> LightCurveData:
    return generate_synthetic(SyntheticParams(**params))


@st.cache_data(show_spinner=False, ttl=86400)
def cached_archive(target: str):
    return lookup_known_planets(target)


# ------------------------------------------------------------------ layout blocks
def metric_cols(n: int) -> list:
    """Like st.columns(n) for metrics, but the cells wrap onto a new line on narrow screens
    (projectors, laptops) instead of truncating the values."""
    row = st.container(horizontal=True, gap="medium")
    return [row.container(width=190) for _ in range(n)]


def plot(fig, key: str | None = None, **kwargs):
    return st.plotly_chart(fig, theme="streamlit", key=key, config={"displaylogo": False}, **kwargs)


def source_badge(data_or_meta) -> None:
    meta = getattr(data_or_meta, "meta", data_or_meta)  # duck-typed: survives module hot-reloads
    src = meta.get("source")
    if src == "synthetic":
        st.badge("SYNTHETIC DATA — not a real star", icon=":material/science:", color="orange")
    elif src == "bundled":
        st.badge("Real observation · offline copy", icon=":material/verified:", color="green")
    elif src == "injected":
        st.badge("Real data + SIMULATED injected transits", icon=":material/science:", color="orange")
    elif src == "mast":
        st.badge("Real observation · MAST", icon=":material/satellite_alt:", color="green")
    else:
        st.badge(f"Unknown source: {src}", color="gray")


def dataset_header(data: LightCurveData) -> None:
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        st.markdown(f"**Dataset:** {data.label}")
    with c2:
        source_badge(data)
    if data.is_synthetic:
        st.warning("You are working with **synthetic** data generated by ExoScope for teaching and testing. "
                   "Results describe the injected signal, not a real star.", icon=":material/science:")


def require_dataset() -> LightCurveData:
    d = dataset()
    if d is None:
        empty_state("No light curve loaded yet",
                    "Choose a real target, a bundled real sample or a synthetic light curve on **Explore Targets**, "
                    "or start instantly with the Kepler-8 sample.")
        st.stop()
    return d


def require_analysis() -> AnalysisResult:
    r = analysis()
    if r is None:
        d = dataset()
        if d is None:
            empty_state("Nothing analysed yet",
                        "This page uses the results of an analysis. Load data and run the detectors first, "
                        "or use the quick start to analyse a real Kepler light curve with default settings.")
        else:
            st.info(f"**{d.label}** is loaded but not analysed yet.", icon=":material/info:")
            if st.button("Run analysis with default settings", type="primary", icon=":material/play_arrow:"):
                run_current(settings())
                st.rerun()
            st.page_link(PAGES["analyze"], label="Or choose settings on Analyze Light Curves",
                         icon=":material/tune:")
        st.stop()
    return r


def empty_state(title: str, body: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(body)
        c1, c2 = st.columns(2)
        if c1.button("Quick start: analyse Kepler-8 (real data)", type="primary", icon=":material/rocket_launch:",
                     key=f"qs_{title}"):
            quick_start("kepler8_q2")
            st.rerun()
        c2.page_link(PAGES["explore"], label="Explore targets", icon=":material/travel_explore:")


def guided() -> bool:
    return bool(st.session_state.get("guided", True))


def concept(title: str, body: str) -> None:
    with st.expander(f"① Concept — {title}", expanded=guided(), icon=":material/school:"):
        st.markdown(body)


def worked_example(title: str):
    return st.expander(f"② Worked example — {title}", expanded=guided(), icon=":material/calculate:")


def your_data_header(text: str = "③ Your data") -> None:
    st.subheader(text)


def learned(points: list[str]) -> None:
    with st.container(border=True):
        st.markdown("**④ What did we learn?**")
        st.markdown("\n".join(f"- {p}" for p in points))


def fmt(v: Any, digits: int = 4, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, (float, np.floating)):
        return "—" if not np.isfinite(v) else f"{v:,.{digits}g}{suffix}"
    return f"{v}{suffix}"


def sidebar_status() -> None:
    with st.sidebar:
        st.divider()
        d = dataset()
        if d is None:
            st.caption("No dataset loaded")
        else:
            st.caption("Current dataset")
            st.markdown(f"**{d.label}**")
            source_badge(d)
            r = analysis()
            if r is not None and r.best is not None:
                st.caption(f"Best BLS period: **{r.best.period:.5f} d** · "
                           f"{r.evaluation.n_pass}/{len(r.evaluation.applicable)} checks passed")
            elif r is not None:
                st.caption("Analysed (threshold only)")
            else:
                st.caption("Not analysed yet")
        st.toggle("Guided learning mode", key="guided",
                  help="Open the concept and worked-example sections automatically on every Lab page.")
