import streamlit as st

from exoscope import __version__
from exoscope.ui.common import PAGES, analysis, cached_synthetic, quick_start, run_current, set_dataset, settings
from exoscope.data.synthetic import SyntheticParams
from dataclasses import asdict

st.title("ExoScope")
st.markdown(
    "#### Detect, compare and evaluate exoplanet transit signals — and see the Discrete Mathematics behind every step."
)
st.caption(f"Version {__version__} · Discrete Mathematics course project · Real Kepler & TESS data via MAST")

st.info(
    "ExoScope reports **potential transit candidates**. A repeating dip is consistent with a planet, but eclipsing "
    "binaries, blended stars and instrument effects can look the same. Confirmation needs independent follow-up.",
    icon=":material/info:",
)

st.subheader("Start in one click")
c1, c2, c3 = st.columns(3)
with c1.container(border=True, height="stretch"):
    st.markdown("**Kepler-8 · real data**")
    st.caption("A hot Jupiter with deep 0.9% transits every 3.52 days. Both detectors agree.")
    if st.button("Analyse Kepler-8", type="primary", key="qs_k8", icon=":material/rocket_launch:"):
        quick_start("kepler8_q2")
        st.switch_page(PAGES["analyze"])
with c2.container(border=True, height="stretch"):
    st.markdown("**Kepler-10 · real data**")
    st.caption("A rocky planet with tiny ~130 ppm transits. The threshold method misses most; BLS finds them.")
    if st.button("Analyse Kepler-10", key="qs_k10", icon=":material/rocket_launch:"):
        quick_start("kepler10_q3")
        st.switch_page(PAGES["analyze"])
with c3.container(border=True, height="stretch"):
    st.markdown("**Synthetic star · known answer**")
    st.caption("Generated data with injected transits, so precision and recall can be measured exactly.")
    if st.button("Analyse synthetic data", key="qs_syn", icon=":material/science:"):
        set_dataset(cached_synthetic(asdict(SyntheticParams())))
        run_current(settings())
        st.switch_page(PAGES["analyze"])

if analysis() is not None:
    st.success(f"An analysis of **{analysis().dataset_meta.get('target')}** is loaded — every page below uses it.",
               icon=":material/check_circle:")

st.subheader("How it works")
steps = [
    (":material/travel_explore:", "1 · Retrieve", "Kepler/TESS light curves from MAST (cached), bundled real samples, or synthetic data."),
    (":material/tune:", "2 · Preprocess", "Clean, normalise, detrend and clip outliers — every step logged and explained."),
    (":material/monitoring:", "3 · Detect", "Improved threshold detector and Box Least Squares, run side by side."),
    (":material/science:", "4 · Discrete Math Lab", "Sets, number theory, graphs, relations and combinatorics on your results."),
    (":material/fact_check:", "5 · Evaluate & export", "Explainable evidence checklist, history, CSV/JSON/HTML reports."),
]
cols = st.columns(len(steps))
for col, (icon, title, text) in zip(cols, steps):
    with col.container(border=True, height="stretch"):
        st.markdown(f"{icon} **{title}**")
        st.caption(text)

st.subheader("Where the mathematics lives")
modules = [
    ("sets", "Set Theory", "A ∪ B, A ∩ B, A − B, complements and a Venn diagram of which detector found which transit."),
    ("numbers", "Number Theory", "Intervals as integer multiples of P, congruences mod P, approximate GCD and a modular clock."),
    ("graphs", "Graph Theory", "Events as vertices, periodic consistency as edges; components, cliques and non-transitivity."),
    ("relations", "Relations & Functions", "Observation → flux is a function; event ↔ algorithm is many-to-many."),
    ("combinatorics", "Combinatorics", "Cartesian products of settings, constrained counting and C(n, k) experiments."),
]
cols = st.columns(len(modules))
for col, (key, title, text) in zip(cols, modules):
    with col.container(border=True, height="stretch"):
        st.page_link(PAGES[key], label=title, icon=":material/arrow_forward:")
        st.caption(text)

st.caption("Data: NASA Kepler & TESS missions via the Mikulski Archive for Space Telescopes (MAST); reference "
           "parameters from the NASA Exoplanet Archive. Built with Lightkurve, Astropy, NetworkX, Plotly and Streamlit.")
