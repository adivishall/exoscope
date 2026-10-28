from dataclasses import asdict

import pandas as pd
import streamlit as st

from exoscope import plots
from exoscope.data.model import DataError
from exoscope.data.retrieval import RetrievalError
from exoscope.data.synthetic import SyntheticParams
from exoscope.data.targets import MISSIONS, TARGETS, validate_target_name
from exoscope.ui.common import (metric_cols, PAGES, cached_download, cached_sample, cached_samples, cached_search,
                                cached_synthetic, dataset, dataset_header, fmt, plot, references, set_dataset)

st.title("Explore Targets")
st.caption("Load a light curve: search MAST for real Kepler/TESS observations, use a bundled real sample "
           "(works offline), or generate synthetic data with a known answer.")

tab_search, tab_samples, tab_synth = st.tabs(["Search MAST (real data)", "Bundled real samples", "Synthetic generator"])

# ------------------------------------------------------------------ MAST search
with tab_search:
    st.markdown("Pick a curated target or type any Kepler/TESS identifier (e.g. `Kepler-8`, `KIC 11904151`, "
                "`TIC 100100827`, `WASP-18`).")
    curated = {f"{t.name} ({t.mission}) — {t.description}": t for t in TARGETS}
    c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="bottom")
    choice = c1.selectbox("Curated targets", ["— type my own —"] + list(curated), index=1)
    default_name = curated[choice].name if choice in curated else ""
    default_mission = curated[choice].mission if choice in curated else "Kepler"
    name = c2.text_input("Target name / ID", value=default_name, max_chars=40)
    mission = c3.radio("Mission", MISSIONS, index=MISSIONS.index(default_mission), horizontal=False)
    if choice in curated:
        t = curated[choice]
        st.caption(f"Difficulty for a threshold detector: **{t.difficulty}**")

    if st.button("Search MAST", type="primary", icon=":material/search:"):
        try:
            clean = validate_target_name(name)
            with st.spinner(f"Searching MAST for {clean} ({mission})… (first search can take up to a minute)"):
                st.session_state.search = {"target": clean, "mission": mission,
                                           "products": cached_search(clean, mission)}
        except ValueError as exc:
            st.error(str(exc))
        except RetrievalError as exc:
            st.error(str(exc), icon=":material/cloud_off:")
            st.caption("Tip: the **Bundled real samples** tab works without a network connection.")

    search = st.session_state.get("search")
    if search:
        products = search["products"]
        st.markdown(f"**{len(products)}** {search['mission']} light curve(s) found for **{search['target']}** "
                    "(PDCSAP flux; Kepler long cadence / TESS 2-minute SPOC).")
        df = pd.DataFrame(products)[["index", "dataset", "year", "author", "exptime_s", "target_name"]]
        st.dataframe(df, hide_index=True, height=min(300, 38 + 35 * len(df)),
                     column_config={"exptime_s": st.column_config.NumberColumn("exposure [s]")})
        labels = [f"#{p['index']} · {p['dataset']} ({p['year']})" for p in products]
        pick = st.selectbox("Observation to load", range(len(products)), format_func=lambda i: labels[i])
        if st.button("Download and load", icon=":material/download:"):
            try:
                with st.spinner(f"Downloading {products[pick]['dataset']}… (cached after the first time)"):
                    data = cached_download(search["target"], search["mission"], pick, products)
                set_dataset(data)
                st.success(f"Loaded {data.label}.")
            except (RetrievalError, DataError, ValueError) as exc:
                st.error(str(exc), icon=":material/error:")

# ------------------------------------------------------------------ bundled samples
with tab_samples:
    st.markdown("Exact offline copies of archived Kepler/TESS light curves, saved by `scripts/fetch_samples.py`. "
                "They are **real observations** — useful when the network is slow during a demonstration.")
    samples = cached_samples()
    if not samples:
        st.warning("No bundled samples found. Run `python scripts/fetch_samples.py` to create them.")
    cols = st.columns(max(1, len(samples)))
    for col, s in zip(cols, samples):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{s.get('target')}** · {s.get('mission')} {s.get('dataset')}")
            st.caption(f"Retrieved {s.get('retrieved_on')} · product `{s.get('product', '')}`")
            if st.button("Load", key=f"load_{s['sample_id']}", icon=":material/upload:"):
                set_dataset(cached_sample(s["sample_id"]))
                st.success(f"Loaded {s.get('target')}.")

# ------------------------------------------------------------------ synthetic
with tab_synth:
    st.warning("Synthetic light curves are **not real stars**. They are labelled SYNTHETIC everywhere in the app "
               "and exist so you can test the detectors against a known answer.", icon=":material/science:")
    d = SyntheticParams()
    with st.form("synthetic"):
        c1, c2, c3 = st.columns(3)
        period = c1.slider("Orbital period [days]", 0.5, 15.0, d.period_days, 0.05)
        depth = c1.slider("Transit depth [ppm]", 50, 30_000, int(d.depth_ppm), 50)
        duration = c1.slider("Transit duration [hours]", 0.5, 12.0, d.duration_hours, 0.25)
        noise = c2.slider("White noise [ppm per point]", 0, 5_000, int(d.noise_ppm), 25)
        var_amp = c2.slider("Stellar variability amplitude [ppm]", 0, 20_000, int(d.variability_amp_ppm), 100)
        var_p = c2.slider("Variability period [days]", 0.5, 30.0, d.variability_period_days, 0.5)
        baseline = c3.slider("Observation length [days]", 5.0, 90.0, d.baseline_days, 1.0)
        cadence = c3.select_slider("Cadence [minutes]", [2.0, 10.0, 30.0], value=d.cadence_minutes)
        gaps = c3.slider("Data gaps", 0, 5, d.n_gaps)
        with st.expander("More options"):
            e1, e2 = st.columns(2)
            outliers = e1.slider("Outlier fraction", 0.0, 0.02, d.outlier_fraction, 0.001, format="%.3f")
            missing = e1.slider("Missing-value fraction", 0.0, 0.2, d.missing_fraction, 0.005, format="%.3f")
            odd = e2.slider("Odd/even depth ratio (≠ 1 imitates an eclipsing binary)", 0.2, 2.0, 1.0, 0.05)
            seed = e2.number_input("Random seed", 0, 10_000, d.seed)
        submitted = st.form_submit_button("Generate synthetic light curve", type="primary", icon=":material/science:")
    if submitted:
        params = SyntheticParams(period_days=period, depth_ppm=float(depth), duration_hours=duration,
                                 noise_ppm=float(noise), variability_amp_ppm=float(var_amp),
                                 variability_period_days=var_p, baseline_days=baseline, cadence_minutes=cadence,
                                 n_gaps=gaps, outlier_fraction=outliers, missing_fraction=missing,
                                 odd_depth_ratio=odd, seed=int(seed))
        try:
            params.validate()
            set_dataset(cached_synthetic(asdict(params)))
            st.success("Synthetic light curve generated.")
        except ValueError as exc:
            st.error(str(exc))

# ------------------------------------------------------------------ current dataset
data = dataset()
st.divider()
if data is None:
    st.info("No light curve loaded yet — choose one above.", icon=":material/info:")
    st.stop()

st.subheader("Loaded light curve")
dataset_header(data)
s = data.summary()
m = metric_cols(6)
m[0].metric("Measurements", f"{s['n_points']:,}")
m[1].metric("Usable", f"{s['n_usable']:,}", help="Finite flux and not quality-flagged")
m[2].metric("Baseline", fmt(s.get("baseline_days"), 4, " d"))
m[3].metric("Cadence", fmt(s.get("cadence_minutes"), 3, " min"))
m[4].metric("Point-to-point scatter", fmt(s.get("p2p_scatter_ppm"), 3, " ppm"))
m[5].metric("Completeness", f"{s['completeness']:.1%}")
q = metric_cols(4)
q[0].metric("Missing (NaN)", s["n_nan"])
q[1].metric("Quality-flagged", s["n_quality_flagged"])
q[2].metric("Largest gap", fmt(s.get("largest_gap_days"), 3, " d"))
q[3].metric("Duty cycle", fmt(s.get("duty_cycle"), 3), help="Fraction of the baseline covered by usable cadences")

with st.expander("Source and header information"):
    st.markdown(f"**Source:** {data.source_label}")
    meta_rows = {k: v for k, v in data.meta.items() if k not in ("synthetic_params",)}
    st.dataframe(pd.DataFrame({"key": list(meta_rows), "value": [str(v) for v in meta_rows.values()]}),
                 hide_index=True)

plot(plots.fig_raw(data.time, data.flux, data.quality_ok, data.time_format,
                   f"Original light curve — {data.label}"), key="raw_explore")

if data.is_real:
    refs, provenance = references()
    with st.container(border=True):
        st.markdown("**Known planets of this star (external reference, used only for validation)**")
        if refs:
            st.dataframe(pd.DataFrame([{"planet": p.name, "period [d]": p.period_days, "depth [%]": p.depth_percent,
                                        "duration [h]": p.duration_hours, "radius [R⊕]": p.radius_earth}
                                       for p in refs]), hide_index=True)
        st.caption(provenance or "No reference information.")
elif data.truth:
    t = data.truth
    st.caption(f"Injected truth: P = {t['period_days']} d, depth = {t['depth_ppm']:g} ppm, duration = "
               f"{t['duration_hours']} h, {sum(e['observed'] for e in t['epochs'])} observed transits.")

st.page_link(PAGES["analyze"], label="Analyze this light curve", icon=":material/arrow_forward:")
