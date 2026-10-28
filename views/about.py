import numpy as np
import pandas as pd
import streamlit as st

from exoscope import __version__
from exoscope.detection.bls import ALGORITHM_VERSION as BLS_VERSION
from exoscope.detection.threshold import ALGORITHM_VERSION as THR_VERSION
from exoscope.legacy import legacy_detect
from exoscope.ui.common import analysis

st.title("About & Methodology")

tab_about, tab_methods, tab_legacy, tab_dm, tab_lit, tab_sdg, tab_limits = st.tabs(
    ["Project", "Algorithms", "v1 → v2 detector", "Discrete Mathematics map", "Literature", "SDG alignment",
     "Limitations"])

with tab_about:
    st.markdown(f"""
**ExoScope {__version__}** — an interactive exoplanet transit detection and analysis system, built as a
Discrete Mathematics course project.

**Novelty.** *An interactive exoplanet transit analysis platform that demonstrates how Discrete Mathematics concepts
can be applied to detect, compare, visualise and evaluate periodic astronomical events.* The mathematics is not a
side explanation: set operations decide how the detectors are compared, the approximate-GCD / congruence method
produces the threshold detector's period, the event graph exposes which dips support each other, relations
describe the pipeline's structure, and combinatorics drives systematic experiments.

**Objectives**
1. Retrieve and process real Kepler/TESS light curves reliably (with caching and offline samples).
2. Compare an improved threshold detector with Box Least Squares on the same data.
3. Implement five discrete-mathematics modules that operate on real analysis output.
4. Explain every result: evidence checklist, guided lessons, quizzes, glossary.
5. Make analyses reproducible and exportable (history database, CSV/JSON/HTML reports).

**Workflow:** retrieve → preprocess → detect (threshold, BLS) → compare (sets) → analyse periodicity (number theory,
graphs) → evaluate (checklist) → export.

**Data & tools.** NASA Kepler and TESS data from MAST, accessed with Lightkurve; reference parameters from the NASA
Exoplanet Archive; Astropy (BLS), NumPy/SciPy, NetworkX, Plotly, Streamlit, SQLite.
""")

with tab_methods:
    st.markdown(f"""
#### Preprocessing
NaN removal → mission quality mask → median normalisation → Savitzky–Golay detrending with iterative sigma-masking
(segments split at data gaps) → upward sigma-clipping → optional binning. Each step is logged with point counts.

#### Threshold detector (`{THR_VERSION}`)
1. Robust noise σ = 1.4826 × MAD of the processed flux.
2. Flag points below median − k·σ.
3. Group flagged cadences into runs (allowing a few unflagged points inside), keep runs with ≥ m points.
4. Each run becomes an event: flux-deficit-weighted centre, mean depth, duration, SNR = depth/σ·√n.
5. Period: **approximate GCD** — candidates Δᵢⱼ/k, scored by the size of the largest congruence class mod P
   times its fill fraction, refined by a least-squares fit of tₙ = t₀ + nP.

#### Box Least Squares (`{BLS_VERSION}`)
Astropy's `BoxLeastSquares` on a uniform frequency grid (Δf = D_min/T², capped for speed) over the trial durations.
Reports period, t₀, duration, depth ± error, depth SNR and SDE. Up to three signals by iterative masking. Each
signal is expanded into its predicted epochs; epochs with data are events (set B), epochs in gaps form G.

#### Comparison
Events are matched one-to-one within ±τ hours with the Hungarian algorithm; matched pairs and unmatched events
become the elements of U = A ∪ B ∪ G.

#### Candidate evaluation (explainable checklist)
Repeated events (≥ 3) · intervals as integer multiples of P · BLS SNR (≥ 10 pass, 7.1–10 marginal, < 7.1 not detected — the strongest noise peak among ~10⁵ trials is typically SNR 4–6) · odd/even core-depth agreement
(< 3σ, errors from epoch-to-epoch scatter) · box preferred over sinusoid · no deep secondary eclipse at phase 0.5 ·
data completeness and transit coverage · algorithm agreement (Jaccard) · depth < 3% · duration consistent with the
period. The verdict is *gated*: a failed SNR test means “no significant signal”; a failed false-positive test
(odd/even, secondary, shape, depth) means “signs of a non-planetary origin”. The pass count is **not** a probability.

#### Validation
`scripts/evaluate.py` measures period recovery, precision, recall and F1 on a seeded synthetic benchmark and period
agreement with the archive on the bundled real light curves. Results are in `docs/EVALUATION.md`.
""")

with tab_legacy:
    st.markdown("""
The original ExoScope (v1) flagged every point below **mean − 3·std** as a dip and took the **median spacing** of
flagged points as the period. v2 keeps that code (`exoscope/legacy.py`) so the two can be compared on the same data.
""")
    r = analysis()
    if r is None:
        st.info("Analyse a light curve to see the v1 and v2 detectors side by side.")
    else:
        lc = r.processed
        leg = legacy_detect(lc.time, lc.flux)
        thr = r.threshold
        ref = r.reference_check["reference_period"] if r.reference_check else (
            r.truth_metrics["true_period"] if r.truth_metrics else None)
        rows = [
            {"": "Threshold level", "v1 (legacy)": f"mean − 3·std = {(1 - leg.threshold) * 1e6:,.0f} ppm below 1",
             "v2 (improved)": f"median − {thr.settings.sigma_threshold:g}·σ_MAD = {(1 - thr.threshold_level) * 1e6:,.0f} ppm below 1" if thr else "—"},
            {"": "What counts as a dip", "v1 (legacy)": f"{len(leg.dip_times)} individual points",
             "v2 (improved)": f"{len(thr.events)} grouped events (≥ {thr.settings.min_points} points)" if thr else "—"},
            {"": "Period estimate", "v1 (legacy)": f"{leg.period_estimate:.5f} d (median spacing)" if leg.period_estimate else "none",
             "v2 (improved)": f"{thr.period.period:.5f} d (approximate GCD)" if thr and thr.period and thr.period.members and not thr.period.ambiguous else "none"},
        ]
        if ref:
            v1_err = abs(leg.period_estimate - ref) / ref * 100 if leg.period_estimate else np.nan
            v2_err = (abs(thr.period.period - ref) / ref * 100 if thr and thr.period and thr.period.members
                      and not thr.period.ambiguous else np.nan)
            rows.append({"": f"Period error vs reference ({ref:.5f} d)", "v1 (legacy)": f"{v1_err:.2f}%",
                         "v2 (improved)": f"{v2_err:.3f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Computed live on the current processed light curve. The v1 median spacing collapses towards the "
                   "cadence when several points of one transit are flagged, and doubles when transits are missed.")

with tab_dm:
    st.dataframe(pd.DataFrame([
        {"concept": "Sets, union/intersection/difference, complement", "where": "Set Theory; method comparison; evaluation",
         "what it does": "Compares detectors event by event; precision/recall as |D ∩ T|, |D − T|, |T − D|."},
        {"concept": "Inclusion–exclusion, Jaccard index", "where": "Set Theory; agreement criterion",
         "what it does": "Counts and quantifies agreement between methods."},
        {"concept": "Divisibility with tolerance, integer multiples", "where": "Number Theory; interval criterion",
         "what it does": "Tests whether gaps are whole numbers of orbits; reveals missed transits."},
        {"concept": "Congruence mod P, approximate GCD", "where": "Threshold detector period; Number Theory",
         "what it does": "Finds the period as the largest set of events congruent modulo P."},
        {"concept": "Euclid's algorithm, rational approximation", "where": "Number Theory",
         "what it does": "Exact integer GCDs (and why they fail with noise); alias detection P ≈ (p/q)·P'."},
        {"concept": "Graphs, components, cliques, adjacency matrix", "where": "Graph Theory; report",
         "what it does": "Network of mutually consistent events; isolated vertices flag noise."},
        {"concept": "Relations, functions, injectivity, order relations", "where": "Relations & Functions",
         "what it does": "Describes the pipeline mappings; non-transitivity of tolerance relations."},
        {"concept": "Cartesian product, rule of product, constraints, C(n, k), projections",
         "where": "Combinatorics; experiment runner",
         "what it does": "Enumerates and counts configurations, shares work across configurations."},
    ]), hide_index=True)

with tab_lit:
    st.markdown("How ExoScope relates to the background reading (*Finding Exoplanets with the Transit Method*, a guide "
                "to six papers and one software tool). **Implemented** means it is in the code; **not implemented** is "
                "stated plainly.")
    st.dataframe(pd.DataFrame([
        {"work": "Collier Cameron et al. 2006 — fast hybrid BLS search (SuperWASP)",
         "idea": "Remove systematics first; coarse BLS grid, refine near peaks",
         "in ExoScope": "Implemented in spirit: detrending before search; frequency grid sized from the shortest "
                        "duration and capped. Not implemented: SysRem (needs many stars per frame), Newton–Raphson refinement."},
        {"work": "Giacobbe et al. 2012 — cool-star pilot survey (APACHE)",
         "idea": "Injection tests give detection limits; red noise ≈ 1.3× worse than white",
         "in ExoScope": "Implemented: injection–recovery into real Kepler light curves vs white-noise twins "
                        "(scripts/evaluate.py → docs/EVALUATION.md §4)."},
        {"work": "Sanchis-Ojeda et al. 2014 — ultra-short-period planets",
         "idea": "FFT search for sub-day periods",
         "in ExoScope": "Not implemented (BLS only). Kepler-10 b (0.84 d) is still recovered by BLS."},
        {"work": "Hippke & Heller 2019 — Transit Least Squares",
         "idea": "Realistic transit shape instead of a box",
         "in ExoScope": "Not implemented; the box-model limitation is documented. The synthetic generator does "
                        "inject trapezoidal transits, so BLS is tested on non-box shapes."},
        {"work": "Shallue & Vanderburg 2018 — CNN vetting",
         "idea": "Machine learning replaces manual vetting after the search",
         "in ExoScope": "Not implemented; vetting is an explainable rule checklist (odd/even, secondary eclipse, "
                        "shape, depth, duration, SNR) by design, so each decision can be explained."},
        {"work": "Rodenbeck et al. 2018 — Kepler-1625b exomoon",
         "idea": "Preprocessing choices can create a 'discovery'; test weak signals by injection",
         "in ExoScope": "Implemented: Experiment mode and the Combinatorics explorer show how settings change results; "
                        "injection–recovery measures what is genuinely detectable."},
        {"work": "lightkurve (KeplerGO)",
         "idea": "Download, clean and analyse Kepler/TESS data in Python",
         "in ExoScope": "Used for MAST search/download and quality flags; detrending and BLS are run by ExoScope's own "
                        "code (SciPy Savitzky–Golay, Astropy BoxLeastSquares)."},
    ]), hide_index=True, column_config={"in ExoScope": st.column_config.TextColumn(width="large")})

with tab_sdg:
    st.markdown("""
#### Primary: SDG 9 — Industry, Innovation and Infrastructure (Target 9.5)
Target 9.5 calls for enhancing scientific research and upgrading technological capabilities. ExoScope contributes in a
small, educational way: it builds open, reproducible computational methods on publicly funded space-mission data,
documents its algorithms and limitations, and lets students run and compare research-grade techniques (BLS) on real
observations.

#### Secondary: SDG 4 — Quality Education
Guided lessons, worked examples, quizzes with feedback and interactive experiments connect abstract discrete
mathematics to a real scientific problem, supporting active learning.

*These are statements of alignment and intended contribution; the application does not by itself achieve the UN
goals.*
""")

with tab_limits:
    st.markdown("""
- **Candidates only.** ExoScope cannot confirm planets; no centroid, pixel-level, spectroscopic or radial-velocity
  vetting is done.
- **Box model.** BLS fits a box; limb darkening, ingress/egress shape and grazing transits are not modelled, so depths
  are approximate.
- **One segment at a time.** Each analysis uses one Kepler quarter / TESS sector; long-period planets (fewer than 3
  transits in the segment) cannot be confirmed as periodic.
- **Checklist thresholds** (SNR 7.1, 3σ, 3%, …) are conventions for teaching, not a calibrated false-positive rate.
- **Synthetic benchmark** uses white noise and sinusoidal variability; real stars have red noise and systematics, so
  synthetic precision/recall overestimate real-world performance.
- **Reference labels** come from the NASA Exoplanet Archive and only cover known planets; an unmatched signal is not
  automatically false, and agreement with a known period does not validate other signals.
- **Network** access to MAST is needed for new targets; the first download of a light curve can take a minute.
""")
