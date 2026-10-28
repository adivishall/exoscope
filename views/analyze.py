import numpy as np
import pandas as pd
import streamlit as st

from exoscope import plots
from exoscope.detection.bls import BLSSettings
from exoscope.detection.threshold import ThresholdSettings
from exoscope.pipeline import AnalysisSettings, default_settings_for, diff_results, diff_settings
from exoscope.processing import PreprocessSettings
from exoscope.ui.common import (PAGES, analysis, dataset_header, metric_cols, plot, references, require_dataset, run_current,
                                settings)

st.title("Analyze Light Curves")
data = require_dataset()
dataset_header(data)

DURATION_CHOICES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]
st.session_state.setdefault("form_version", 0)
v = st.session_state.form_version
cur: AnalysisSettings = settings()
summ = data.summary()
baseline = summ.get("baseline_days", 30.0)

# ------------------------------------------------------------------ settings
with st.expander("Analysis settings", expanded=analysis() is None, icon=":material/tune:"):
    with st.form(f"settings_{v}"):
        st.markdown("**Preprocessing**")
        p1, p2, p3, p4 = st.columns(4)
        pre = cur.preprocess
        quality = p1.toggle("Remove quality-flagged cadences", pre.remove_quality_flagged)
        flatten = p1.toggle("Detrend (flatten)", pre.flatten)
        window = p2.slider("Flatten window [days]", 0.1, 5.0, float(pre.flatten_window_days), 0.05,
                           help="Should be ≥ 3× the transit duration, or the trend will eat the transit.")
        poly = p2.select_slider("Polynomial order", [1, 2, 3], value=pre.flatten_polyorder)
        up_on = p3.toggle("Clip upward outliers", pre.clip_upper_sigma is not None)
        up = p3.slider("Upper clip [σ]", 2.0, 10.0, float(pre.clip_upper_sigma or 4.0), 0.5)
        lo_on = p3.toggle("Clip downward outliers", pre.clip_lower_sigma is not None,
                          help="Usually OFF: downward clipping removes transit points.")
        lo = p3.slider("Lower clip [σ]", 2.0, 20.0, float(pre.clip_lower_sigma or 6.0), 0.5)
        bin_on = p4.toggle("Bin data", pre.bin_minutes is not None, help="Speeds up BLS on 2-minute data.")
        bin_min = p4.slider("Bin size [minutes]", 2.0, 60.0, float(pre.bin_minutes or 10.0), 1.0)

        st.markdown("**Detection**")
        d1, d2, d3, d4 = st.columns(4)
        run_thr = d1.toggle("Threshold detector", cur.run_threshold)
        run_bls = d1.toggle("Box Least Squares", cur.run_bls)
        thr = cur.threshold
        k = d2.slider("Threshold k [σ below median]", 1.5, 8.0, float(thr.sigma_threshold), 0.1)
        min_pts = d2.slider("Min consecutive points per event", 1, 10, int(thr.min_points))
        gap_pts = d2.slider("Allowed unflagged points inside an event", 0, 5, int(thr.max_gap_points))
        b = cur.bls
        pmin = d3.number_input("BLS min period [d]", 0.1, max(0.11, baseline / 2), float(min(b.period_min, baseline / 2)),
                               0.05, format="%.2f")
        pmax = d3.number_input("BLS max period [d] (0 = half the baseline)", 0.0, float(baseline),
                               float(b.period_max or 0.0), 0.5, format="%.2f")
        durs = d4.multiselect("BLS trial durations [h]", DURATION_CHOICES,
                              [x for x in b.durations_hours if x in DURATION_CHOICES] or [1.0, 2.0, 3.0])
        n_sig = d4.select_slider("Signals to search for", [1, 2, 3], value=b.n_signals,
                                 help="After each signal is found its transits are masked and BLS runs again.")
        grid = d4.select_slider("Max trial periods", [5_000, 10_000, 30_000, 60_000, 100_000], value=b.max_periods)
        st.markdown("**Comparison**")
        tol = st.slider("Event-matching tolerance τ [hours]", 0.25, 6.0, float(cur.match_tolerance_hours), 0.25,
                        help="Events from the two methods closer than τ count as the same element in set comparisons.")
        submitted = st.form_submit_button("Run analysis", type="primary", icon=":material/play_arrow:")
    if st.button("Reset to default settings", icon=":material/restart_alt:"):
        st.session_state.settings = default_settings_for(data)
        st.session_state.form_version += 1
        st.rerun()

if submitted:
    try:
        new = AnalysisSettings(
            preprocess=PreprocessSettings(quality, flatten, window, poly, up if up_on else None,
                                          lo if lo_on else None, bin_min if bin_on else None),
            threshold=ThresholdSettings(k, min_pts, gap_pts, tol),
            bls=BLSSettings(period_min=pmin, period_max=pmax or None, durations_hours=tuple(sorted(durs)),
                            max_periods=grid, n_signals=n_sig),
            run_threshold=run_thr, run_bls=run_bls, match_tolerance_hours=tol,
        )
        new.validate()
        if run_bls:
            new.bls.validate()
        run_current(new)
    except ValueError as exc:
        st.error(str(exc))

r = analysis()
if r is None:
    st.info("Choose settings above and press **Run analysis**.", icon=":material/info:")
    st.stop()

# ------------------------------------------------------------------ headline
best = r.best
thr_res = r.threshold
m = metric_cols(6)
m[0].metric("Points analysed", f"{r.processed.n_points:,}")
m[1].metric("Noise (robust σ)", f"{r.processed.noise * 1e6:,.0f} ppm")
m[2].metric("Threshold events", len(thr_res.events) if thr_res else "—")
m[3].metric("BLS period", f"{best.period:.5f} d" if best else "—")
m[4].metric("BLS depth", f"{best.depth * 1e6:,.0f} ppm" if best else "—")
m[5].metric("BLS SNR", f"{best.depth_snr:.1f}" if best else "—")
st.caption(f"{r.evaluation.verdict} · {r.evaluation.n_pass}/{len(r.evaluation.applicable)} checks passed · "
           f"total runtime {r.timings.get('total_s', 0):.2f} s")

tabs = st.tabs(["Light curves", "Threshold", "BLS", "Compare methods", "Experiment mode", "Preprocessing explained"])

with tabs[0]:
    plot(plots.fig_raw(data.time, data.flux, data.quality_ok, data.time_format), key="lc_raw")
    if r.settings.preprocess.flatten:
        plot(plots.fig_detrend(r.processed, data.time_format), key="lc_detrend")
    t0_, t1_ = float(r.processed.time[0]), float(r.processed.time[-1])
    zoom = st.slider("Zoom time window", t0_, t1_, (t0_, t1_), format="%.2f")
    plot(plots.fig_detections(r.processed, thr_res, best, data.time_format, xrange=zoom), key="lc_det")

with tabs[1]:
    if thr_res is None:
        st.info("The threshold detector was not run.")
    else:
        c = metric_cols(5)
        c[0].metric("Events", len(thr_res.events))
        c[1].metric("Flagged points", thr_res.n_flagged_points)
        c[2].metric("Rejected short runs", thr_res.rejected_runs, help="Runs with fewer than the minimum points")
        c[3].metric("Threshold level", f"{(1 - thr_res.threshold_level) * 1e6:,.0f} ppm below median")
        pe = thr_res.period
        c[4].metric("Period (approx. GCD)", f"{pe.period:.5f} d" if pe and pe.members and not pe.ambiguous else "—")
        if pe:
            st.caption(f"Period estimate: {pe.note} "
                       + (f"P = {pe.period:.5f} ± {pe.period_err:.5f} d." if pe.members and not pe.ambiguous else ""))
        st.markdown("Points below *median − k·σ* are grouped into runs; runs with at least the minimum number of "
                    "points become events. The period comes from the **Number Theory** module (largest set of events "
                    "congruent modulo P).")
        if thr_res.events:
            df = pd.DataFrame([e.to_dict() for e in thr_res.events])
            st.dataframe(df[["event_id", "t_center", "depth_ppm", "duration_hours", "snr", "n_points", "epoch"]],
                         hide_index=True, height=min(400, 38 + 35 * len(df)),
                         column_config={"t_center": st.column_config.NumberColumn(format="%.4f"),
                                        "depth_ppm": st.column_config.NumberColumn(format="%.0f"),
                                        "duration_hours": st.column_config.NumberColumn(format="%.2f"),
                                        "snr": st.column_config.NumberColumn(format="%.1f")})
        else:
            st.warning("No events crossed the threshold. Shallow transits (e.g. Kepler-10 b) are often below the "
                       "per-point noise — try BLS, or lower k (expect more false events).")

with tabs[2]:
    if r.bls is None or best is None:
        st.info("BLS was not run.")
    else:
        cands = r.bls.candidates
        idx = 0
        if len(cands) > 1:
            idx = st.radio("Signal", range(len(cands)), horizontal=True,
                           format_func=lambda i: f"#{i + 1}: P = {cands[i].period:.4f} d")
        cand = cands[idx]
        c = metric_cols(6)
        c[0].metric("Period", f"{cand.period:.5f} d")
        c[1].metric("Duration", f"{cand.duration * 24:.2f} h")
        c[2].metric("Depth", f"{cand.depth * 1e6:,.0f} ± {cand.depth_err * 1e6:,.0f} ppm")
        c[3].metric("Depth SNR", f"{cand.depth_snr:.1f}")
        c[4].metric("SDE", f"{cand.sde:.1f}", help="(peak − mean power) / std of the periodogram")
        c[5].metric("Transits observed", f"{cand.n_transits_observed}/{cand.n_transits_expected}")
        refs, _ = references()
        plot(plots.fig_periodogram(cand, refs), key=f"pg_{idx}")
        plot(plots.fig_folded(r.processed, cand), key=f"fold_{idx}")
        st.caption(f"{r.bls.n_periods:,} trial periods from {r.settings.bls.period_min:g} to "
                   f"{r.bls.period_max_used:.3g} d{' (grid capped — raise “Max trial periods” for a finer grid)' if r.bls.grid_capped else ''}; "
                   f"{r.bls.version}; {r.bls.runtime_s:.2f} s."
                   + (f" Per-point errors were scaled ×{r.bls.error_scale:.2f} to match the measured scatter, so the "
                      "SNR is not optimistic." if r.bls.error_scale > 1.01 else ""))

with tabs[3]:
    rows = []
    thr_p = thr_res.period if thr_res else None
    rows.append({"quantity": "Type of output", "Threshold": "individual dips", "BLS": "one periodic signal → epochs"})
    rows.append({"quantity": "Events", "Threshold": len(thr_res.events) if thr_res else None,
                 "BLS": best.n_transits_observed if best else None})
    rows.append({"quantity": "Period [d]",
                 "Threshold": round(thr_p.period, 5) if thr_p and thr_p.members and not thr_p.ambiguous else None,
                 "BLS": round(best.period, 5) if best else None})
    if thr_res and thr_res.events:
        rows.append({"quantity": "Median event depth [ppm]",
                     "Threshold": round(float(np.median([e.depth for e in thr_res.events])) * 1e6),
                     "BLS": round(best.depth * 1e6) if best else None})
        rows.append({"quantity": "Median event duration [h]",
                     "Threshold": round(float(np.median([e.duration for e in thr_res.events])) * 24, 2),
                     "BLS": round(best.duration * 24, 2) if best else None})
    rows.append({"quantity": "Signal strength", "Threshold": f"median event SNR {np.median([e.snr for e in thr_res.events]):.1f}"
                 if thr_res and thr_res.events else "—", "BLS": f"SNR {best.depth_snr:.1f}, SDE {best.sde:.1f}" if best else "—"})
    rows.append({"quantity": "Runtime [s]", "Threshold": round(r.timings.get("threshold_s", np.nan), 4),
                 "BLS": round(r.timings.get("bls_s", np.nan), 3)})
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True)
    s = r.comparison.summary()
    if r.comparison.bls_available and thr_res is not None:
        plot(plots.fig_method_comparison([
            {"metric": "events detected", "threshold": s["|A|"], "bls": s["|B|"]},
            {"metric": "found by both (A∩B)", "threshold": s["|A∩B|"], "bls": s["|A∩B|"]},
            {"metric": "found only by this method", "threshold": s["|A−B|"], "bls": s["|B−A|"]},
        ]), key="cmp_bar")
        st.markdown(
            f"- **Agreement:** {s['|A∩B|']} events are found by both (Jaccard {s['jaccard']:.2f}). "
            f"{s['|B−A|']} BLS epochs were missed by the threshold detector — typically transits shallower than "
            f"its per-point threshold; {s['|A−B|']} threshold events have no BLS counterpart — candidates for noise, "
            "systematics or a second signal.\n"
            "- **Why they differ:** the threshold method judges each dip on its own, so it needs every transit to "
            "stand out from the noise. BLS adds up all transits at a trial period, so its sensitivity grows with the "
            "number of transits.\n"
            "- See the **Set Theory** page for the full set comparison.")
        st.page_link(PAGES["sets"], label="Open Set Theory", icon=":material/join_inner:")
    if r.truth_metrics:
        st.markdown("**Against the injected truth (synthetic data)**")
        tm = r.truth_metrics
        trows = []
        for meth in ("threshold", "bls"):
            if meth in tm:
                trows.append({"method": meth, **{k: tm[meth][k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                              "period error %": tm[meth].get("period_error_percent")})
        st.dataframe(pd.DataFrame(trows), hide_index=True)
    if r.reference_check:
        rc = r.reference_check
        st.markdown(f"**Reference check:** detected P = {rc['detected_period']:.5f} d vs **{rc['planet']}** "
                    f"P = {rc['reference_period']:.5f} d (NASA Exoplanet Archive) → ratio ≈ {rc['fraction']} "
                    f"(*{rc['kind']}*), period error {rc['period_error_percent']:.3f}%.")

with tabs[4]:
    prev = st.session_state.get("previous_analysis")
    st.markdown("Change any setting above and run again — this tab shows **what you changed** and **how the "
                "output changed** compared with the previous run on the same dataset.")
    if prev is None:
        st.info("Run the analysis a second time with different settings to compare.", icon=":material/compare_arrows:")
    else:
        sd = diff_settings(prev.settings, r.settings)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Settings changed**")
            if sd:
                st.dataframe(pd.DataFrame(sd).astype(str), hide_index=True)
            else:
                st.caption("No settings changed (identical rerun).")
        with c2:
            st.markdown("**Results**")
            rd = pd.DataFrame(diff_results(prev, r))
            st.dataframe(rd.astype(str), hide_index=True,
                         column_config={"changed": st.column_config.TextColumn("changed?")})

with tabs[5]:
    st.dataframe(pd.DataFrame(r.processed.steps), hide_index=True)
    st.markdown(
        "- **Normalising** puts every star on the same scale (1.0 = normal brightness) so depths read as fractions.\n"
        "- **Detrending** removes slow variations (spots, rotation, instrument drifts). A window that is too short "
        "follows the transit itself and makes it shallower — compare windows of 0.25 d and 2 d in Experiment mode.\n"
        "- **Upward clipping** removes cosmic rays and flares without touching transits. Turning on *downward* "
        "clipping can delete the very points that make up a transit.\n"
        "- **Binning** averages neighbouring points: less noise per point and faster BLS, but short transits get "
        "smeared if the bins are too wide.")
    if r.processed.clipped_time is not None:
        st.caption(f"{len(r.processed.clipped_time)} points were clipped as outliers.")
