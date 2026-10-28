import re

import pandas as pd
import streamlit as st

from exoscope import plots, reporting
from exoscope.ui.common import metric_cols, STATUS_ICON, dataset_header, fmt, plot, references, require_analysis, require_dataset

st.title("Candidate Results")
data = require_dataset()
r = require_analysis()
dataset_header(data)

ev = r.evaluation
level_style = {"strong": st.success, "possible": st.warning, "non-planetary": st.error, "noise": st.error,
               "none": st.info}
level_style[ev.level](f"**{ev.verdict}**  \n{ev.n_pass} of {len(ev.applicable)} applicable checks passed. "
                      "This is a count of satisfied evidence criteria — **not** a probability that a planet exists.")

st.subheader("Evidence checklist")
st.dataframe(
    pd.DataFrame([{"": STATUS_ICON[c.status], "check": c.name, "status": c.status, "measured": c.value,
                   "rule": c.rule} for c in ev.criteria]),
    hide_index=True, column_config={"": st.column_config.TextColumn(width="small")},
)
with st.expander("Why each check matters"):
    for c in ev.criteria:
        st.markdown(f"**{c.name}** — {c.explanation}")

st.subheader("Candidate summaries")
cands = reporting.candidates_frame(r)
if cands.empty:
    st.info("No periodic candidate was produced. Run BLS or adjust the threshold settings.")
else:
    st.dataframe(cands, hide_index=True)
best = r.best
if best:
    c = metric_cols(5)
    c[0].metric("Period", f"{best.period:.5f} d")
    c[1].metric("Mid-transit t₀", f"{best.t0:.4f}")
    c[2].metric("Depth", f"{best.depth * 1e6:,.0f} ppm")
    ratio = best.depth ** 0.5 if best.depth > 0 else float("nan")
    c[3].metric("Rp/Rs ≈ √δ", fmt(ratio, 3))
    c[4].metric("Odd/even difference", fmt(best.odd_even_sigma, 2, "σ"))
    plot(plots.fig_folded(r.processed, best), key="res_fold")

if r.reference_check:
    rc = r.reference_check
    st.info(f"**External reference:** the archive lists **{rc['planet']}** with P = {rc['reference_period']:.5f} d. "
            f"ExoScope's period is {rc['detected_period']:.5f} d — ratio ≈ {rc['fraction']} (*{rc['kind']}*), "
            f"error {rc['period_error_percent']:.3f}%. The planet's status comes from the archive's independent "
            "confirmation, not from this analysis.", icon=":material/public:")
if r.truth_metrics:
    tm = r.truth_metrics
    st.info("**Synthetic data — measured against the injected truth:** " + "; ".join(
        f"{m}: precision {tm[m]['precision']:.2f}, recall {tm[m]['recall']:.2f}, F1 {tm[m]['f1']:.2f}"
        + (f", period error {tm[m]['period_error_percent']:.3f}%" if "period_error_percent" in tm[m] else "")
        for m in ("threshold", "bls") if m in tm), icon=":material/science:")

st.subheader("Event table")
events = reporting.events_frame(r)
methods = st.pills("Show events from", ["threshold", "bls"], default=["threshold", "bls"], selection_mode="multi")
shown = events[events["method"].isin(methods or [])]
st.dataframe(shown, hide_index=True, height=min(420, 38 + 35 * max(len(shown), 1)),
             column_config={"t_center": st.column_config.NumberColumn(format="%.4f"),
                            "t_start": st.column_config.NumberColumn(format="%.4f"),
                            "t_end": st.column_config.NumberColumn(format="%.4f"),
                            "depth_ppm": st.column_config.NumberColumn(format="%.0f"),
                            "duration_hours": st.column_config.NumberColumn(format="%.2f"),
                            "snr": st.column_config.NumberColumn(format="%.1f")})

st.subheader("Export")
slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{data.target}_{data.meta.get('dataset', '')}").strip("_")[:50]
exp = st.session_state.get("experiment")
exp_info = exp.get("report_info") if exp else None
e = st.columns(5)
e[0].download_button("Events (CSV)", events.to_csv(index=False), f"exoscope_{slug}_events.csv", "text/csv",
                     icon=":material/table:")
e[1].download_button("Candidates (CSV)", cands.to_csv(index=False), f"exoscope_{slug}_candidates.csv", "text/csv",
                     icon=":material/table:")
e[2].download_button("Settings + metadata (JSON)", reporting.metadata_json(r), f"exoscope_{slug}_metadata.json",
                     "application/json", icon=":material/data_object:")
e[3].download_button("Report (Markdown)", reporting.markdown_report(r, exp_info), f"exoscope_{slug}_report.md",
                     "text/markdown", icon=":material/description:")
refs, _ = references()
figs = [("Processed light curve with detections", plots.fig_detections(r.processed, r.threshold, best, data.time_format))]
if best:
    figs += [("BLS periodogram", plots.fig_periodogram(best, refs)), ("Phase-folded transit", plots.fig_folded(r.processed, best))]
figs.append(("Set comparison", plots.fig_venn(r.comparison)))
e[4].download_button("Report (HTML, with charts)", reporting.html_report(r, figs, exp_info),
                     f"exoscope_{slug}_report.html", "text/html", icon=":material/web:")
st.caption("The HTML report loads Plotly from its CDN to draw the interactive charts. "
           + ("It includes your latest Combinatorics experiment." if exp_info else
              "Run a Combinatorics experiment to include its results in the report."))
