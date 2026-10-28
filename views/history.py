import pandas as pd
import streamlit as st

from exoscope import storage
from exoscope.data.model import DataError
from exoscope.data.reload import reload_dataset
from exoscope.data.retrieval import RetrievalError
from exoscope.pipeline import AnalysisSettings
from exoscope.ui.common import metric_cols, PAGES, run_current, set_dataset

st.title("Analysis History")
st.caption(f"Every analysis is saved automatically to a local SQLite database (`{storage.db_path().name}`; set "
           "`EXOSCOPE_DB_PATH` to change it). Settings, algorithm versions, candidates and events are stored; raw "
           "light curves are re-loaded from the cache, the bundled samples, or regenerated from the synthetic seed.")

try:
    rows = storage.list_analyses()
except Exception as exc:
    st.error(f"Could not open the history database: {exc}")
    st.stop()

if not rows:
    st.info("No analyses saved yet. Run one on **Analyze Light Curves**.", icon=":material/info:")
    st.page_link(PAGES["analyze"], label="Go to Analyze", icon=":material/arrow_forward:")
    st.stop()

table = pd.DataFrame([{
    "id": r["id"], "when (UTC)": r["created_at"].replace("T", " ")[:16], "target": r["target"], "dataset": r["dataset"],
    "source": r["source"], "BLS period [d]": r["headline"].get("bls_period_days"),
    "BLS SNR": r["headline"].get("bls_snr"), "threshold events": r["headline"].get("threshold_events"),
    "checks passed": r["headline"].get("criteria_passed"),
} for r in rows])
st.dataframe(table, hide_index=True, height=min(400, 38 + 35 * len(table)),
             column_config={"BLS period [d]": st.column_config.NumberColumn(format="%.5f"),
                            "BLS SNR": st.column_config.NumberColumn(format="%.1f")})

ids = [r["id"] for r in rows]
label = {r["id"]: f"#{r['id']} · {r['target']} · {r['dataset']} · {r['created_at'][:16]}" for r in rows}

tab_open, tab_compare, tab_exp = st.tabs(["Open an analysis", "Compare two analyses", "Combinatorics experiments"])

with tab_open:
    aid = st.selectbox("Analysis", ids, format_func=label.get)
    rec = storage.load_analysis(aid)
    if rec:
        c = metric_cols(4)
        h = rec["headline"]
        c[0].metric("BLS period", f"{h['bls_period_days']:.5f} d" if h.get("bls_period_days") else "—")
        c[1].metric("Threshold events", h.get("threshold_events") if h.get("threshold_events") is not None else "—")
        c[2].metric("Checks passed", h.get("criteria_passed", "—"))
        c[3].metric("Source", rec["source"])
        with st.expander("Stored settings"):
            st.json(rec["settings"])
        with st.expander("Evidence checklist"):
            st.dataframe(pd.DataFrame(rec["evaluation"])[["name", "status", "value"]], hide_index=True)
        with st.expander(f"Stored events ({len(rec['events'])})"):
            st.dataframe(pd.DataFrame(rec["events"]), hide_index=True)
        b1, b2 = st.columns(2)
        if b1.button("Reload this dataset and re-run with these settings", type="primary", icon=":material/replay:"):
            try:
                with st.spinner("Reloading data…"):
                    data = reload_dataset(rec["dataset"]["meta"])
                set_dataset(data)
                settings = AnalysisSettings.from_dict(rec["settings"])
                if run_current(settings) is not None:
                    st.switch_page(PAGES["analyze"])
            except (DataError, RetrievalError, ValueError) as exc:
                st.error(f"Could not reload: {exc}")
        if b2.button("Delete this entry", icon=":material/delete:"):
            storage.delete_analysis(aid)
            st.rerun()

with tab_compare:
    if len(ids) < 2:
        st.info("Save at least two analyses to compare them.")
    else:
        c1, c2 = st.columns(2)
        a_id = c1.selectbox("Analysis A", ids, index=1, format_func=label.get)
        b_id = c2.selectbox("Analysis B", ids, index=0, format_func=label.get)
        A, B = storage.load_analysis(a_id), storage.load_analysis(b_id)

        def flat(d, prefix=""):
            out = {}
            for k, v in (d or {}).items():
                if isinstance(v, dict):
                    out.update(flat(v, f"{prefix}{k}."))
                else:
                    out[f"{prefix}{k}"] = v
            return out

        fa, fb = flat(A["settings"]), flat(B["settings"])
        sdiff = [{"setting": k, "A": str(fa.get(k)), "B": str(fb.get(k))} for k in sorted(set(fa) | set(fb))
                 if fa.get(k) != fb.get(k)]
        st.markdown("**Settings that differ**")
        st.dataframe(pd.DataFrame(sdiff) if sdiff else pd.DataFrame([{"setting": "(identical settings)"}]),
                     hide_index=True)
        keys = ["points_analysed", "noise_ppm", "threshold_events", "threshold_period_days", "bls_period_days",
                "bls_depth_ppm", "bls_snr", "bls_transits_observed", "jaccard", "criteria_passed"]
        st.markdown("**Results**")
        st.dataframe(pd.DataFrame([{"result": k, "A": str(A["headline"].get(k)), "B": str(B["headline"].get(k))}
                                   for k in keys]), hide_index=True)
        if A["target"] != B["target"]:
            st.caption("These analyses are of different targets — differences reflect the data, not only the settings.")

with tab_exp:
    exps = storage.list_experiments()
    if not exps:
        st.info("No combinatorics experiments saved yet.")
    for e in exps[:10]:
        with st.expander(f"#{e['id']} · {e['target']} · {e['dataset']} · {e['created_at'][:16]} · "
                         f"{e['stats'].get('configs_run')} configurations"):
            st.json(e["space"])
            st.dataframe(pd.DataFrame(e["results"]), hide_index=True)
