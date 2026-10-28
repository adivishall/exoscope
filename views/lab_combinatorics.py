import itertools
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from exoscope import plots, storage
from exoscope.dmath.combinatorics import n_choose_k, select_configs
from exoscope.experiments import DEFAULT_SELECTION, MAX_RUNS, PARAMETER_CATALOG, build_space, run_configurations
from exoscope.ui.common import (metric_cols, concept, dataset_header, learned, plot, references, require_analysis, require_dataset,
                                worked_example)

st.title("Combinatorics — the parameter combination explorer")
data = require_dataset()
r = require_analysis()
dataset_header(data)

concept("counting configurations", r"""
Each analysis setting is a finite **set of choices**, e.g. $S_1$ = thresholds $\{2.5, 3, 3.5, 4\}$,
$S_2$ = minimum points $\{1, 2, 3\}$. A complete configuration is one element of the **Cartesian product**
$S_1 \times S_2 \times \dots \times S_k$, and by the **rule of product**

$$|S_1 \times \dots \times S_k| = |S_1|\cdot|S_2|\cdots|S_k|.$$

* **Constraints** remove configurations that make no physical sense (counting with restrictions), e.g. a detrending
  window shorter than 3× the transit duration would erase the transit.
* Running only $m$ of $n$ valid configurations can be done in $\binom{n}{m}$ ways; ExoScope takes the first $m$ or a
  seeded random sample.
* Preprocessing depends only on preprocessing parameters, so the number of distinct preprocessing runs is the size of
  the **projection** of the product onto those parameters — usually much smaller.
* **Possible ≠ meaningful.** Many configurations give the *same* result (the map configuration → result is
  many-to-one), and a large count does not mean many independent discoveries.
""")

with worked_example("a small product you can list by hand"):
    S1, S2, S3 = [2.5, 3.0], [1, 2], ["0.5 d", "1 d"]
    prod = list(itertools.product(S1, S2, S3))
    st.markdown(f"S₁ = {{2.5, 3}} (threshold), S₂ = {{1, 2}} (min points), S₃ = {{0.5 d, 1 d}} (window) → "
                f"|S₁ × S₂ × S₃| = 2 · 2 · 2 = **{len(prod)}**")
    st.dataframe(pd.DataFrame(prod, columns=["k [σ]", "min points", "window"]), hide_index=True)
    st.caption(f"Choosing 3 of these 8 to run: C(8, 3) = {n_choose_k(8, 3)} ways. Projection onto 'window' has "
               "size 2 → only 2 distinct detrending runs are needed.")

st.subheader("③ Your data")
base = r.settings
st.markdown("Choose the values of each parameter (each multiselect is one set Sᵢ). Unselected parameters keep the "
            "value from your current analysis.")
selection = {}
cols = st.columns(2)
for i, (key, p) in enumerate(PARAMETER_CATALOG.items()):
    default = DEFAULT_SELECTION.get(key, [])
    selection[key] = cols[i % 2].multiselect(f"{p.label}", p.values, default=default, format_func=p.fmt, key=f"cmb_{key}")

try:
    space = build_space({k: v for k, v in selection.items() if v}, base)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

configs_all = list(space.enumerate())
valid = [c for c in configs_all if space.is_valid(c)]
pre_proj = space.projection_size("preprocess", valid) if valid else 0
m = metric_cols(4)
m[0].metric("|S₁ × … × S_k|", f"{space.size():,}", help=space.formula())
m[1].metric("Valid (constraints)", f"{len(valid):,}")
m[2].metric("Distinct preprocessing runs", pre_proj, help="Size of the projection onto preprocessing parameters")
m[3].metric("Distinct BLS runs", space.projection_size(["preprocess", "bls"], valid) if valid else 0,
            help="BLS depends on preprocessing and BLS parameters only")
st.markdown(f"Rule of product: **{space.formula()}** configurations; {space.size() - len(valid)} violate a constraint.")
with st.expander("Constraints and excluded configurations"):
    for c in space.constraints:
        st.markdown(f"- {c.description}")
    bad = [c for c in configs_all if not space.is_valid(c)][:20]
    if bad:
        st.dataframe(pd.DataFrame([{"configuration": space.label(c), "violates": "; ".join(space.violated(c))}
                                   for c in bad]), hide_index=True)
    else:
        st.caption("No configuration is excluded.")

if not valid:
    st.stop()
c1, c2, c3 = st.columns(3)
limit = c1.slider("Configurations to run", 1, min(MAX_RUNS, len(valid)), min(24, len(valid)))
strategy = c2.radio("Selection", ["first", "random"], horizontal=True,
                    format_func={"first": "first m (lexicographic)", "random": "seeded random sample"}.get)
seed = c3.number_input("Seed", 0, 9999, 0, disabled=strategy == "first")
st.caption(f"Ways to choose {limit} of {len(valid)} valid configurations: C({len(valid)}, {limit}) = "
           f"{n_choose_k(len(valid), limit):,}. Runs are capped at {MAX_RUNS} to keep the app responsive.")
chosen = select_configs(valid, limit, strategy, int(seed))

with st.expander(f"Configuration table ({len(chosen)} selected)"):
    st.dataframe(pd.DataFrame([{p.label: p.fmt(c[p.key]) for p in space.parameters} for c in chosen]), hide_index=True)

if st.button(f"Run {len(chosen)} configuration(s)", type="primary", icon=":material/play_arrow:"):
    bar = st.progress(0.0, text="Starting…")

    def progress(i, n, label):
        bar.progress(i / n if n else 1.0, text=f"{i}/{n} · {label}")

    refs, _ = references()
    df, stats = run_configurations(data, base, space, chosen, references=refs, progress=progress)
    info = {"formula": space.formula(), "size": space.size(), "n_valid": len(valid), "n_run": len(chosen),
            "preprocessing_runs": stats["preprocessing_runs"], "bls_runs": stats["bls_runs"]}
    st.session_state.experiment = {"results": df.to_dict("records"), "stats": stats, "report_info": info,
                                   "params": [p.label for p in space.parameters]}
    try:
        storage.save_experiment(data.target, str(data.meta.get("dataset", "")),
                                {p.key: [p.fmt(v) for v in p.values] for p in space.parameters},
                                df.to_dict("records"), {**stats, **info},
                                datetime.now(timezone.utc).isoformat(timespec="seconds"))
    except Exception as exc:
        st.toast(f"Experiment not saved: {exc}", icon="⚠️")

exp = st.session_state.get("experiment")
if exp:
    df = pd.DataFrame(exp["results"])
    stats = exp["stats"]
    st.success(f"Ran {stats['configs_run']} configurations with only {stats['preprocessing_runs']} preprocessing "
               f"run(s) and {stats['bls_runs']} BLS run(s) — memoisation by projection.", icon=":material/check_circle:")
    st.dataframe(df, hide_index=True, height=min(420, 38 + 35 * len(df)))
    ok = df[df["status"] == "ok"]
    params = [p for p in exp["params"] if p in ok.columns]
    metric_opts = [c for c in ["threshold_events", "|A∩B|", "jaccard", "bls_snr", "thr_period_error_%",
                               "bls_period_error_%", "runtime_s"] if c in ok.columns]
    if len(ok) and params and metric_opts:
        c1, c2, c3 = st.columns(3)
        xp = c1.selectbox("Columns", params, index=0)
        yp = c2.selectbox("Rows", params, index=min(1, len(params) - 1))
        metric = c3.selectbox("Value", metric_opts)
        if xp != yp:
            pivot = ok.pivot_table(index=yp, columns=xp, values=metric, aggfunc="mean")
            plot(plots.fig_heatmap(pivot, f"Mean {metric} by {yp} × {xp}", xp, yp, metric), key="cmb_heat")
        else:
            st.bar_chart(ok.groupby(xp)[metric].mean(), color="#3987e5", height=300)
        n_out = ok["threshold_events"].nunique()
        st.markdown(
            f"**Possible vs meaningful:** {len(ok)} configurations produced only **{n_out}** distinct threshold "
            "event counts" + (f" and {ok['bls_period_d'].round(4).nunique()} distinct BLS periods" if "bls_period_d" in ok else "")
            + ". The map configuration → result is many-to-one: more configurations do not mean more independent "
            "findings, and settings that agree on the period make the detection more robust.")

learned([
    f"The chosen sets give {space.formula()} configurations; constraints leave {len(valid)} valid.",
    f"Only {pre_proj} distinct preprocessing run(s) are needed — the size of a projection, not of the product.",
    "Counting first (before running) tells you whether an experiment is feasible.",
    "Results are many-to-one in the configurations: robustness means the answer stays the same across settings.",
])
