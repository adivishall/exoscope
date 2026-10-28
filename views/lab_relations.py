import numpy as np
import pandas as pd
import streamlit as st

from exoscope import plots
from exoscope.dmath.relations import (Relation, config_count_function, consistency_relation, event_algorithm_relation,
                                      event_epoch_function, example_relations, interval_function,
                                      observation_flux_relation, precedes_relation)
from exoscope.ui.common import concept, dataset_header, learned, plot, require_analysis, require_dataset, worked_example

st.title("Relations & Functions — the structure of the pipeline")
data = require_dataset()
r = require_analysis()
dataset_header(data)

concept("relations, functions and their types", r"""
A **relation** $R$ from a set $X$ (the *domain*) to a set $Y$ (the *codomain*) is any subset of the Cartesian product:
$R \subseteq X \times Y$, written as **ordered pairs** $(x, y)$.

* $R$ is a **function** $f: X \to Y$ when **every** $x \in X$ is paired with **exactly one** $y$.
* A function is **injective** (one-to-one) if no two inputs share an output, **surjective** (onto) if every
  $y \in Y$ is hit, and **bijective** if both.
* By degree: **one-to-one**, **one-to-many** (some $x$ has several $y$ — not a function), **many-to-one** (a function
  that is not injective) and **many-to-many**.
* A relation on a single set ($X = Y$) can be **reflexive**, **symmetric**, **antisymmetric**, **transitive**;
  an *equivalence* has the first, second and fourth; a *partial order* has the first, third and fourth.

**In ExoScope:** observation time → flux is a function; event ↔ algorithm is many-to-many (an event can be found by
both detectors) — so it must **not** be called a function.
""")


def show_relation(rel: Relation, key: str) -> None:
    props = rel.properties()
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(plots.fig_relation(rel), key=f"map_{key}")
    with c2:
        st.markdown(f"**Domain X** ({rel.domain_label}): {len(rel.domain)} element(s)  \n"
                    f"**Codomain Y** ({rel.codomain_label})" + (f": {len(rel.codomain)} element(s)" if not rel.infinite_codomain else ""))
        if rel.codomain_note:
            st.caption(rel.codomain_note)
        badge = {"one-to-one": "blue", "one-to-many": "orange", "many-to-one": "green", "many-to-many": "violet"}
        st.badge(props["type"], color=badge[props["type"]])
        checks = [("Function (each x has exactly one y)", props["function"]),
                  ("Injective", props["injective"]), ("Surjective onto Y", props["surjective"]),
                  ("Bijective", props["bijective"])]
        if rel.is_homogeneous:
            checks += [("Reflexive", props["reflexive"]), ("Symmetric", props["symmetric"]),
                       ("Antisymmetric", props["antisymmetric"]), ("Transitive", props["transitive"]),
                       ("Equivalence relation", props["equivalence"]), ("Strict order", props["strict_order"])]
        st.dataframe(pd.DataFrame([{"property": n, "holds": "✅ yes" if v else "❌ no"} for n, v in checks]),
                     hide_index=True)
        viol = rel.function_violations()
        if viol:
            st.markdown("**Why it is not a function:**")
            st.markdown("\n".join(f"- {v}" for v in viol[:6]) + ("\n- …" if len(viol) > 6 else ""))
        if rel.is_homogeneous and not props["transitive"]:
            ce = rel.transitive_counterexample()
            if ce:
                st.caption(f"Not transitive: ({ce[0]}, {ce[1]}) and ({ce[1]}, {ce[2]}) ∈ R but ({ce[0]}, {ce[2]}) ∉ R.")
        for note in rel.notes:
            st.caption(note)
    with st.expander("Ordered pairs and relation matrix"):
        st.code(rel.ordered_pairs(40), language=None, wrap_lines=True)
        if not rel.infinite_codomain or rel.is_homogeneous:
            plot(plots.fig_relation_matrix(rel), key=f"mat_{key}")


with worked_example("the four relationship types"):
    ex = example_relations()
    kind = st.segmented_control("Example", list(ex), default="many-to-many") or "many-to-many"
    show_relation(ex[kind], f"ex_{kind}")

st.subheader("③ Your data")
best, thr, comp = r.best, r.threshold, r.comparison
options = {"f: Observation → Flux": "obs"}
if comp.union():
    options["R: Event ↔ Algorithm"] = "alg"
if best and best.events:
    options["g: Event → Epoch number"] = "epoch"
if thr and len(thr.events) >= 2:
    options["h: Event pair → Δt"] = "interval"
    options["~: periodic consistency (on events)"] = "consistency"
    options["≺: occurs before (on events)"] = "precedes"
exp = st.session_state.get("experiment")
if exp is not None:
    options["k: Configuration → #detections"] = "config"
label = st.selectbox("Relation", list(options))
key = options[label]

if key == "obs":
    start = st.slider("First observation shown", 0, max(0, r.processed.n_points - 8), 0)
    rel = observation_flux_relation(r.processed.time, r.processed.flux, n=8, start=start)
    st.caption("Duplicate timestamps are removed during preprocessing precisely so that this relation is a function.")
elif key == "alg":
    rel = event_algorithm_relation(comp)
    st.caption(f"Domain: the {len(comp.union())} elements of A ∪ B (matched within ±{comp.tolerance_hours:g} h).")
elif key == "epoch":
    who = st.radio("Events", ["Threshold events", "BLS epochs"] if thr and thr.events else ["BLS epochs"], horizontal=True)
    rel = event_epoch_function(thr.events if who.startswith("Threshold") else best.events, best.period, best.t0)
    st.caption(f"g(e) = round((t_e − t₀)/P) with P = {best.period:.5f} d. If two events get the same epoch, g is not "
               "injective: one transit was split into two detections (or a false event sits nearby).")
elif key == "interval":
    rel = interval_function(thr.events, limit=10)
elif key == "consistency":
    P = best.period if best else (thr.period.period if thr.period else float(np.median(np.diff([e.t_center for e in thr.events]))))
    tol = st.slider("Tolerance ε [h]", 0.25, 6.0, float(r.settings.match_tolerance_hours), 0.25, key="rel_tol")
    rel = consistency_relation(thr.events[:14], P, tol)
    st.caption(f"First {min(14, len(thr.events))} threshold events; u ~ v ⇔ u = v or |Δt − nP| ≤ ε (n ≤ 10) at P = {P:.5f} d.")
elif key == "precedes":
    rel = precedes_relation(thr.events[:12])
else:
    df = pd.DataFrame(exp["results"])
    ok = df[df["status"] == "ok"]
    rel = config_count_function(ok["config"].tolist(), ok["threshold_events"].astype(int).tolist())
    st.caption("From your latest Combinatorics experiment: each configuration determines one count (a function); "
               "many configurations share a count (many-to-one).")
show_relation(rel, key)

st.subheader("Build your own relation")
txt = st.text_area("Ordered pairs, one per line as `x, y`", "E1, Threshold\nE1, BLS\nE2, BLS\nE3, Threshold", height=120)
try:
    pairs = set()
    for line in txt.strip().splitlines():
        if line.strip():
            x, y = [p.strip() for p in line.split(",", 1)]
            if not x or not y:
                raise ValueError
            pairs.add((x, y))
    dom = sorted({x for x, _ in pairs})
    cod = sorted({y for _, y in pairs})
    extra_dom = st.text_input("Extra domain elements with no image (optional, comma-separated)", "")
    dom += [e.strip() for e in extra_dom.split(",") if e.strip() and e.strip() not in dom]
    show_relation(Relation("Your relation", dom, cod, pairs, "X", "Y"), "custom")
except ValueError:
    st.error("Each line must look like `x, y`.")

learned([
    "Observation → flux is a function: every timestamp has exactly one measured flux.",
    "Event ↔ algorithm is many-to-many in general, so it is a relation but not a function.",
    "Event → epoch is a function into ℤ; failure of injectivity flags split or spurious detections.",
    "'Occurs before' is a strict order, while 'periodically consistent' is reflexive and symmetric but usually not "
    "transitive — properties of relations describe real behaviour of the data.",
])
