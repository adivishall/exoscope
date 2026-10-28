from types import SimpleNamespace

import pandas as pd
import streamlit as st

from exoscope import plots
from exoscope.dmath.graphs import EdgeRule, adjacency_matrix, build_event_graph, component_period, summarize_graph
from exoscope.ui.common import metric_cols, concept, dataset_header, learned, plot, require_analysis, require_dataset, worked_example

st.title("Graph Theory — the transit-event network")
data = require_dataset()
r = require_analysis()
dataset_header(data)

concept("events as a graph", r"""
We build a graph $G = (V, E)$:

* **Vertices** $V$ — detected events.
* **Edges** $E$ — the documented **periodic-consistency rule**: for a trial period $P$, tolerance $\varepsilon$ and
  maximum multiple $N$,
  $$\{u, v\} \in E \iff \exists\, n \in \{1,\dots,N\}: \big|\,|t_v - t_u| - nP\,\big| \le \varepsilon$$
  optionally also requiring similar depths (deeper/shallower $\le r$).

So an edge says "these two dips *could* be transits of the same object, $n$ orbits apart".

* A **connected component** collects events linked by chains of consistent intervals.
* A **clique** is a set of events that are *pairwise* consistent — stronger evidence than a chain.
* An **isolated vertex** is supported by no other event — often noise.
* The relation "is consistent with" is reflexive and symmetric but **not always transitive**: $u\sim v$ and $v\sim w$
  can hold while $u \not\sim w$ because small timing errors add up. Hence it is *not* an equivalence relation;
  connected components are the classes of its transitive closure.

Graph connectivity is **evidence of periodicity, not proof** that a planet exists.
""")

with worked_example("six events, one of them noise"):
    demo = [SimpleNamespace(event_id=f"E{i + 1}", t_center=t, depth=d, method="demo")
            for i, (t, d) in enumerate([(1.0, 0.010), (4.02, 0.0098), (7.0, 0.0101), (8.6, 0.004), (13.01, 0.0099),
                                        (16.0, 0.0102)])]
    dp = st.slider("Trial period for the example [days]", 1.0, 6.0, 3.0, 0.05, key="demo_p")
    Gd = build_event_graph(demo, EdgeRule(dp, tolerance_hours=1.5, max_multiple=6))
    sd = summarize_graph(Gd)
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(plots.fig_event_graph(Gd, sd.components), key="demo_graph")
    with c2:
        st.markdown(f"|V| = {sd.n_vertices}, |E| = {sd.n_edges}, components = {len(sd.components)}, "
                    f"isolated = {sd.isolated or '∅'}")
        st.markdown(f"Largest clique: {sd.max_clique}")
        st.caption("At P = 3 d, E1, E2, E3, E5, E6 are pairwise consistent (E5 is 4 periods after E1: one transit at "
                   "10 d was missed). E4 at 8.6 d fits nothing and stays isolated. Try P = 1.5 d or 6 d.")

st.subheader("③ Your data")
best = r.best
thr = r.threshold
options = []
if thr and thr.events:
    options.append("Threshold events (A)")
if r.comparison.bls_available and thr:
    options.append("All detected events (A ∪ B)")
if best and best.events:
    options.append("BLS epochs (B)")
if not options:
    st.warning("No events detected — adjust settings on the Analyze page.")
    st.stop()
choice = st.radio("Vertex set", options, horizontal=True,
                  help="BLS epochs are equally spaced by construction, so their graph is trivially connected; "
                       "threshold events are the interesting case.")
if choice.startswith("Threshold"):
    events = thr.events
elif choice.startswith("BLS"):
    events = best.events
else:
    by_id = {e.event_id: e for e in thr.events + best.events}
    comp = r.comparison
    events = []
    for k in comp.sorted_ids(comp.union()):
        el = comp.elements[k]
        src = by_id[el.threshold_id or el.bls_id]
        events.append(SimpleNamespace(event_id=el.label, t_center=el.t_center, depth=src.depth, t_start=src.t_start,
                                      t_end=src.t_end, method="both" if el.threshold_id and el.bls_id else src.method))

p_default = best.period if best else (thr.period.period if thr.period and thr.period.members else 1.0)
c = st.columns(4)
P = c[0].number_input("Period P [d]", 0.05, 500.0, float(p_default), float(p_default / 1000), format="%.5f")
tol_h = c[1].slider("Tolerance ε [h]", 0.25, 6.0, float(r.settings.match_tolerance_hours), 0.25)
nmax = c[2].slider("Max multiple N", 1, 200, 20 if len(events) > 40 else 10)
use_depth = c[3].toggle("Require similar depth", False)
ratio = c[3].slider("Max depth ratio r", 1.1, 5.0, 2.0, 0.1, disabled=not use_depth)
rule = EdgeRule(P, tol_h, nmax, ratio if use_depth else None)
try:
    rule.validate()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
st.caption(f"Edge rule: {rule.describe()}.")

G = build_event_graph(events, rule)
gs = summarize_graph(G)
m = metric_cols(6)
m[0].metric("|V|", gs.n_vertices)
m[1].metric("|E|", gs.n_edges)
m[2].metric("Density", f"{gs.density:.2f}", help="|E| / C(|V|, 2)")
m[3].metric("Components", len(gs.components))
m[4].metric("Isolated vertices", len(gs.isolated))
m[5].metric("Largest clique", len(gs.max_clique))

layout = st.segmented_control("Layout", ["time", "spring", "circular"], default="time",
                              format_func={"time": "time × depth", "spring": "force-directed",
                                           "circular": "circle (time order)"}.get) or "time"
selected = st.session_state.get("graph_selected")
ev = plot(plots.fig_event_graph(G, gs.components, layout, selected), key="event_graph", on_select="rerun",
          selection_mode="points")
try:
    pts = ev.selection.points if ev else []
except AttributeError:
    pts = []
clicked = None
for pnt in pts:
    cd = pnt.get("customdata")
    if cd and len(cd) >= 4 and cd[3] in G:
        clicked = cd[3]
        break
    if layout == "time" and pnt.get("x") is not None:  # clicked near a vertex but hit an edge marker
        clicked = min(G.nodes, key=lambda k: abs(G.nodes[k]["t"] - float(pnt["x"])))
        break
# Plotly keeps the selection across reruns: act only on a *new* click so the selectbox stays usable.
if clicked is not None and clicked != st.session_state.get("graph_last_click"):
    st.session_state.graph_last_click = clicked
    st.session_state.graph_selected = selected = clicked
    st.session_state.pop("graph_vertex_pick", None)  # let the selectbox follow the click

st.markdown("**Linked view** — click a vertex above or choose one here to see its region of the light curve.")
order = sorted(G.nodes, key=lambda k: G.nodes[k]["t"])
idx = order.index(selected) if selected in order else 0
pick = st.selectbox("Vertex", order, index=idx, key="graph_vertex_pick",
                    format_func=lambda k: f"{k} (t = {G.nodes[k]['t']:.4f}, degree {G.degree(k)})")
st.session_state.graph_selected = pick
tc = G.nodes[pick]["t"]
half = max(3 * (best.duration if best else 0.15), 0.3)
ids = [pick] if not choice.startswith("All") else [x for x in pick.split("≈")]
plot(plots.fig_detections(r.processed, thr, best, data.time_format, xrange=(tc - half, tc + half), highlight=ids),
     key="graph_lc")
nbrs = sorted(G.neighbors(pick), key=lambda k: G.nodes[k]["t"])
if nbrs:
    st.dataframe(pd.DataFrame([{"neighbour": n, "Δt [d]": G.edges[pick, n]["dt"], "n": G.edges[pick, n]["n"],
                                "Δt − nP [h]": G.edges[pick, n]["residual_hours"]} for n in nbrs]), hide_index=True,
                 height=min(300, 38 + 35 * len(nbrs)))
else:
    st.caption(f"{pick} is isolated: no other event lies a whole number of periods away (within ±{tol_h:g} h).")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Connected components** (largest first)")
    rows = []
    for i, cset in enumerate(gs.components[:15], 1):
        fit = component_period(G, cset, P)
        rows.append({"#": i, "size": len(cset), "vertices": ", ".join(cset[:8]) + ("…" if len(cset) > 8 else ""),
                     "refit P [d]": round(fit["period"], 5) if fit else None,
                     "timing rms [h]": round(fit["rms_hours"], 2) if fit else None})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Refit P uses only the events of that component (integer epochs + least squares).")
with c2:
    st.markdown("**Is 'consistent with' transitive?**")
    if gs.is_transitive:
        st.success("On this vertex set every chain u–v–w also has the edge u–w (each component is a clique).")
    else:
        u, v_, w = gs.transitivity_counterexample
        st.warning(f"No: {u}–{v_} and {v_}–{w} are edges, but {u}–{w} is not. Residuals "
                   f"{G.edges[u, v_]['residual_hours']:+.2f} h and {G.edges[v_, w]['residual_hours']:+.2f} h add up "
                   "beyond the tolerance (or the multiple exceeds N). The relation is reflexive and symmetric but "
                   "not transitive → not an equivalence relation.")
    st.markdown(f"**Maximum clique** ({len(gs.max_clique)} events): {', '.join(gs.max_clique[:20])}"
                + ("…" if len(gs.max_clique) > 20 else ""))

if gs.n_vertices <= 60:
    nodes, M = adjacency_matrix(G)
    with st.expander("Adjacency matrix"):
        plot(plots.fig_adjacency(nodes, M), key="adj")
        st.caption("Symmetric (undirected graph) with zeros on the diagonal (no self-loops). Row sums are degrees.")

largest = len(gs.components[0]) if gs.components else 0
learned([
    f"{largest} of {gs.n_vertices} events form the largest connected component at P = {P:.5f} d; "
    f"{len(gs.isolated)} are isolated.",
    "A component means 'linked by a chain'; a clique means 'every pair agrees' — the clique is the stronger evidence.",
    "Tolerance relations are generally not transitive, so components (transitive closure) can join events that do "
    "not directly agree.",
    "Tightening ε or demanding similar depths removes edges and splits components: graph structure depends on the "
    "documented edge rule.",
])
