import numpy as np
import pandas as pd
import streamlit as st

from exoscope import plots
from exoscope.dmath.sets import compare_detections, detection_metrics, roster
from exoscope.ui.common import metric_cols, concept, dataset_header, learned, plot, require_analysis, require_dataset, worked_example

st.title("Set Theory — comparing detections")
data = require_dataset()
r = require_analysis()
dataset_header(data)

concept("sets of detections", r"""
A **set** is a collection of distinct elements. We describe each detector's output as a set:

* $A$ = events found by the **threshold** detector,
* $B$ = transit **epochs predicted by BLS** that contain data,
* $G$ = BLS-predicted epochs that fall in **data gaps**, and the universe $U = A \cup B \cup G$.

**Comparing like with like.** The threshold method returns *individual dips*; BLS returns *one periodic signal*.
We expand the BLS signal into its individual predicted transits so both sets contain the same kind of object
(events). Two events are the **same element** if their mid-times differ by at most a tolerance $\tau$ — matched
one-to-one by an optimal assignment.

| Operation | Notation | Meaning here |
|---|---|---|
| Union | $A \cup B$ | found by at least one method |
| Intersection | $A \cap B$ | found by both — agreement |
| Difference | $A - B$ | threshold only (noise? second signal?) |
| Difference | $B - A$ | BLS only (usually shallow transits) |
| Complement | $(A\cup B)^c = U - (A \cup B)$ | expected but unobservable (gaps) |
| Symmetric difference | $A \triangle B$ | disagreements |
""")

with worked_example("build the sets yourself"):
    U = [f"E{i}" for i in range(1, 9)]
    c1, c2 = st.columns(2)
    A = set(c1.multiselect("Elements of A (threshold)", U, ["E1", "E2", "E3", "E5"]))
    B = set(c2.multiselect("Elements of B (BLS)", U, ["E2", "E3", "E4", "E5", "E6"]))
    ops = {
        "A ∪ B": A | B, "A ∩ B": A & B, "A − B": A - B, "B − A": B - A,
        "Aᶜ = U − A": set(U) - A, "(A ∪ B)ᶜ": set(U) - (A | B), "A △ B": A ^ B,
    }
    st.dataframe(pd.DataFrame([{"operation": k, "result": roster(sorted(v, key=lambda x: int(x[1:]))),
                                "size": len(v)} for k, v in ops.items()]), hide_index=True)
    st.markdown(f"Inclusion–exclusion: |A ∪ B| = |A| + |B| − |A ∩ B| → {len(A | B)} = {len(A)} + {len(B)} − "
                f"{len(A & B)} ✔︎  (U = {roster(U)})")

st.subheader("③ Your data")
if not r.comparison.bls_available or r.threshold is None:
    st.warning("Run **both** detectors on the Analyze page to compare their sets.")
    st.stop()

tol = st.slider("Matching tolerance τ [hours]", 0.25, 6.0, float(r.settings.match_tolerance_hours), 0.25,
                help="Recomputes the matching live; the stored analysis is unchanged.")
best = r.best
comp = compare_detections(r.threshold.events, best.events, best.unobserved_times, tol)
s = comp.summary()

m = metric_cols(6)
m[0].metric("|U|", s["|U|"])
m[1].metric("|A| threshold", s["|A|"])
m[2].metric("|B| BLS", s["|B|"])
m[3].metric("|A ∩ B|", s["|A∩B|"], f"{s['shared_%']:.0f}% of A ∪ B", delta_color="off")
m[4].metric("Jaccard", f"{s['jaccard']:.2f}")
ie = comp.inclusion_exclusion()
m[5].metric("Inclusion–exclusion", "holds ✔︎" if ie["holds"] else "fails ✘",
            f"{ie['union']} = {ie['A']} + {ie['B']} − {ie['intersection']}", delta_color="off")

c1, c2 = st.columns([1, 1])
with c1:
    plot(plots.fig_venn(comp), key="venn")
with c2:
    op = st.radio("Highlight a set on the light curve",
                  ["A ∩ B", "A − B", "B − A", "A ∪ B", "A △ B", "Aᶜ (relative to U)", "(A ∪ B)ᶜ"], horizontal=True)
    sets = {"A ∩ B": comp.intersection(), "A − B": comp.a_minus_b(), "B − A": comp.b_minus_a(),
            "A ∪ B": comp.union(), "A △ B": comp.symmetric_difference(),
            "Aᶜ (relative to U)": comp.complement(comp.A), "(A ∪ B)ᶜ": comp.complement(comp.union())}
    chosen = sets[op]
    st.markdown(f"**{op}** has **{len(chosen)}** element(s):")
    st.code(roster(chosen, comp, 20), language=None, wrap_lines=True)
    meaning = {
        "A ∩ B": "Transits both methods agree on.",
        "A − B": "Threshold dips with no BLS transit within τ: noise, systematics, or a different signal.",
        "B − A": "Epochs BLS predicts (with data) that the threshold missed — usually below the per-point threshold.",
        "A ∪ B": "Everything found by at least one method.",
        "A △ B": "All disagreements: (A − B) ∪ (B − A).",
        "Aᶜ (relative to U)": "Evaluated epochs the threshold detector did not find (includes gaps).",
        "(A ∪ B)ᶜ": "Predicted epochs that fall in data gaps — no method could see them.",
    }
    st.caption(meaning[op])

ids = set()
for k in chosen:
    e = comp.elements[k]
    ids.update(x for x in (e.threshold_id, e.bls_id) if x)
plot(plots.fig_detections(r.processed, r.threshold, best, data.time_format, highlight=sorted(ids)), key="sets_lc")
plot(plots.fig_set_timeline(comp, data.time_format), key="sets_timeline")

with st.expander("All elements of U with their region"):
    st.dataframe(pd.DataFrame(comp.table()), hide_index=True)

st.markdown("**How the tolerance τ changes the sets**")
taus = np.arange(0.25, 6.01, 0.25)
sweep = []
for t in taus:
    cs = compare_detections(r.threshold.events, best.events, best.unobserved_times, float(t)).summary()
    sweep.append({"τ [h]": t, "|A ∩ B|": cs["|A∩B|"], "|A − B|": cs["|A−B|"], "|B − A|": cs["|B−A|"]})
st.line_chart(pd.DataFrame(sweep).set_index("τ [h]"), color=["#199e70", "#3987e5", "#d95926"], height=260)
st.caption("Too small a τ splits genuinely identical transits; too large a τ matches unrelated events.")

if data.truth:
    st.markdown("**Precision and recall are set operations** (synthetic data, so the true set T is known)")
    T = [e["t_center"] for e in data.truth["epochs"] if e["observed"]]
    rows = []
    for name, times in (("Threshold (A)", [e.t_center for e in r.threshold.events]),
                        ("BLS (B)", [e.t_center for e in best.events])):
        mtr = detection_metrics(times, T, tol / 24)
        rows.append({"detections D": name, "TP = |D ∩ T|": mtr["tp"], "FP = |D − T|": mtr["fp"],
                     "FN = |T − D|": mtr["fn"], "precision": mtr["precision"], "recall": mtr["recall"], "F1": mtr["f1"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

learned([
    f"With τ = {tol:g} h the detectors share {s['|A∩B|']} of {s['|A∪B|']} events (Jaccard {s['jaccard']:.2f}).",
    f"B − A has {s['|B−A|']} element(s): transits that BLS recovers by combining many epochs but that a per-point "
    "threshold cannot see." if s["|B−A|"] else "B − A is empty: every BLS epoch was also caught by the threshold.",
    f"A − B has {s['|A−B|']} element(s): dips that do not fit the periodic signal and deserve scrutiny.",
    "Choosing the universe matters: the complement only has meaning relative to a clearly defined U.",
    "Set membership depends on a matching rule — a modelling choice that must be stated, like τ here.",
])
