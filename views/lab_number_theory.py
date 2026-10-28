import numpy as np
import pandas as pd
import streamlit as st

from exoscope import plots
from exoscope.dmath.number_theory import (alias_relation, estimate_period, euclid_steps, gcd_list, interval_table,
                                          modular_residues)
from exoscope.ui.common import (metric_cols, concept, dataset_header, learned, plot, references, require_analysis,
                                require_dataset, worked_example)



@st.cache_data(show_spinner=False, max_entries=20)
def _estimate(times: tuple[float, ...], tol: float):
    return estimate_period(np.array(times), tol)


st.title("Number Theory — periodicity and integer relationships")
data = require_dataset()
r = require_analysis()
dataset_header(data)

concept("multiples, remainders and congruences", r"""
If one body orbits with period $P$, its transits happen at $t_n = t_0 + n\,P$ with **integer** epoch $n$. Therefore:

* **Divisibility.** The gap between any two transits is an integer multiple of $P$. Measurements are noisy, so we use
  *approximate divisibility*:  $P \mid_\varepsilon \Delta \iff \exists\,n \in \mathbb{Z}^+ : |\Delta - nP| \le \varepsilon$.
* **Missed transits.** A gap with $n = 3$ means two transits in between were not detected (data gap or below threshold).
  Consecutive events are *not* always one period apart.
* **Congruence.** All transit times leave the same remainder modulo $P$: $t_i \equiv t_0 \pmod P$. Phase-folding is
  exactly this reduction, and a wrong $P$ scatters the remainders.
* **GCD.** The period is the (approximate) **greatest common divisor** of the intervals. Euclid's algorithm finds exact
  GCDs of integers, but noise breaks it — which is why ExoScope searches for the largest set of events that are
  congruent modulo a candidate $P$ (an *approximate GCD*).
""")

with worked_example("find a period from five event times"):
    txt = st.text_input("Event times in hours (comma-separated)", "10, 34, 58, 106, 130")
    noisy = st.toggle("Add measurement noise (±0.6 h)", False)
    try:
        hrs = np.array(sorted(float(x) for x in txt.split(",") if x.strip()))
        if len(hrs) < 2:
            raise ValueError
        if noisy:
            hrs = hrs + np.random.default_rng(1).uniform(-0.6, 0.6, len(hrs))
        gaps = np.diff(hrs)
        st.markdown("Intervals Δ: " + ", ".join(f"{g:.2f}" for g in gaps) + " h")
        ints = [int(round(g)) for g in gaps]
        steps, _ = euclid_steps(ints[0], ints[1]) if len(ints) >= 2 else ([], ints[0])
        g_all = gcd_list(ints)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Exact GCD of the rounded intervals {ints}** = **{g_all}**")
            if steps:
                st.caption(f"Euclid's algorithm on {ints[0]} and {ints[1]}:")
                st.dataframe(pd.DataFrame([{"a": s['a'], "b": s['b'], "a = q·b + r": f"{s['a']} = {s['q']}·{s['b']} + {s['r']}"}
                                           for s in steps]), hide_index=True)
        with c2:
            est = estimate_period(hrs, tol=1.0, min_period=4.0)
            if est and est.members:
                st.markdown(f"**Approximate GCD (tolerance 1 h)** = **{est.period:.2f} h**")
                st.caption(f"{est.note} Epochs n = {est.epochs}: gaps of n > 1 reveal missed events.")
            elif est:
                st.markdown(est.note)
        if noisy and g_all < 4:
            st.warning(f"With noise the exact GCD collapses to {g_all} — meaningless. The tolerant version still works.")
    except ValueError:
        st.error("Enter at least two numbers separated by commas.")

st.subheader("③ Your data")
best = r.best
thr = r.threshold
sources = {}
if thr and len(thr.events) >= 2:
    sources["Threshold events (A)"] = thr.events
if best and len(best.events) >= 2:
    sources["BLS epochs (B)"] = best.events
if not sources:
    st.warning("Fewer than two events were detected — adjust the detection settings on the Analyze page.")
    st.stop()
src = st.radio("Events to study", list(sources), horizontal=True)
events = sorted(sources[src], key=lambda e: e.t_center)
times = np.array([e.t_center for e in events])
labels = [e.event_id for e in events]

p_default = best.period if best else (thr.period.period if thr.period and thr.period.members else float(np.median(np.diff(times))))
c1, c2 = st.columns([3, 1])
P = c1.slider("Trial period P [days]", float(p_default * 0.4), float(p_default * 2.5), float(p_default),
              float(p_default / 4000), format="%.5f",
              help="Drag away from the detected period and watch the remainders scatter.")
tol_h = c2.slider("Tolerance ε [hours]", 0.25, 6.0, float(r.settings.match_tolerance_hours), 0.25)
tol = tol_h / 24
t0 = best.t0 if (best and src.startswith("BLS")) else times[0]

rows = interval_table(times, P, tol)
consistent = sum(x["consistent"] for x in rows)
m = metric_cols(4)
m[0].metric("Events", len(times))
m[1].metric("Intervals ≈ n·P", f"{consistent}/{len(rows)}")
m[2].metric("Missed transits implied", sum(x["missed_between"] or 0 for x in rows))
cong = np.sum(np.abs(((times - t0 + P / 2) % P) - P / 2) <= tol)
m[3].metric("Events ≡ t₀ (mod P)", f"{cong}/{len(times)}")

t1, t2, t3, t4 = st.tabs(["Intervals as multiples", "Modular clock", "Approximate GCD & O−C", "Integer arithmetic"])
with t1:
    plot(plots.fig_intervals(times, P), key="nt_hist")
    df = pd.DataFrame(rows)
    st.dataframe(df[["pair", "delta_days", "ratio", "n", "residual_hours", "consistent", "missed_between"]],
                 hide_index=True, height=min(380, 38 + 35 * len(df)),
                 column_config={"delta_days": st.column_config.NumberColumn("Δ [d]", format="%.4f"),
                                "ratio": st.column_config.NumberColumn("Δ / P", format="%.3f"),
                                "n": st.column_config.NumberColumn("n = round(Δ/P)"),
                                "residual_hours": st.column_config.NumberColumn("Δ − nP [h]", format="%+.2f"),
                                "consistent": st.column_config.CheckboxColumn("|Δ − nP| ≤ ε"),
                                "missed_between": st.column_config.NumberColumn("missed (n − 1)")})
with t2:
    plot(plots.fig_mod_clock(times, labels, P, t0, tol), key="nt_clock")
    st.caption("Each event is placed at angle 360°·((t − t₀) mod P)/P. Congruent events stack at the top (0°). "
               "At the right period they cluster; at a wrong period they spread around the circle.")
with t3:
    est = _estimate(tuple(times), tol)
    if est is None or not est.members:
        st.info(est.note if est else "Not enough events for an estimate.")
    else:
        c = metric_cols(4)
        c[0].metric("Approximate-GCD period", f"{est.period:.5f} d")
        c[1].metric("Uncertainty", f"± {est.period_err:.5f} d")
        c[2].metric("Events in the congruence class", f"{est.n_members}/{est.n_events}")
        c[3].metric("Candidates tested", f"{est.candidates_tested:,}")
        st.caption(f"{est.note} Candidates were every pairwise Δ divided by k = 1…10; the score is members × fill "
                   "fraction, so a sub-multiple P/k (which also 'divides' everything) loses to P. The integer epochs n "
                   "of the members are then fitted by least squares: t = t₀ + n·P.")
        plot(plots.fig_oc(est.epochs, est.residuals_days, [labels[i] for i in est.members]), key="nt_oc")
        comps = []
        if best:
            rel = alias_relation(best.period, est.period)
            comps.append({"compare with": "BLS period", "value [d]": best.period, "ratio": rel["ratio"],
                          "≈ p/q": str(rel["fraction"]), "relation": rel["kind"]})
        refs, _ = references()
        for p in refs:
            rel = alias_relation(p.period_days, est.period)
            comps.append({"compare with": f"{p.name} (archive)", "value [d]": p.period_days, "ratio": rel["ratio"],
                          "≈ p/q": str(rel["fraction"]), "relation": rel["kind"]})
        if comps:
            st.markdown("**Rational approximation of period ratios** (continued fractions, denominator ≤ 4): "
                        "a ratio near 1/2 or 2 reveals an alias (half or double period).")
            st.dataframe(pd.DataFrame(comps), hide_index=True)
with t4:
    unit_name = st.selectbox("Discretise time in units of", ["1 hour", "10 minutes", "1 cadence"])
    unit = {"1 hour": 1 / 24, "10 minutes": 10 / 1440, "1 cadence": r.processed.cadence_days}[unit_name]
    res = modular_residues(times, P, t0, unit)
    k = min(len(times), 12)
    st.markdown(f"Integer times T = round((t − t₀)/unit), modulus m = round(P/unit) = **{res['modulus']}** units.")
    st.dataframe(pd.DataFrame({"event": labels[:k], "T (units)": res["T"][:k],
                               "T mod m": res["residues"][:k], "distance to 0 (mod m)": res["circular"][:k]}),
                 hide_index=True)
    ints = [int(x) for x in np.diff(res["T"])[: k - 1]]
    if len(ints) >= 2:
        g = gcd_list(ints)
        st.markdown(f"Exact GCD of the first integer intervals {ints[:8]}{'…' if len(ints) > 8 else ''} = **{g}** "
                    f"units, compared with m = {res['modulus']}.")
        st.caption("Exact integer GCDs are fragile: a single unit of timing noise changes the answer. That is why the "
                   "tolerance-based congruence test is used for real data.")

learned([
    f"At P = {P:.5f} d, {consistent} of {len(rows)} intervals are whole numbers of orbits within ±{tol_h:g} h, and "
    f"{cong} of {len(times)} events are congruent to t₀ modulo P.",
    "Intervals with n > 1 show missed transits — taking the median interval (the original v1 method) would be "
    "fooled by them.",
    "The period is an approximate GCD; exact integer GCDs break down as soon as timing noise is present.",
    "Periods related by small rational ratios (1/2, 2) are aliases — number theory tells you which periods are "
    "'the same signal'.",
])
