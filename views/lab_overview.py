import streamlit as st

from exoscope.ui.common import PAGES, analysis, empty_state

st.title("Discrete Mathematics Lab")
st.markdown(
    "Each module runs a discrete-mathematics method on **your current analysis** — the same events, periods and "
    "settings shown on the Analyze page. Every module follows four steps: **① concept → ② worked example on a tiny "
    "dataset → ③ your data, interactively → ④ what did we learn?** Toggle *Guided learning mode* in the sidebar to "
    "open or close the explanations."
)

r = analysis()
if r is None:
    empty_state("Load and analyse a light curve first",
                "The lab works on real detector output. Start with the Kepler-8 sample or pick a target.")
else:
    s = r.comparison.summary()
    st.success(f"Using the analysis of **{r.dataset_meta.get('target')}** — |A| = {s['|A|']} threshold events, "
               f"|B| = {s['|B|']} BLS epochs" + (f", BLS period {r.best.period:.5f} d." if r.best else "."),
               icon=":material/check_circle:")

modules = [
    ("sets", "Set Theory", "Which detector found which transit?",
     "Union, intersection, difference, complement relative to a defined universe, inclusion–exclusion, Jaccard "
     "similarity and precision/recall as set operations."),
    ("numbers", "Number Theory", "Why do repeated dips imply a period?",
     "Intervals as integer multiples, tolerance divisibility, congruence mod P on a modular clock, Euclid's GCD vs "
     "approximate GCD, O−C diagrams and alias detection with rational approximations."),
    ("graphs", "Graph Theory", "Which events support each other?",
     "Event network with a documented edge rule, connected components, cliques, adjacency matrix, and why the "
     "consistency relation is not transitive. Click a vertex to see it on the light curve."),
    ("relations", "Relations & Functions", "What structure does the pipeline have?",
     "Domain/codomain, ordered pairs, mapping diagrams and relation matrices; function, injective, surjective "
     "checks; one-to-one / one-to-many / many-to-one / many-to-many; reflexive, symmetric, transitive."),
    ("combinatorics", "Combinatorics", "How many experiments are there, and which matter?",
     "Cartesian products of parameter sets, rule of product, constraints, projections, C(n, k) subset selection and "
     "execution of selected configurations with a comparison of results."),
]
for key, title, question, text in modules:
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        c1.page_link(PAGES[key], label=title, icon=":material/arrow_forward:")
        c1.caption(question)
        c2.markdown(text)
