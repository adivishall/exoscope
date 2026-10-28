"""ExoScope — entry point.  Run with:  streamlit run app.py"""

import warnings

import streamlit as st

warnings.filterwarnings("ignore", module="lightkurve")

from exoscope.ui.common import PAGES, init_state, sidebar_status  # noqa: E402

st.set_page_config(page_title="ExoScope", page_icon=":material/radar:", layout="wide",
                   initial_sidebar_state="auto")
init_state()

st.markdown(
    """<style>
    .block-container {padding-top: 2.2rem; max-width: 1280px;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    div[data-testid="stExpander"] details summary p {font-weight: 600;}
    </style>""",
    unsafe_allow_html=True,
)

P = st.Page
nav = st.navigation(
    {
        "ExoScope": [
            P(PAGES["home"], title="Home", icon=":material/home:", default=True),
            P(PAGES["explore"], title="Explore Targets", icon=":material/travel_explore:"),
            P(PAGES["analyze"], title="Analyze Light Curves", icon=":material/monitoring:"),
            P(PAGES["results"], title="Candidate Results", icon=":material/fact_check:"),
        ],
        "Discrete Mathematics Lab": [
            P(PAGES["lab"], title="Lab overview", icon=":material/science:"),
            P(PAGES["sets"], title="Set Theory", icon=":material/join_inner:"),
            P(PAGES["numbers"], title="Number Theory", icon=":material/tag:"),
            P(PAGES["graphs"], title="Graph Theory", icon=":material/hub:"),
            P(PAGES["relations"], title="Relations & Functions", icon=":material/arrow_right_alt:"),
            P(PAGES["combinatorics"], title="Combinatorics", icon=":material/grid_view:"),
        ],
        "Learn & review": [
            P(PAGES["learn"], title="Learning Center", icon=":material/menu_book:"),
            P(PAGES["history"], title="Analysis History", icon=":material/history:"),
            P(PAGES["about"], title="About / Methodology", icon=":material/info:"),
        ],
    },
    expanded=True,
)
sidebar_status()
nav.run()
