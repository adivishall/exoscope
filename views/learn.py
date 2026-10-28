import pandas as pd
import streamlit as st

from exoscope.learning.content import FORMULAS, GLOSSARY, LESSONS
from exoscope.learning.quiz import build_quiz, score
from exoscope.ui.common import analysis

st.title("Learning Center")
tab_lessons, tab_quiz, tab_gloss, tab_formula = st.tabs(["Guided lessons", "Quizzes", "Glossary", "Formula reference"])

with tab_lessons:
    titles = [l["title"] for l in LESSONS]
    choice = st.radio("Lesson", titles, horizontal=True)
    lesson = LESSONS[titles.index(choice)]
    key = f"step_{lesson['id']}"
    st.session_state.setdefault(key, 0)
    step = st.session_state[key]
    n = len(lesson["steps"])
    st.progress((step + 1) / n, text=f"Step {step + 1} of {n}")
    with st.container(border=True):
        st.markdown(lesson["steps"][step])
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("◀ Back", disabled=step == 0, key=f"back_{lesson['id']}"):
        st.session_state[key] -= 1
        st.rerun()
    if c2.button("Next ▶", disabled=step == n - 1, type="primary", key=f"next_{lesson['id']}"):
        st.session_state[key] += 1
        st.rerun()
    if step == n - 1:
        st.success(f"**What did we learn?** {lesson['learned']}", icon=":material/lightbulb:")

with tab_quiz:
    r = analysis()
    topics = sorted({q.topic for q in build_quiz(r)})
    picked = st.pills("Topics", topics, default=topics, selection_mode="multi")
    st.session_state.setdefault("quiz_attempt", 0)
    attempt = st.session_state.quiz_attempt
    qs = build_quiz(r, picked or topics, seed=attempt)
    if r is not None:
        st.caption(f"Questions marked ★ use numbers from your current analysis of {r.dataset_meta.get('target')}.")
    else:
        st.caption("Analyse a light curve to unlock questions (★) that use your own results.")
    submitted = st.session_state.get(f"quiz_submitted_{attempt}", False)
    answers = {}
    for i, q in enumerate(qs, 1):
        with st.container(border=True):
            st.markdown(f"**Q{i}. {'★ ' if q.dynamic else ''}[{q.topic}]** {q.prompt}")
            choice = st.radio("Answer", range(len(q.options)), format_func=lambda j, q=q: q.options[j], index=None,
                              key=f"quiz_{attempt}_{q.qid}", label_visibility="collapsed", disabled=submitted)
            answers[q.qid] = choice
            if submitted:
                if q.check(choice):
                    st.success(f"Correct. {q.explanation}", icon=":material/check:")
                else:
                    shown = "no answer" if choice is None else f"you chose “{q.options[choice]}”"
                    st.error(f"Not quite ({shown}). Answer: **{q.options[q.answer]}**. {q.explanation}",
                             icon=":material/close:")
    if not submitted:
        if st.button("Submit answers", type="primary", icon=":material/task_alt:"):
            st.session_state[f"quiz_submitted_{attempt}"] = True
            st.rerun()
    else:
        got, total = score(qs, answers)
        st.metric("Score", f"{got} / {total}", f"{100 * got / total:.0f}%" if total else None, delta_color="off")
        if st.button("Retry with shuffled options", icon=":material/refresh:"):
            st.session_state.quiz_attempt += 1
            st.rerun()

with tab_gloss:
    q = st.text_input("Search terms", placeholder="e.g. congruence, BLS, clique")
    areas = sorted({g["area"] for g in GLOSSARY})
    area = st.pills("Area", areas, default=areas, selection_mode="multi")
    rows = [g for g in GLOSSARY if g["area"] in (area or areas)
            and (not q or q.lower() in (g["term"] + g["definition"] + g["example"]).lower())]
    st.dataframe(pd.DataFrame(rows)[["term", "area", "definition", "example"]] if rows else pd.DataFrame(),
                 hide_index=True, height=min(700, 38 + 35 * max(1, len(rows))),
                 column_config={"definition": st.column_config.TextColumn(width="large"),
                                "example": st.column_config.TextColumn(width="large")})
    if not rows:
        st.caption("No matching terms.")

with tab_formula:
    for f in FORMULAS:
        with st.container(border=True):
            st.markdown(f"**{f['name']}**")
            st.latex(f["latex"])
            if f["note"]:
                st.caption(f["note"])
