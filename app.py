"""Streamlit entry point.

Launch from the project folder:

    py -3.13 -m streamlit run app.py
"""

import streamlit as st

from screening.dashboard import DISCLAIMER

st.set_page_config(
    page_title="Diabetes screening research prototype",
    page_icon=":material/fact_check:",
    layout="wide",
)

page = st.navigation(
    {
        "Screening tool": [
            st.Page(
                "app_pages/screening_tool.py",
                title="Screen a person",
                icon=":material/assignment_ind:",
                help="Enter a person's answers during a health camp. The tool picks Set A, B or C "
                     "automatically and shows an alert based on the chosen threshold.",
            ),
            st.Page(
                "app_pages/tool_settings.py",
                title="Tool settings",
                icon=":material/settings:",
                help="Choose the alert and priority thresholds, overall or by age group, "
                     "and see what each option means.",
            ),
        ],
        "Research": [
        st.Page(
            "app_pages/overview.py",
            title="Overview",
            icon=":material/home:",
            default=True,
            help="The research question, target definition, dataset size, class balance, "
                 "and what feature sets A, B and C contain.",
        ),
        st.Page(
            "app_pages/explore.py",
            title="Explore the data",
            icon=":material/query_stats:",
            help="Missing values, class distribution, BMI, predictor summaries, "
                 "and the repeated-answer-pattern caveat.",
        ),
        st.Page(
            "app_pages/compare.py",
            title="Compare models",
            icon=":material/bar_chart:",
            help="Held-out test results for every model and feature set, ROC and "
                 "precision–recall curves, confusion counts, and validation thresholds.",
        ),
        st.Page(
            "app_pages/demo.py",
            title="Try the three-question model",
            icon=":material/tune:",
            help="Enter Set C answers and see the uncalibrated model score relative "
                 "to the research threshold. Not a diagnosis.",
        ),
        ],
    },
    position="top",
)

st.title(page.title, anchor=False)
st.caption(f":material/science: {DISCLAIMER}")

page.run()
