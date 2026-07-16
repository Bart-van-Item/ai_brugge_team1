"""
Interactive Streamlit dashboard for the PV / weather analysis.

Multipage app (sidebar navigation), grouped into:
- Start:    Overview, Data guide
- Sites:    House 1, House 2, Reactor, Compare
- Analysis: Time of day, Weather, Anomalies
- Machine learning: Models, Predict

Run: streamlit run dashboard.py
"""

import streamlit as st

from dashboard_app.pages.start import page_overview, page_data_guide
from dashboard_app.pages.sites import page_house1, page_house2, page_reactor, page_compare
from dashboard_app.pages.analysis import page_time_of_day, page_weather, page_anomalies
from dashboard_app.pages.ml import page_ml_models, page_predict, page_today, page_this_week, page_replay

nav = st.navigation({
    "Start": [
        st.Page(page_overview, title="Overview", default=True),
        st.Page(page_data_guide, title="Data guide"),
    ],
    "Sites": [
        st.Page(page_house1, title="House 1"),
        st.Page(page_house2, title="House 2"),
        st.Page(page_reactor, title="Reactor"),
        st.Page(page_compare, title="Compare"),
    ],
    "Analysis": [
        st.Page(page_time_of_day, title="Time of day"),
        st.Page(page_weather, title="Weather"),
        st.Page(page_anomalies, title="Anomalies"),
    ],
    "Machine learning": [
        st.Page(page_ml_models, title="Models"),
        st.Page(page_predict, title="Predict"),
        st.Page(page_today, title="Today"),
        st.Page(page_this_week, title="This week"),
        st.Page(page_replay, title="Replay a day"),
    ],
}, expanded=True)  # always show all pages, no "View more" collapse
st.sidebar.caption("Per-15-min PV & weather · Bruges region · since Jan 2025 · "
                   "weather by [Open-Meteo](https://open-meteo.com) (CC BY 4.0)")
_, _qr_col, _ = st.sidebar.columns([1, 4, 1])
_qr_col.image("assets/qrcode_ai-brugge-team1.png", width="stretch",
              caption="ai-brugge-team1.streamlit.app")
nav.run()
