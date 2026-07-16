import streamlit as st

from analysis import SITES

from dashboard_app.data import date_bounds

# --- global sidebar controls, shared by every page --------------------------

st.set_page_config(page_title="PV Dashboard — AI Brugge Team 1", layout="wide")

MIN_DATE, MAX_DATE = date_bounds()


def filter_controls(key: str, with_sites: bool = True):
    """Date range (+ optional site filter) shown inline above the graphs, inside
    an expander so it stays close to the charts without taking much space.
    The active range shows in the collapsed header."""
    current = st.session_state.get(f"{key}_date", (MIN_DATE, MAX_DATE))
    with st.expander(f"Filters — {current[0]:%d %b %Y} to {current[1]:%d %b %Y}", expanded=False):
        date_range = st.slider(
            "Date range", min_value=MIN_DATE, max_value=MAX_DATE,
            value=(MIN_DATE, MAX_DATE), key=f"{key}_date",
        )
        selected_sites = list(SITES)
        if with_sites:
            selected_sites = st.multiselect(
                "Sites", options=list(SITES), default=list(SITES), key=f"{key}_sites",
            ) or list(SITES)
    return date_range, selected_sites
