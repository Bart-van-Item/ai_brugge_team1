import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import SITES

from dashboard_app.config import SITE_COLORS, SITE_FILL, SITE_INFO, RESAMPLE_RULES, PLOTLY_LAYOUT
from dashboard_app.data import get_daily_energy, period_delta, in_range
from dashboard_app.widgets import filter_controls


def page_overview():
    st.title("Solar PV Dashboard")
    st.markdown(
        "Output, weather and machine learning for **three PV installations** in the Bruges region. "
        "Use the menu on the left to explore each site, compare them, or try the prediction model."
    )
    date_range, selected_sites = filter_controls("overview")

    cols = st.columns(len(SITES))
    for col, name in zip(cols, SITES):
        daily = get_daily_energy(name)
        delta = period_delta(daily, date_range)
        col.metric(
            SITE_INFO[name]["label"],
            f"{daily[lambda s: in_range(s.index, date_range)].sum():,.0f} kWh",
            delta=delta,
            help=f"Total over the selected range vs the equal-length period before it. {SITES[name]['kwp']} kWp installed.",
        )

    st.subheader("Energy output over time")
    resolution = st.radio("Aggregation", list(RESAMPLE_RULES), horizontal=True, index=0, key="ov_res",
                          help="How energy is summed before plotting: per day, per week, or per month. Coarser aggregation smooths out daily weather noise.")
    fig = go.Figure()
    for name in selected_sites:
        agg = get_daily_energy(name)[lambda s: in_range(s.index, date_range)].resample(
            RESAMPLE_RULES[resolution]).sum(min_count=1)
        color = SITE_COLORS[name]
        fig.add_trace(go.Scatter(
            x=agg.index, y=agg.values, name=name, legendgroup=name,
            fill="tozeroy", mode="lines",
            line=dict(color=color, width=1.5),
            fillcolor=SITE_FILL[name],
        ))
    fig.update_layout(yaxis_title="Energy (kWh)", xaxis_title="Date",
                      title=f"{resolution} energy output per site", height=450,
                      **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")


def page_data_guide():
    st.title("Data guide")
    st.markdown(
        "Everything is built from per-15-minute **weather** data (Open-Meteo) and **PV output** data, "
        "for three sites. The full column reference is in `docs/data-dictionary.md`."
    )
    st.subheader("The three data sources")
    st.dataframe(pd.DataFrame([
        {"source": "PV output (house1, house2)", "format": "CSV, comma decimals", "unit": "Wh per 15 min",
         "note": "one row per timestamp, from the inverter"},
        {"source": "Reactor meter", "format": "CSV, semicolons, comma decimals, BOM", "unit": "kWh per 15 min",
         "note": "grid meter, 3 register rows per timestamp; empty = no reading"},
        {"source": "Weather (all sites)", "format": "CSV, 3 metadata rows then header", "unit": "various",
         "note": "Open-Meteo; column order differs per site"},
    ]), width="stretch", hide_index=True)
    st.caption("All energy is normalized to kWh and joined per 15 minutes in the cleaning step (prep_data.py).")

    st.subheader("What we know about the panels")
    st.markdown(
        "- **No panel brand, model or type** is in the data, only capacity (kWp), inverter size and array layout.\n"
        "- The **EAN code** in the reactor file (`541454897100239158`) is a Belgian grid connection ID "
        "(Fluvius), it identifies the metering point, not the panel.\n"
        "- The houses have no EAN, their data comes from the inverter, not the grid meter."
    )

    st.subheader("Weather columns (the predictors)")
    st.dataframe(pd.DataFrame([
        {"column": "shortwave_radiation", "unit": "W/m²", "meaning": "global horizontal irradiance — main driver"},
        {"column": "direct / diffuse / direct_normal", "unit": "W/m²", "meaning": "beam vs scattered components"},
        {"column": "global_tilted_irradiance", "unit": "W/m²", "meaning": "irradiance on a tilted plane (panel-like)"},
        {"column": "terrestrial_radiation", "unit": "W/m²", "meaning": "clear-sky theoretical maximum"},
        {"column": "temperature_2m / humidity / dew_point", "unit": "°C / %", "meaning": "air conditions"},
        {"column": "weather_code", "unit": "WMO", "meaning": "0 clear … 45 fog, 51-55 drizzle, 61-65 rain, 71-75 snow"},
        {"column": "is_day", "unit": "0/1", "meaning": "daylight flag"},
    ]), width="stretch", hide_index=True)

    st.subheader("Sources & attribution")
    st.markdown(
        "- **Weather (historical and forecast):** [Open-Meteo](https://open-meteo.com) for the Bruges "
        "region, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); based on ERA5 and "
        "national weather model data.\n"
        "- **PV output:** per-15-minute inverter and grid-meter readings from the three installations, "
        "collected by AI Brugge Team 1 since January 2025."
    )
