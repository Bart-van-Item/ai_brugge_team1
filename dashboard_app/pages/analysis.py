import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import SITES
from weather_correlation import RAD_COL, TEMP_COL

from dashboard_app.config import SITE_COLORS, SITE_INFO, PLOTLY_LAYOUT
from dashboard_app.data import (get_joined, get_yield_ratio, get_anomalies,
                                get_daily_profile, in_range)
from dashboard_app.widgets import filter_controls


def page_time_of_day():
    st.title("Time of day")
    st.markdown("How output is distributed across the day. Pick a site, or compare all of them.")
    choice = st.selectbox("Site", ["All sites"] + [SITE_INFO[s]["label"] for s in SITES])

    fig = go.Figure()
    names = list(SITES) if choice == "All sites" else [s for s in SITES if SITE_INFO[s]["label"] == choice]
    for name in names:
        profile = get_daily_profile(name)
        fig.add_trace(go.Scatter(x=profile.index, y=profile.values, name=SITE_INFO[name]["label"],
                                 mode="lines+markers", line=dict(color=SITE_COLORS[name])))
    fig.update_layout(height=460, xaxis_title="Hour of day (UTC)",
                      yaxis_title="Mean output (kWh / 15 min)", title="Average output by hour of day",
                      **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Averaged over sunny days. Times are UTC; solar noon at this longitude is ~11:47 UTC.")


def page_weather():
    st.title("Weather and output")
    date_range, selected_sites = filter_controls("weather")

    st.subheader("Irradiance vs output",
                 help="Irradiance is the solar power hitting the ground (W/m²), the main driver of PV output. A tight, straight cloud of points means output follows the sun closely, as expected from a healthy installation.")
    st.caption("Each point is one quarter-hour. Stronger sites track irradiance more tightly.")
    site = st.selectbox("Site", selected_sites, key="weather_site")
    df = get_joined(site)
    df = df[in_range(df.index, date_range)]
    corr = df["energy"].corr(df[RAD_COL]) if len(df) else float("nan")
    st.metric("corr(energy, irradiance)", f"{corr:.3f}",
              help=f"Pearson correlation between irradiance (W/m²) and energy output (kWh) per 15-min slot. "
                   f"1.0 = perfect linear relationship, 0 = no relationship. Above 0.95 is expected for a well-functioning installation. "
                   f"Calculated over {len(df):,} quarter-hours in the selected period.")
    df_plot = df.copy()
    df_plot["month"] = df_plot.index.month_name()
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    df_plot["month"] = pd.Categorical(df_plot["month"], categories=month_order, ordered=True)
    fig = px.scatter(df_plot, x=RAD_COL, y="energy", color="month", opacity=0.4,
                     category_orders={"month": month_order},
                     color_discrete_sequence=px.colors.cyclical.HSV,
                     labels={"energy": "Energy per 15 min (kWh)", "month": "Month"})
    fig.update_layout(height=450, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Temperature effect at fixed irradiance")
    st.caption("Irradiance band fixed to compare comparable light. The upward trend is a seasonal artefact (see below).")
    band_low, band_high = st.slider("Irradiance band (W/m²)", 0, 1000, (400, 600), step=50, key="weather_band",
                                    help="Only compare quarter-hours with roughly equal sunlight. Fixing the light level isolates the temperature effect from the effect of how bright it is.")
    band = df[(df[RAD_COL] >= band_low) & (df[RAD_COL] <= band_high)].copy()
    if len(band):
        band["temp_bin"] = pd.cut(band[TEMP_COL], bins=[-10, 5, 10, 15, 20, 25, 30, 40])
        grouped = band.groupby("temp_bin", observed=True)["energy_per_kwp"].mean().reset_index()
        grouped["temp_bin"] = grouped["temp_bin"].astype(str)
        fig = px.bar(grouped, x="temp_bin", y="energy_per_kwp",
                     labels={"temp_bin": "Temperature (°C)", "energy_per_kwp": "Mean kWh/kWp per 15 min"},
                     color_discrete_sequence=[SITE_COLORS[site]])
        fig.update_layout(height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No data in this irradiance band for the selected period.")

    st.info(
        "**Why temperature seems to raise yield:** hotter panels are actually less efficient, but at "
        "fixed irradiance, temperature still correlates with season/sun angle, so the trend is a "
        "seasonal artefact, not a physical gain."
    )


def page_anomalies():
    st.title("Underperforming days")
    st.caption(
        "Daily yield ratio = kWh/kWp output per W/m² of that day's total irradiance. "
        "Days far below the site's own median are flagged."
    )
    date_range, selected_sites = filter_controls("anomalies")
    col1, col2 = st.columns(2)
    z_threshold = col1.slider("Anomaly threshold (z-score)", -3.0, -0.5, -1.5, step=0.1, key="anom_z",
                              help="A z-score measures how far a day sits from the site's own average, counted in standard deviations. "
                                   "0 is an average day, -1 is one standard deviation below average, -2 is well below. "
                                   "A day is flagged when its yield ratio falls below this threshold, so a more negative value is stricter and flags fewer days.")
    min_rad = col2.slider("Min daily irradiance (W/m²)", 0, 5000, 1000, step=250, key="anom_rad",
                          help="Daily irradiance is the total sunlight energy that reached the panels that day, summed over all quarter-hours. "
                               "Days below this level were too dark or cloudy to judge output fairly, so they are skipped. Raise it to only compare bright days.")

    fig = go.Figure()
    all_anomalies = []
    for name in selected_sites:
        daily = get_yield_ratio(name, min_rad)[lambda d: in_range(d.index, date_range)]
        anomalies = get_anomalies(name, z_threshold, min_rad)
        anomalies = anomalies[in_range(anomalies.index, date_range)]
        # legendgroup ties the anomaly markers to their site's line, so hiding a
        # site via the legend hides its flagged days too
        fig.add_trace(go.Scatter(x=daily.index, y=daily["ratio"], name=name,
                                 legendgroup=name,
                                 line=dict(color=SITE_COLORS[name], width=1.5)))
        if len(anomalies):
            fig.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies["ratio"], mode="markers",
                name=f"{name} anomaly", showlegend=False, legendgroup=name,
                marker=dict(color="#e63946", size=10, symbol="x-open", line=dict(width=2.5, color="#e63946")),
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    f"Site: {name}<br>"
                    "Yield ratio: %{y:.4f}<br>"
                    "z-score: %{customdata:.2f}<extra></extra>"
                ),
                customdata=anomalies["z_score"].values,
            ))
            tmp = anomalies[["ratio", "z_score"]].copy()
            tmp.insert(0, "date", anomalies.index)
            tmp.insert(0, "site", name)
            all_anomalies.append(tmp.reset_index(drop=True))
    fig.update_layout(yaxis_title="kWh/kWp per W/m²", xaxis_title="Date",
                      title="Daily yield ratio with flagged days", height=450,
                      **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    if all_anomalies:
        table = pd.concat(all_anomalies).sort_values("date")
        st.subheader(f"{len(table)} flagged day-site combinations",
                     help="Each row is one site on one day where the yield ratio fell far below that site's own median. "
                          "Days flagged at multiple sites at once point to a shared weather cause rather than a local fault.")
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.info("No anomalies at these thresholds for the selected period.")

    st.info(
        "**The shared bad days were drizzle/fog, not snow.** Three days were flagged at every site at "
        "once (2026-01-10, 2025-12-23, 2025-11-20). The WMO codes show drizzle, rain and fog with very "
        "high humidity, a shared weather cause rather than a per-site fault."
    )
