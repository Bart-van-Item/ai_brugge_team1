import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import SITES

from dashboard_app.config import SITE_COLORS, SITE_FILL, SITE_INFO, TILT_NOTE, RESAMPLE_RULES, PLOTLY_LAYOUT
from dashboard_app.data import (get_daily_energy, get_coverage, get_ml_csv, get_daily_profile,
                                in_range, MONTH_ABBR)
from dashboard_app.widgets import filter_controls


def render_site(name: str):
    info = SITE_INFO[name]
    st.title(info["label"])
    date_range, _ = filter_controls(f"site_{name}", with_sites=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Installed", f"{SITES[name]['kwp']} kWp",
              help="Total DC capacity of the solar panels in kilowatt-peak (kWp) — the rated output under standard test conditions: 1000 W/m² irradiance, 25 °C panel temperature.")
    c2.metric("Inverter", f"{info['inverter_kw']} kW",
              help="The inverter converts DC (direct current) from the panels into AC (alternating current) for household use and the grid. "
                   "This is the maximum AC power it can export.")
    c3.metric("DC/AC ratio", f"{info['dcac']}",
              help=f"DC (direct current) is the power the solar panels generate; AC (alternating current) is what the inverter outputs to the grid and household. "
                   f"The ratio is panel capacity ({SITES[name]['kwp']} kWp DC) divided by inverter capacity ({info['inverter_kw']} kW AC). "
                   f"Above 1.0 means the panels can produce more than the inverter can export, so output is clipped on very sunny days. "
                   f"This is intentional: sunny peak hours are short, so oversizing the panels increases total yield without needing a bigger inverter.")
    tilt = get_ml_csv("tilt_results.csv").set_index("site").loc[name]
    c4.metric("Orientation", f"{tilt['facing']} · {tilt['tilt_deg']}° tilt",
              help=f"Estimated from the data by matching simulated panel planes to the measured output "
                   f"(method on the Models page). Azimuth {tilt['azimuth_deg']}° — 90° is east, 180° south, "
                   f"270° west — plausible range {tilt['azimuth_ridge']}°; tilt plausible range "
                   f"{tilt['tilt_ridge']}°. {TILT_NOTE[name]}")
    st.caption(info["arrays"])

    cov = get_coverage(name)
    span_note = (f"Data coverage: {cov['start']} to {cov['end']} "
                 f"({cov['days']} days).")
    if cov["missing_months"]:
        missing = ", ".join(MONTH_ABBR[m - 1] for m in cov["missing_months"])
        st.info(
            f"{span_note} This site has no data yet for: **{missing}**. "
            "Model predictions for those months are extrapolation, based on "
            "irradiance rather than direct experience of that season.",
            icon="ℹ️",
        )
    else:
        st.caption(f"{span_note} Full calendar year covered.")

    daily = get_daily_energy(name)[lambda s: in_range(s.index, date_range)]
    st.subheader("Daily output",
                 help="Total energy produced each day (kWh). The dashed line is a centered 7-day average that smooths out day-to-day weather swings.")
    color = SITE_COLORS[name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily.values, name="Daily",
        fill="tozeroy", mode="lines",
        line=dict(color=color, width=1),
        fillcolor=SITE_FILL[name],
    ))
    if len(daily) >= 7:
        rolling = daily.rolling(7, center=True, min_periods=4).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values, name="7-day avg",
            mode="lines", line=dict(color=color, width=2.5, dash="dash"),
        ))
    fig.update_layout(height=380, title=f"{info['label']} daily energy", yaxis_title="Energy (kWh)",
                      xaxis_title="Date", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Average day shape",
                 help="Mean output per 15-min slot across sunny days, showing the typical shape of a production day. The peak hour reveals orientation: morning peak = east-facing, midday = south, evening = west.")
    profile = get_daily_profile(name)
    fig = px.line(x=profile.index, y=profile.values, markers=True,
                  labels={"x": "Hour of day (UTC)", "y": "Mean output (kWh / 15 min)"},
                  color_discrete_sequence=[color])
    fig.update_layout(height=340, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Average over sunny days. Peak around **{profile.idxmax():02d}:00 UTC**, consistent with the "
        f"array facing **{tilt['facing']}** (morning peak = east, midday = south, evening = west)."
    )


def page_house1():
    render_site("house1")


def page_house2():
    render_site("house2")


def page_reactor():
    render_site("reactor")


def page_compare():
    st.title("Compare sites")
    st.markdown("Pick what to compare across the three installations.")
    date_range, selected_sites = filter_controls("compare")
    view = st.selectbox(
        "Compare by",
        ["Specific yield (kWh/kWp)", "Output over time", "Average day shape", "Characteristics table"],
        help="Specific yield (kWh/kWp) is energy output divided by installed panel capacity. "
             "It is the fair way to compare installations of different sizes: it answers how much each site produces per unit of panel, not in total.",
    )

    if view == "Characteristics table":
        tilt = get_ml_csv("tilt_results.csv").set_index("site")
        rows = []
        for name in SITES:
            daily = get_daily_energy(name)[lambda s: in_range(s.index, date_range)]
            rows.append({
                "site": SITE_INFO[name]["label"], "kWp": SITES[name]["kwp"],
                "inverter (kW)": SITE_INFO[name]["inverter_kw"], "DC/AC": SITE_INFO[name]["dcac"],
                "orientation": f"{tilt.loc[name, 'facing']} ({tilt.loc[name, 'azimuth_deg']}°)",
                "tilt (est.)": f"~{tilt.loc[name, 'tilt_deg']}°",
                "mean daily kWh": round(daily.mean(), 1),
                "mean kWh/kWp": round((daily / SITES[name]["kwp"]).mean(), 2),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Specific yield (kWh/kWp) is the fair comparison: it removes the effect of installation size.")
        return

    if view == "Specific yield (kWh/kWp)":
        fig = go.Figure()
        for name in selected_sites:
            daily = get_daily_energy(name)[lambda s: in_range(s.index, date_range)]
            sy = (daily / SITES[name]["kwp"]).resample("MS").mean()
            fig.add_trace(go.Bar(x=sy.index, y=sy.values, name=name, marker_color=SITE_COLORS[name]))
        fig.update_layout(barmode="group", height=450, yaxis_title="Mean daily kWh/kWp",
                          xaxis_title="Month", title="Specific yield per month (size-normalized)",
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        st.caption("Same panel area would produce this per kWp. Removes the size advantage of the reactor.")
        return

    if view == "Output over time":
        resolution = st.radio("Aggregation", list(RESAMPLE_RULES), horizontal=True, index=2, key="cmp_res")
        fig = go.Figure()
        for name in selected_sites:
            agg = get_daily_energy(name)[lambda s: in_range(s.index, date_range)].resample(
                RESAMPLE_RULES[resolution]).sum(min_count=1)
            fig.add_trace(go.Scatter(x=agg.index, y=agg.values, name=name, line=dict(color=SITE_COLORS[name])))
        fig.update_layout(height=450, yaxis_title="Energy (kWh)", xaxis_title="Date",
                          title=f"{resolution} output per site", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        return

    # Average day shape, each site normalized to its own peak so the shapes
    # overlay; otherwise the reactor dwarfs the houses and hides the timing
    fig = go.Figure()
    for name in selected_sites:
        profile = get_daily_profile(name)
        rel = profile / profile.max()
        fig.add_trace(go.Scatter(x=rel.index, y=rel.values, name=name,
                                 mode="lines+markers", line=dict(color=SITE_COLORS[name])))
    fig.update_layout(height=450, xaxis_title="Hour of day (UTC)",
                      yaxis_title="Relative output (share of own peak)",
                      title="Average day shape per site (normalized)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Each curve is scaled to its own peak, so only the timing differs. The reactor's "
               "near-flat east-west layout gives a broad curve centred near solar noon; the steep "
               "west-south-west houses peak later in the afternoon.")
