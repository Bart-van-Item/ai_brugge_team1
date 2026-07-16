from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import SITES

from dashboard_app.config import SITE_COLORS, SITE_DOT, SITE_INFO, PLOTLY_LAYOUT
from dashboard_app.data import get_joined, get_ml_csv, get_orientations, get_clipping_curve
from dashboard_app.predictions import (predict_compact, predict_sweep, predict_day, predict_forecast,
                                       fetch_forecast, get_day_backtest, weather_icon,
                                       _fetch_empty_state, DAY_PROFILES, OPEN_METEO_CREDIT)
from dashboard_app.widgets import MIN_DATE, MAX_DATE


def page_ml_models():
    st.title("Machine learning: models")
    st.caption("How we predict PV output, which models we tried, and the reasoning behind the choices.")

    st.subheader("Why per-site models: installations differ",
                 help="Clipping is when the panels generate more power than the inverter can convert, so the inverter caps the output. "
                      "This shows up as the output curve flattening at high irradiance, and it happens at a different point for each site, so one shared model would misread it.")
    st.markdown(
        "A model trained across sites could confuse hardware differences with weather/orientation. The "
        "**DC/AC ratio** (panel kWp vs inverter kW) differs a lot, and a high ratio means the inverter "
        "**clips** output at high irradiance, a per-site non-linearity from hardware alone. "
        "We verified this empirically: a pooled per-kWp model trained on all three sites scored *worse* "
        "on the reactor than its own single-site model (R² 0.86 vs 0.80), so per-site models stay."
    )
    st.markdown("**Clipping is visible in the data** — max output flattens once the inverter limit is hit:")
    clip = get_clipping_curve()
    fig = px.line(clip, x="irradiance", y="rel_max_output", color="site", markers=True,
                  color_discrete_map=SITE_COLORS,
                  labels={"irradiance": "Irradiance (W/m²)", "rel_max_output": "Max output (relative to peak)"})
    fig.update_layout(height=380, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Model comparison",
                 help="R² (0 to 1) is how much of the output the model explains: 1.0 is perfect, 0 is no better than guessing the average. "
                      "time_split means the model is trained on older data and tested on the newest period, the honest test for time-series. "
                      "A random split would let the model peek at same-day values during training and look better than it really is.")
    st.markdown(
        "Three ML models plus the physics baseline, three ways of splitting the data. We rank on "
        "**time_split** (train on the past, test on the newest period), the only honest split for "
        "time-series: a random split leaks same-day quarters into both train and test."
    )
    results = get_ml_csv("results.csv")
    res_q = results[(results["resolution"] == "quarterly") & (results["method"] == "time_split")]
    fig = px.bar(res_q, x="site", y="r2", color="model", barmode="group",
                 labels={"r2": "R² (time_split)", "site": ""},
                 title="Quarterly model accuracy per site (higher is better)")
    fig.update_layout(height=400, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    best = res_q.loc[res_q.groupby("site")["r2"].idxmax()]
    cols = st.columns(len(best))
    for col, (_, row) in zip(cols, best.iterrows()):
        col.metric(f"{row['site']} best", row["model"],
                   help=f"R² = {row['r2']:.3f} — share of variance explained (1.0 = perfect). "
                        f"MAE = {row['mae_kwh']:.3f} kWh per 15-min slot — average absolute prediction error. "
                        f"Evaluated on a time split: trained on older data, tested on the most recent period.")
    st.caption(
        "Daily totals are no longer a separate model: the quarterly model's predictions are summed "
        "per day, which beat the standalone daily models nearly everywhere (reactor 0.32 → 0.81)."
    )

    st.subheader("What made the models better",
                 help="Four changes, each validated on the honest time split before being adopted. "
                      "Numbers below are R² for the boosting model; the daily columns sum its "
                      "quarter-hourly predictions per day.")
    st.markdown(
        "- **Nights count as zero.** The houses' PV loggers don't report at night, so those quarters "
        "were missing and got dropped: the model never learned that nights produce nothing. A panel at "
        "night produces exactly 0, so missing night quarters are now filled with 0 (roughly doubling "
        "the training data). Missing *daytime* quarters stay excluded: for House 1 many are real "
        "logging outages at productive hours, and zero-filling those made the model worse.\n"
        "- **Clear-sky index.** Measured irradiance divided by the theoretical cloudless maximum: "
        "'how cloudy' as one number, independent of season and hour.\n"
        "- **Physical limits.** Predictions are clipped to the inverter's capacity (kW × 0.25 kWh per "
        "quarter), so no forecast can exceed what the hardware passes.\n"
        "- **Daily = summed quarterly.** The standalone daily models had only 196–534 samples; "
        "summing the quarterly model per day replaced them."
    )
    st.markdown(
        "| R² (time split) | house1 15-min | house2 15-min | reactor 15-min | house1 daily | house2 daily | reactor daily |\n"
        "|---|---|---|---|---|---|---|\n"
        "| before | 0.60 | 0.78 | 0.86 | 0.49 | 0.69 | 0.32 |\n"
        "| after | **0.75** | **0.84** | 0.86 | **0.58** | **0.77** | **0.81** |"
    )
    st.caption(
        "The reactor's 15-min score is flat because its meter already records nights. Hyperparameter "
        "tuning and cross-site pooling were also tested and added nothing, so they were not adopted."
    )

    st.subheader("Does the ML earn its keep? Physics baseline",
                 help="The physics baseline predicts output straight from horizontal irradiance (a single-feature "
                      "linear fit). It's the floor to beat: if a model can't clear it, the extra complexity isn't paying off.")
    st.markdown(
        "Solar output should track sunlight, so the simplest honest benchmark is *output proportional to "
        "irradiance*. The gap between that baseline and the full models is the value the ML actually adds."
    )
    variants = get_ml_csv("variant_results.csv")
    vq = variants[(variants["resolution"] == "quarterly") & (variants["method"] == "time_split") &
                  (variants["variant"] == "fair_lag")]
    phys = vq[vq["model"] == "physics"].set_index("site")["r2"]
    bestml = vq[vq["model"] != "physics"].groupby("site")["r2"].max()
    base = pd.DataFrame({"site": list(SITES)})
    base["physics"] = base["site"].map(phys)
    base["best_ml"] = base["site"].map(bestml)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=base["site"], y=base["physics"], name="Physics baseline",
                         marker_color="#b0b0b0"))
    fig.add_trace(go.Bar(x=base["site"], y=base["best_ml"], name="Best ML model",
                         marker_color="#1f77b4"))
    fig.update_layout(height=380, barmode="group", yaxis_title="R² (time_split)",
                      title="Physics baseline vs best ML model per site", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "On quarter-hourly data the ML roughly doubles the explained variance for the houses "
        "(irradiance alone is a poor proxy once orientation and clipping matter), and still adds a clear "
        "margin at the Reactor, whose near-flat panels track plain irradiance most closely. The baseline "
        "itself improved with the night fill: predicting zero at zero irradiance is trivially right, "
        "which is exactly why the honest comparison keeps it in."
    )

    st.subheader("What lag features add",
                 help="A lag feature is the irradiance from a few steps earlier (15, 30, 60 min ago) plus short "
                      "rolling averages. PV output is autocorrelated, so recent sunlight helps predict the next slot.")
    st.markdown(
        "PV output carries momentum: a bright previous hour usually means a bright next slot. Adding recent "
        "irradiance (`fair_lag`) on top of the leak-free feature set (`fair`) lifts the honest score for the "
        "houses, most on House 1."
    )
    lag_rows = []
    for site in SITES:
        fair = variants[(variants.site == site) & (variants.variant == "fair") &
                        (variants.method == "time_split") & (variants.resolution == "quarterly") &
                        (variants.model != "physics")]["r2"].max()
        lag = variants[(variants.site == site) & (variants.variant == "fair_lag") &
                       (variants.method == "time_split") & (variants.resolution == "quarterly") &
                       (variants.model != "physics")]["r2"].max()
        lag_rows.append({"site": site, "without lags (fair)": round(fair, 3),
                         "with lags (fair_lag)": round(lag, 3), "gain": round(lag - fair, 3)})
    st.dataframe(pd.DataFrame(lag_rows), width="stretch", hide_index=True)
    st.caption(
        "Quarterly, honest time_split. The gain is largest where the raw weather signal is noisier. "
        "For the Reactor the plain feature set was already strong and lags now cost a fraction on this "
        "split; cross-validation still picks fair_lag as its most robust variant."
    )

    st.subheader("General vs specific models",
                 help="A feature is an input the model uses to predict output, such as irradiance, hour of day or temperature. "
                      "n_features is how many inputs each variant uses. The question here is whether a small set of well-chosen features predicts as well as the full set.")
    st.markdown(
        "Does a smaller, focused model do as well as the full one? `direction` (irradiance + hour of "
        "day) is nearly as good as `general` with far fewer features, and `dir_season_temp` matches it."
    )
    combo = get_ml_csv("combo_results.csv")
    pivot = combo.pivot(index="combo", columns="site", values="r2")
    order = ["general", "direction", "dir_season", "dir_temp", "dir_season_temp", "core_hour"]
    pivot = pivot.reindex([c for c in order if c in pivot.index])
    pivot.insert(0, "n_features", combo.groupby("combo")["n_features"].first())
    st.dataframe(pivot.style.format("{:.3f}", subset=list(SITES)), width="stretch")
    st.caption(
        "Surprises: irradiance alone is weak without the hour of day, and adding season hurts under a "
        "time split. Temperature helps, it's a direct physical driver. Note: this experiment predates "
        "the target fixes above, so its absolute scores run lower; the relative comparison is what "
        "drove the feature choices."
    )

    st.subheader("Panel orientation and tilt",
                 help="Estimated purely from the output and weather data: we simulate how much sun a panel "
                      "at every candidate direction and angle would catch, and keep the one that best matches "
                      "the measured output. No site visit needed.")
    st.markdown(
        "We don't know the panels' direction or angle from metadata, so we **reverse-engineer them from "
        "the data**: the sun's position at any moment is pure geometry, so for thousands of virtual panels "
        "(every direction × every angle) we compute what they *would* have produced, and keep the one that "
        "matches the meter best."
    )

    tilt = get_ml_csv("tilt_results.csv")
    fig = go.Figure()
    for _, row in tilt.iterrows():
        name = row["site"]
        color = SITE_COLORS[name]
        label = SITE_INFO[name]["label"]
        az_lo, az_hi = (int(v) for v in row["azimuth_ridge"].split("-"))
        tl_lo, tl_hi = (int(v) for v in row["tilt_ridge"].split("-"))
        # translucent arc = plausible directions, translucent spoke = plausible tilts
        fig.add_trace(go.Scatterpolar(
            theta=list(range(az_lo, az_hi + 1, 2)), r=[row["tilt_deg"]] * len(range(az_lo, az_hi + 1, 2)),
            mode="lines", line=dict(color=color, width=7), opacity=0.3,
            legendgroup=name, showlegend=False,
            hovertemplate=f"plausible directions {az_lo}–{az_hi}°<extra>{label}</extra>",
        ))
        fig.add_trace(go.Scatterpolar(
            theta=[row["azimuth_deg"]] * 2, r=[tl_lo, tl_hi],
            mode="lines", line=dict(color=color, width=7), opacity=0.3,
            legendgroup=name, showlegend=False,
            hovertemplate=f"plausible tilts {tl_lo}–{tl_hi}°<extra>{label}</extra>",
        ))
        fig.add_trace(go.Scatterpolar(
            theta=[row["azimuth_deg"]], r=[row["tilt_deg"]],
            mode="markers", marker=dict(color=color, size=13, line=dict(width=2, color="white")),
            name=label, legendgroup=name,
            hovertemplate=(f"{row['facing']} ({row['azimuth_deg']}°), tilt {row['tilt_deg']}°"
                           f"<extra>{label}</extra>"),
        ))
    fig.update_layout(
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90,
                             tickvals=list(range(0, 360, 45)),
                             ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            radialaxis=dict(range=[0, 65], tickvals=[0, 15, 30, 45, 60],
                            ticksuffix="°", angle=90, tickangle=90),
        ),
        height=440, title="Best-fit plane per site (compass = direction, distance from centre = tilt)",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Read it like a map seen from above: the dot's compass position is where the panels face, "
               "and the further from the centre, the steeper they stand (centre = lying flat). "
               "The shaded arcs are the plausible ranges: short arc = confident estimate.")

    st.dataframe(
        tilt.assign(site=tilt["site"].map(lambda s: SITE_INFO[s]["label"]))[
            ["site", "facing", "azimuth_deg", "azimuth_ridge", "tilt_deg", "tilt_ridge", "r2_midday"]],
        column_config={
            "site": st.column_config.TextColumn("Site"),
            "facing": st.column_config.TextColumn(
                "Facing", help="Compass direction the panels point, from the best-fit azimuth."),
            "azimuth_deg": st.column_config.NumberColumn(
                "Azimuth (°)", help="Compass angle of the panel plane: 90° = east, 180° = south, 270° = west."),
            "azimuth_ridge": st.column_config.TextColumn(
                "Plausible directions", help="All azimuths that explain the data nearly as well as the best one "
                                             "(within 0.01 R²). A narrow range means the direction is well determined."),
            "tilt_deg": st.column_config.NumberColumn(
                "Tilt (°)", help="Angle from horizontal: 0° = lying flat on the roof, 90° = vertical."),
            "tilt_ridge": st.column_config.TextColumn(
                "Plausible tilts", help="All tilts that explain the midday data nearly as well as the best one. "
                                        "The reactor's range is wide on paper (0-28°) but every value in it is a "
                                        "low tilt, so 'nearly flat' is a firm conclusion."),
            "r2_midday": st.column_config.NumberColumn(
                "Midday fit R²", format="%.2f",
                help="How much of the clear-sky midday output the fitted plane explains (1 = perfect). "
                     "Tilt is fitted on midday data only, because there the seasonal sun height isolates "
                     "the tilt signal from morning/evening effects."),
        },
        width="stretch", hide_index=True,
    )

    st.markdown("##### Could there be a second direction?")
    st.markdown(
        "House 1 and the Reactor are known to have panels in **two** directions, so we also fitted every "
        "*pair* of virtual panels, letting the data decide how much capacity faces each way. "
        "Verdict: **a second plane never improves the fit** (gain ≈ 0), so the two faces are too similar "
        "to separate from output data alone. House 2 (single direction) is the built-in control: its fit "
        "correctly collapses back to one plane."
    )
    two = get_ml_csv("two_plane_results.csv")
    two_disp = pd.DataFrame({
        "site": two["site"].map(lambda s: SITE_INFO[s]["label"]),
        "best split found": [
            f"{r.facing_1} {r.azimuth_1}°/{r.tilt_1}° ({r.share_1:.0%}) + "
            f"{r.facing_2} {r.azimuth_2}°/{r.tilt_2}° ({r.share_2:.0%})"
            for r in two.itertuples()],
        "one plane R²": two["r2_single_plane"],
        "two planes R²": two["r2_two_plane"],
        "gain": two["gain"],
    })
    st.dataframe(
        two_disp,
        column_config={
            "site": st.column_config.TextColumn("Site"),
            "best split found": st.column_config.TextColumn(
                "Best split found", help="The best two-direction combination the search found: "
                                         "direction, tilt and share of production for each face."),
            "one plane R²": st.column_config.NumberColumn(
                "One plane R²", format="%.3f", help="Fit quality with a single panel direction."),
            "two planes R²": st.column_config.NumberColumn(
                "Two planes R²", format="%.3f", help="Fit quality when a second direction is allowed."),
            "gain": st.column_config.NumberColumn(
                "Gain", format="%.4f", help="Improvement from allowing a second direction. "
                                            "Near zero = the data does not reveal a separate second face."),
        },
        width="stretch", hide_index=True,
    )
    st.caption(
        "Why the reactor still gives itself away: panels at *low* tilt facing opposite ways see almost the "
        "same sky, so their sum is indistinguishable from one nearly-flat panel. A very low fitted tilt "
        "plus a west-leaning daily shape is exactly the fingerprint of an east-west 'tent' layout."
    )

    with st.expander("How did we get these numbers? (plain-language explanation)"):
        st.markdown(
            "**The one idea behind all of it:** a solar panel produces the most when it points straight "
            "at the sun. So the *shape* of a panel's production, over the day and over the seasons, "
            "betrays which way it points. We never climbed on a roof; the meter told us.\n\n"
            "**Step 1 — build virtual panels.** Where the sun is at any moment is pure math, like "
            "knowing where the hands of a clock are. So we can compute, for any imaginary panel "
            "(pick a direction, pick an angle), how much sun it *would* have caught in every 15-minute "
            "slot of the dataset. We built thousands of these virtual panels: every compass direction "
            "from east to west, every angle from flat to steep.\n\n"
            "**Step 2 — find the direction.** A panel facing east has its best hours in the morning; a "
            "panel facing west, in the afternoon. We compare each virtual panel's daily curve with the "
            "real meter readings (only on clear moments, with the sun properly up and the inverter not "
            "maxed out) and keep the direction that matches best. All three sites match a west-of-south "
            "direction.\n\n"
            "**Step 3 — find the tilt.** Around noon the winter sun hangs low (16° above the horizon "
            "here) and the summer sun high (62°). A *steep* panel loves winter noons and wastes summer "
            "noons; a *flat* panel is the opposite. So we look only at clear middays across the seasons "
            "and ask which angle explains the winter-vs-summer pattern. The reactor clearly behaves "
            "like a nearly-flat panel; the houses behave like steep ones.\n\n"
            "**Step 4 — try two directions at once.** Some roofs have panels both ways. So we also "
            "mixed *two* virtual panels in every possible combination and let the math choose the "
            "blend. If a roof really had two clearly different faces, the mix would match the meter "
            "better than any single panel. It never did, which is itself an answer: the two faces of "
            "House 1 and the reactor are so alike to the sun that only their combination is visible.\n\n"
            "**Honesty notes.** Each number comes with a range (everything that fits nearly as well). "
            "The direction is solid for all sites. The tilt is very solid for the reactor and rougher "
            "for the houses, because their steep-panel signal competes with morning haze effects, so "
            "for the houses read the tilt as 'steep, roughly 45-60°' rather than an exact figure."
        )
        st.markdown("**Cross-check with a much simpler method** — just the balance point of the "
                    "average production day (morning-heavy = east, evening-heavy = west), no simulation "
                    "at all. It points the same way, which is reassuring:")
        orient = get_orientations()
        st.dataframe(orient[["site", "peak_hour_utc", "centre_of_mass_hour", "azimuth_deg", "facing"]],
                     column_config={
                         "site": st.column_config.TextColumn("Site"),
                         "peak_hour_utc": st.column_config.NumberColumn(
                             "Peak hour (UTC)", help="Hour of day with the highest average output."),
                         "centre_of_mass_hour": st.column_config.NumberColumn(
                             "Balance point (h)", help="Centre of mass of the average production day. "
                                                       "Solar noon here is ~11:47 UTC; later = more west."),
                         "azimuth_deg": st.column_config.NumberColumn(
                             "Azimuth (°)", help="Balance-point offset from solar noon mapped to a compass "
                                                 "angle (15° per hour). Cruder than the fit above; it "
                                                 "understates how far west a panel faces."),
                         "facing": st.column_config.TextColumn("Facing"),
                     },
                     width="stretch", hide_index=True)


def page_predict():
    st.title("Try the prediction")
    st.markdown(
        "Compact models (irradiance + hour of day + temperature), one per site. "
        "Adjust the sliders to see how conditions shape the predicted daily output curve."
    )
    c1, c2, c3 = st.columns(3)
    irr = c1.slider("Irradiance (W/m²)", 0, 900, 500, step=25, key="pred_irr")
    hour = c2.slider("Hour of day (UTC)", 0, 23, 12, key="pred_hour")
    temp = c3.slider("Temperature (°C)", -5, 35, 18, key="pred_temp")

    hours = list(range(24))
    fig = go.Figure()
    point_values = {}
    for name in SITES:
        preds = [predict_compact(name, irr, h, temp) for h in hours]
        point_values[name] = preds[hour]
        fig.add_trace(go.Scatter(
            x=hours, y=preds, name=SITE_INFO[name]["label"],
            mode="lines", line=dict(color=SITE_COLORS[name], width=2.5),
            hovertemplate="%{y:.3f} kWh<extra>" + SITE_INFO[name]["label"] + "</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[hour], y=[preds[hour]], mode="markers",
            marker=dict(color=SITE_COLORS[name], size=10, line=dict(width=2, color="white")),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_vline(x=hour, line_dash="dot", line_color="rgba(100,100,100,0.4)")
    fig.update_layout(
        xaxis=dict(title="Hour of day (UTC)", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="Predicted output (kWh / 15 min)",
        title=f"Predicted daily curve — {irr} W/m², {temp} °C",
        height=460,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    metric_cols = st.columns(len(SITES))
    for col, name in zip(metric_cols, SITES):
        pred = point_values[name]
        col.metric(
            SITE_INFO[name]["label"],
            f"{pred:.3f} kWh / 15 min",
            help=f"≈ {pred * 4:.2f} kW instantaneous. {SITES[name]['kwp']} kWp installed.",
        )
    st.caption("Times are UTC — solar noon at this longitude is ~11:47 UTC. The peak shifts with orientation: the reactor's flat east-west layout centres near noon, the steep west-south-west houses peak later.")

    st.divider()
    st.subheader("Irradiance sweep")
    st.caption("Output vs sun strength at the hour and temperature set above. Watch each site's curve flatten where its inverter hits its limit.")
    irr_range = list(range(0, 925, 25))
    fig_sweep = go.Figure()
    for name in SITES:
        sweep = predict_sweep(name, hour, temp)
        fig_sweep.add_trace(go.Scatter(
            x=irr_range, y=sweep, name=SITE_INFO[name]["label"],
            mode="lines", line=dict(color=SITE_COLORS[name], width=2.5),
            hovertemplate="%{x} W/m² → %{y:.3f} kWh<extra>" + SITE_INFO[name]["label"] + "</extra>",
        ))
    fig_sweep.add_vline(x=irr, line_dash="dot", line_color="rgba(100,100,100,0.4)")
    fig_sweep.update_layout(
        xaxis_title="Irradiance (W/m²)",
        yaxis_title="Predicted output (kWh / 15 min)",
        title=f"Output vs irradiance — {hour}:00 UTC, {temp} °C",
        height=400,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_sweep, width="stretch")

    st.divider()
    st.subheader("Simulated full day")
    st.caption("Pick a sky condition to generate a realistic irradiance profile. Temperature is taken from the slider above.")
    day_type = st.selectbox("Sky condition", list(DAY_PROFILES), key="pred_day_type")
    irr_prof = DAY_PROFILES[day_type]
    hours_x = list(range(24))

    fig_day = go.Figure()
    fig_day.add_trace(go.Scatter(
        x=hours_x, y=list(irr_prof), name="Irradiance profile",
        mode="lines", fill="tozeroy",
        line=dict(color="rgba(255,190,30,0.7)", width=1.5),
        fillcolor="rgba(255,190,30,0.07)",
        yaxis="y2",
        hovertemplate="%{y:.0f} W/m²<extra>Irradiance</extra>",
    ))
    daily_totals = {}
    for name in SITES:
        preds = predict_day(name, irr_prof, temp)
        daily_totals[name] = sum(preds) * 4
        fig_day.add_trace(go.Scatter(
            x=hours_x, y=preds, name=SITE_INFO[name]["label"],
            mode="lines", line=dict(color=SITE_COLORS[name], width=2.5),
            hovertemplate="%{y:.3f} kWh<extra>" + SITE_INFO[name]["label"] + "</extra>",
        ))
    fig_day.update_layout(
        xaxis=dict(title="Hour of day (UTC)", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Predicted output (kWh / 15 min)"),
        yaxis2=dict(title="Irradiance (W/m²)", overlaying="y", side="right", showgrid=False, range=[0, 1050]),
        title=f"Simulated {day_type.lower()} — {temp} °C",
        height=440,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_day, width="stretch")

    fig_totals = go.Figure()
    for name in SITES:
        fig_totals.add_trace(go.Bar(
            x=[SITE_INFO[name]["label"]], y=[daily_totals[name]],
            marker_color=SITE_COLORS[name], name=SITE_INFO[name]["label"],
            text=[f"{daily_totals[name]:.1f} kWh"], textposition="outside",
        ))
    fig_totals.update_layout(
        yaxis_title="Estimated daily output (kWh)",
        title="Estimated total for the day",
        showlegend=False,
        height=340,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_totals, width="stretch")


def page_today():
    today_str = date.today().isoformat()
    st.title(f"Today — {today_str}")
    st.markdown(
        "Fetch today's weather from Open-Meteo (the same source as the training data) and run each "
        "site's **best model** (see the Models page) on the full weather picture. If the dataset "
        "already contains today's actual output, it is overlaid as a dashed line."
    )

    if st.button("Fetch today's weather", type="primary"):
        with st.spinner("Fetching from Open-Meteo..."):
            try:
                st.session_state["today_fc"] = fetch_forecast(1)
                st.session_state["today_date"] = today_str
            except Exception as exc:
                st.error(f"Could not fetch weather data: {exc}")

    if "today_fc" not in st.session_state:
        _fetch_empty_state("Pull today's hourly weather live from Open-Meteo, then run each "
                           "site's best model to predict the full production day.")
        return

    hourly = st.session_state["today_fc"]["hourly"]
    fetched_date = st.session_state["today_date"]
    hours_x = list(range(24))
    irr_profile = hourly["shortwave_radiation (W/m²)"].tolist()

    avg_temp = hourly["temperature_2m (°C)"].mean()
    peak_irr = max(irr_profile)
    c1, c2 = st.columns(2)
    c1.metric("Peak irradiance", f"{peak_irr:.0f} W/m²",
              help="Highest hourly irradiance value today (W/m²). A clear summer day in Belgium peaks around 800-900 W/m²; an overcast day stays below 150 W/m².")
    c2.metric("Avg temperature", f"{avg_temp:.1f} °C",
              help="Average of today's 24 hourly temperature readings. Higher temperatures slightly reduce panel efficiency — roughly 0.4% per °C above 25 °C for typical silicon panels.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours_x, y=list(irr_profile), name="Irradiance",
        mode="lines", fill="tozeroy",
        line=dict(color="rgba(255,190,30,0.7)", width=1.5),
        fillcolor="rgba(255,190,30,0.07)",
        yaxis="y2",
        hovertemplate="%{y:.0f} W/m²<extra>Irradiance</extra>",
    ))

    daily_totals = {}
    has_actual = False
    for name in SITES:
        preds = predict_forecast(name, hourly)
        daily_totals[name] = float(preds.sum())
        pred_hours = preds.index.hour + preds.index.minute / 60
        fig.add_trace(go.Scatter(
            x=pred_hours, y=preds.values, name=SITE_INFO[name]["label"],
            mode="lines", line=dict(color=SITE_COLORS[name], width=2.5),
            hovertemplate="%{y:.3f} kWh<extra>" + SITE_INFO[name]["label"] + " predicted</extra>",
        ))

        try:
            joined = get_joined(name)
            mask = joined.index.strftime("%Y-%m-%d") == fetched_date
            actual_today = joined.loc[mask, "energy"]
            if not actual_today.empty:
                actual_hours = actual_today.index.hour + actual_today.index.minute / 60
                fig.add_trace(go.Scatter(
                    x=actual_hours, y=actual_today.values,
                    name=f"{SITE_INFO[name]['label']} actual",
                    mode="lines", line=dict(color=SITE_COLORS[name], width=2, dash="dash"),
                    hovertemplate="%{y:.3f} kWh<extra>" + SITE_INFO[name]["label"] + " actual</extra>",
                ))
                has_actual = True
        except Exception:
            pass

    fig.update_layout(
        xaxis=dict(title="Hour of day (UTC)", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Output (kWh / 15 min)"),
        yaxis2=dict(title="Irradiance (W/m²)", overlaying="y", side="right", showgrid=False, range=[0, 1050]),
        title=f"Predicted output for {fetched_date}" + (" — solid = predicted, dashed = actual" if has_actual else ""),
        height=480,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    if not has_actual:
        st.caption("No actual output data found for today in the dataset — showing prediction only.")

    total_cols = st.columns(len(SITES))
    for col, name in zip(total_cols, SITES):
        col.metric(
            f"{SITE_DOT[name]} {SITE_INFO[name]['label']}",
            f"{daily_totals[name]:.1f} kWh",
            help=f"Estimated total for today. {SITES[name]['kwp']} kWp installed.",
        )
    st.caption(f"Estimated day totals. {OPEN_METEO_CREDIT} Fetched for {fetched_date}.")


def page_this_week():
    st.title("This week")
    st.markdown(
        "Fetch the 7-day weather forecast from Open-Meteo and run each site's **best model** "
        "(see the Models page) on the full weather picture. Shows the estimated daily total per "
        "day and the predicted output curve across the week."
    )

    if st.button("Fetch 7-day forecast", type="primary"):
        with st.spinner("Fetching from Open-Meteo..."):
            try:
                st.session_state["week"] = fetch_forecast(7)
            except Exception as exc:
                st.error(f"Could not fetch weather data: {exc}")

    if "week" not in st.session_state:
        _fetch_empty_state("Pull the 7-day forecast live from Open-Meteo, then run each site's "
                           "best model for the estimated production of the week ahead.")
        return

    week = st.session_state["week"]
    dates = week["dates"]
    hourly = week["hourly"]

    # weather outlook: one card per day, before the graphs
    st.subheader("Weather outlook")
    day_cols = st.columns(len(dates))
    for col, i in zip(day_cols, range(len(dates))):
        icon, label = weather_icon(week["code"][i])
        weekday = pd.Timestamp(dates[i]).strftime("%a")
        col.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-weight:600'>{weekday}</div>"
            f"<div style='font-size:2rem;line-height:2.4rem'>{icon}</div>"
            f"<div style='font-size:0.8rem;color:#666'>{label}</div>"
            f"<div style='font-size:0.85rem'>{week['temp_max'][i]:.0f}° / {week['temp_min'][i]:.0f}°</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # one 15-min prediction series per site across the whole window
    week_preds = {name: predict_forecast(name, hourly) for name in SITES}
    daily_totals = {}
    for name in SITES:
        per_day = week_preds[name].groupby(week_preds[name].index.strftime("%Y-%m-%d")).sum()
        daily_totals[name] = [float(per_day.get(d, 0.0)) for d in dates]

    week_total = {name: sum(vals) for name, vals in daily_totals.items()}
    cols = st.columns(len(SITES))
    for col, name in zip(cols, SITES):
        col.metric(
            f"{SITE_INFO[name]['label']} — 7-day total",
            f"{week_total[name]:,.0f} kWh",
            help=f"Sum of the estimated daily output over the {len(dates)} forecast days. {SITES[name]['kwp']} kWp installed.",
        )

    st.subheader("Estimated daily output",
                 help="Predicted total energy per day for each site, from the forecast irradiance and temperature. Weekends and weekdays are not distinguished; only the weather matters.")
    fig_daily = go.Figure()
    for name in SITES:
        fig_daily.add_trace(go.Bar(
            x=dates, y=daily_totals[name], name=SITE_INFO[name]["label"],
            marker_color=SITE_COLORS[name],
            hovertemplate="%{x}<br>%{y:.1f} kWh<extra>" + SITE_INFO[name]["label"] + "</extra>",
        ))
    fig_daily.update_layout(
        barmode="group", height=420, yaxis_title="Estimated output (kWh)",
        xaxis_title="Date", title="Estimated daily output this week", **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_daily, width="stretch")

    st.subheader("Output across the week",
                 help="The full predicted output curve, per 15 minutes, over all forecast days. Each daily bump is one production day; overcast days stay low.")
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=hourly.index, y=hourly["shortwave_radiation (W/m²)"], name="Irradiance",
        mode="lines", fill="tozeroy",
        line=dict(color="rgba(255,190,30,0.7)", width=1),
        fillcolor="rgba(255,190,30,0.07)", yaxis="y2",
        hovertemplate="%{y:.0f} W/m²<extra>Irradiance</extra>",
    ))
    for name in SITES:
        preds = week_preds[name]
        fig_hourly.add_trace(go.Scatter(
            x=preds.index, y=preds.values, name=SITE_INFO[name]["label"],
            mode="lines", line=dict(color=SITE_COLORS[name], width=2),
            hovertemplate="%{y:.3f} kWh<extra>" + SITE_INFO[name]["label"] + "</extra>",
        ))
    fig_hourly.update_layout(
        xaxis=dict(title="Day"),
        yaxis=dict(title="Output (kWh / 15 min)"),
        yaxis2=dict(title="Irradiance (W/m²)", overlaying="y", side="right", showgrid=False, range=[0, 1050]),
        title="Predicted output this week", height=460, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_hourly, width="stretch")
    st.caption(f"{OPEN_METEO_CREDIT} Forecast for {dates[0]} to {dates[-1]}.")


def page_replay():
    st.title("Replay a day")
    st.markdown(
        "Pick any day in the dataset. Each site's **best model** predicts that day from its recorded "
        "weather, so you can compare the prediction against what the panels actually produced."
    )
    day = st.date_input("Day", value=MAX_DATE, min_value=MIN_DATE, max_value=MAX_DATE, key="replay_day")

    fig = go.Figure()
    rows, irr_added = [], False
    for name in SITES:
        data = get_day_backtest(name, day)
        if data is None:
            st.info(f"No output data for {SITE_INFO[name]['label']} on {day}.", icon="ℹ️")
            continue
        pred, actual = data["pred"], data["actual"]
        hours = pred.index.hour + pred.index.minute / 60

        if not irr_added:  # weather is near-identical across the sites, one curve suffices
            fig.add_trace(go.Scatter(
                x=hours, y=data["irr"].values, name="Irradiance",
                mode="lines", fill="tozeroy", yaxis="y2",
                line=dict(color="rgba(255,190,30,0.7)", width=1.5),
                fillcolor="rgba(255,190,30,0.07)",
                hovertemplate="%{y:.0f} W/m²<extra>Irradiance</extra>",
            ))
            irr_added = True
        fig.add_trace(go.Scatter(
            x=hours, y=pred.values, name=SITE_INFO[name]["label"], legendgroup=name,
            mode="lines", line=dict(color=SITE_COLORS[name], width=2.5),
            hovertemplate="%{y:.3f} kWh<extra>" + SITE_INFO[name]["label"] + " predicted</extra>",
        ))
        known = actual.notna()
        fig.add_trace(go.Scatter(
            x=hours[known], y=actual[known].values, name=f"{SITE_INFO[name]['label']} actual",
            legendgroup=name, showlegend=False,
            mode="lines", line=dict(color=SITE_COLORS[name], width=2, dash="dash"),
            hovertemplate="%{y:.3f} kWh<extra>" + SITE_INFO[name]["label"] + " actual</extra>",
        ))

        # totals over the quarters with a known actual, so logging gaps stay fair
        act_total = actual[known].sum()
        pred_total = pred[known].sum()
        error = pred_total - act_total
        rows.append({
            "site": f"{SITE_DOT[name]} {SITE_INFO[name]['label']}",
            "actual (kWh)": round(act_total, 2),
            "predicted (kWh)": round(pred_total, 2),
            "error (kWh)": round(error, 2),
            "error (%)": round(error / act_total * 100, 1) if act_total else None,
            "missing quarters": int((~known).sum()),
        })

    if not rows:
        return
    fig.update_layout(
        xaxis=dict(title="Hour of day (UTC)", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Output (kWh / 15 min)"),
        yaxis2=dict(title="Irradiance (W/m²)", overlaying="y", side="right", showgrid=False, range=[0, 1050]),
        title=f"Predicted vs actual output for {day} — solid = predicted, dashed = actual",
        height=480, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Day totals",
                 help="Totals are summed over the quarters where the site actually reported a reading "
                      "(plus nights, which count as 0), so logging gaps do not inflate the prediction.")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Note: the models are trained on the full history, including this day. This shows how well "
        "the model reproduces a day from its weather, not a blind forecast; for honest held-out "
        "accuracy see the Models page."
    )
