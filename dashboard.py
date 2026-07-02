"""
Interactive Streamlit dashboard for the PV / weather analysis.

Multipage app (sidebar navigation), grouped into:
- Start:    Overview, Data guide
- Sites:    House 1, House 2, Reactor, Compare
- Analysis: Time of day, Weather, Anomalies
- Machine learning: Models, Predict

Run: streamlit run dashboard.py
"""

import math
import sys
from datetime import date
from pathlib import Path

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import SITES, daily_energy
from weather_correlation import joined_data, RAD_COL, TEMP_COL
from anomalies import daily_yield_ratio, flag_anomalies

# the machine-learning/ dir has a hyphen (not importable as a package), so add it
# to the path and import its modules directly
ML_DIR = Path(__file__).resolve().parent / "machine-learning"
sys.path.insert(0, str(ML_DIR))
from orientation import estimate_azimuth  # noqa: E402

SITE_COLORS = {"house1": "#1f77b4", "house2": "#ff7f0e", "reactor": "#2ca02c"}
SITE_FILL = {"house1": "rgba(31,119,180,0.12)", "house2": "rgba(255,127,14,0.12)", "reactor": "rgba(44,160,44,0.12)"}
SITE_DOT = {"house1": "🔵", "house2": "🟠", "reactor": "🟢"}  # matches SITE_COLORS for text labels

# installation metadata, used by the site pages and Compare
SITE_INFO = {
    "house1": {"label": "House 1", "inverter_kw": 4.0, "dcac": 1.56,
               "arrays": "3 arrays (4 + 1.5 + 0.75 kWp), 2 directions"},
    "house2": {"label": "House 2", "inverter_kw": 2.2, "dcac": 1.09,
               "arrays": "1 array (2.4 kWp), 1 direction"},
    "reactor": {"label": "Reactor", "inverter_kw": 22.0, "dcac": 1.49,
                "arrays": "2 arrays (16.35 + 16.35 kWp)"},
}

# WMO weather codes present in this dataset, grouped for readability
WMO_LABELS = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Rain", 65: "Heavy rain",
    73: "Moderate snow", 75: "Heavy snow",
}

RESAMPLE_RULES = {"Day": "D", "Week": "W", "Month": "MS"}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(t=60, b=40),
)


@st.cache_data
def get_daily_energy(site_name: str) -> pd.Series:
    return daily_energy(site_name)


ALL_MONTHS = list(range(1, 13))
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_data
def get_coverage(site_name: str) -> dict:
    """Data span and which calendar months this site has ever recorded, so the
    site page can flag that predictions for unseen months are extrapolation."""
    daily = get_daily_energy(site_name).dropna()
    seen = sorted(set(daily.index.month))
    missing = [m for m in ALL_MONTHS if m not in seen]
    return {
        "start": daily.index.min().date(),
        "end": daily.index.max().date(),
        "days": len(daily),
        "seen_months": seen,
        "missing_months": missing,
    }


@st.cache_data
def get_joined(site_name: str) -> pd.DataFrame:
    return joined_data(site_name)


@st.cache_data
def get_yield_ratio(site_name: str, min_rad: float) -> pd.DataFrame:
    return daily_yield_ratio(site_name, min_rad)


@st.cache_data
def get_anomalies(site_name: str, z: float, min_rad: float) -> pd.DataFrame:
    return flag_anomalies(site_name, z, min_rad)


@st.cache_data
def get_ml_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ML_DIR / name)


@st.cache_data
def get_orientations() -> pd.DataFrame:
    return pd.DataFrame([estimate_azimuth(s) for s in SITES])


@st.cache_data
def get_clipping_curve() -> pd.DataFrame:
    """Max output per irradiance bin per site, normalized to each site's own peak,
    so the inverter clipping plateau is visible on one shared 0-1 axis."""
    rows = []
    for name in SITES:
        df = get_joined(name)
        df = df[df[RAD_COL] > 25]
        bins = pd.cut(df[RAD_COL], range(0, 1001, 100))
        peak = df.groupby(bins, observed=True)["energy"].max()
        peak = peak / peak.max()  # normalize to this site's max
        for interval, val in peak.items():
            rows.append({"site": name, "irradiance": interval.mid, "rel_max_output": val})
    return pd.DataFrame(rows)


@st.cache_resource
def _compact_model(site_name: str):
    """Small RandomForest on irradiance + hour (sin/cos) + temperature, for the
    interactive predict widget. Cached so the sliders stay responsive.
    build_features fills unreported night quarters with 0, so this model has
    learned that nights produce nothing."""
    from sklearn.ensemble import RandomForestRegressor
    from features import build_features

    X, y = build_features(site_name, "quarterly")
    feats = X[["shortwave_radiation (W/m²)", "hour_sin", "hour_cos", "temperature_2m (°C)"]]
    model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(feats, y)
    return model


def _clip_pred(site_name: str, value: float) -> float:
    """Clip a 15-min prediction to what is physically possible: never negative,
    never more than the inverter passes in a quarter (kW x 0.25 = kWh)."""
    cap = SITE_INFO[site_name]["inverter_kw"] * 0.25
    return max(0.0, min(cap, float(value)))


def predict_compact(site_name: str, irradiance: float, hour: int, temp: float) -> float:
    import numpy as np

    radians = 2 * np.pi * hour / 24
    row = pd.DataFrame([{
        "shortwave_radiation (W/m²)": irradiance,
        "hour_sin": np.sin(radians), "hour_cos": np.cos(radians),
        "temperature_2m (°C)": temp,
    }])
    return _clip_pred(site_name, _compact_model(site_name).predict(row)[0])


def _irr_profile(peak_wm2: float, peak_hour: float = 12.5, sigma: float = 3.5) -> tuple:
    """Gaussian irradiance profile across 24 hours, zeroed outside daylight."""
    profile = []
    for h in range(24):
        if h < 5 or h > 21:
            profile.append(0.0)
        else:
            profile.append(max(0.0, peak_wm2 * math.exp(-0.5 * ((h - peak_hour) / sigma) ** 2)))
    return tuple(profile)


DAY_PROFILES = {
    "Clear summer": _irr_profile(850),
    "Partly cloudy": _irr_profile(400, sigma=3.0),
    "Overcast": _irr_profile(80, sigma=4.5),
}


@st.cache_data
def predict_sweep(site_name: str, hour: int, temp: float) -> list:
    """Predicted output across all irradiance levels (0-900 W/m²) for one site."""
    import numpy as np
    model = _compact_model(site_name)
    irr_range = list(range(0, 925, 25))
    radians = 2 * math.pi * hour / 24
    df = pd.DataFrame({
        "shortwave_radiation (W/m²)": irr_range,
        "hour_sin": [math.sin(radians)] * len(irr_range),
        "hour_cos": [math.cos(radians)] * len(irr_range),
        "temperature_2m (°C)": [temp] * len(irr_range),
    })
    return [_clip_pred(site_name, p) for p in model.predict(df)]


# Open-Meteo hourly variable -> training column name. The forecast API is the
# same source (and units) as the training weather, so the best models can run
# on it directly. Covers the fair feature set plus terrestrial radiation for
# the clear-sky index; the leaky tilted irradiance is not fetched.
FORECAST_VARS = {
    "temperature_2m": "temperature_2m (°C)",
    "relative_humidity_2m": "relative_humidity_2m (%)",
    "dew_point_2m": "dew_point_2m (°C)",
    "apparent_temperature": "apparent_temperature (°C)",
    "shortwave_radiation": "shortwave_radiation (W/m²)",
    "direct_radiation": "direct_radiation (W/m²)",
    "diffuse_radiation": "diffuse_radiation (W/m²)",
    "direct_normal_irradiance": "direct_normal_irradiance (W/m²)",
    "terrestrial_radiation": "terrestrial_radiation (W/m²)",
    "weather_code": "weather_code (wmo code)",
    "wind_speed_10m": "wind_speed_10m (km/h)",
    "visibility": "visibility (m)",
    "is_day": "is_day ()",
}
# step values: repeat instead of interpolate when moving to the 15-min grid
FORECAST_STEP_VARS = ["weather_code (wmo code)", "is_day ()"]


def fetch_forecast(days: int) -> dict:
    """Fetch `days` days of the full hourly weather set from Open-Meteo (UTC,
    same variables and units as the training data), plus the daily summary
    (weather code, min/max temperature) for the forecast cards."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": 50.908,
            "longitude": 3.248,
            "hourly": ",".join(FORECAST_VARS),
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "timezone": "GMT",
            "forecast_days": days,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    hourly = pd.DataFrame(
        {col: data["hourly"][var] for var, col in FORECAST_VARS.items()},
        index=pd.to_datetime(data["hourly"]["time"]),
    ).astype(float)
    daily = data["daily"]
    n = min(days, len(daily["time"]))
    return {
        "hourly": hourly,
        "dates": daily["time"][:n],
        "code": [int(c) for c in daily["weather_code"][:n]],
        "temp_max": [float(v) for v in daily["temperature_2m_max"][:n]],
        "temp_min": [float(v) for v in daily["temperature_2m_min"][:n]],
    }


def _to_quarter_grid(hourly: pd.DataFrame) -> pd.DataFrame:
    """Resample an hourly forecast onto the 15-min grid the models were trained
    on: smooth variables are time-interpolated, step variables repeated."""
    grid = pd.date_range(hourly.index.min(),
                         hourly.index.max() + pd.Timedelta(minutes=45), freq="15min")
    df = hourly.reindex(grid)
    smooth = [c for c in df.columns if c not in FORECAST_STEP_VARS]
    df[smooth] = df[smooth].interpolate(method="time", limit_direction="both")
    df[FORECAST_STEP_VARS] = df[FORECAST_STEP_VARS].ffill().bfill()
    return df


@st.cache_resource
def _forecast_model(site_name: str):
    """The site's best model for the Today/This week pages: the model type that
    wins the ts_cv ranking in results.csv, trained on the fair_lag features
    (the winning setup for all sites). Trained here rather than loaded from
    models/ because those artifacts are gitignored."""
    from features import build_features
    from train import make_model

    results = get_ml_csv("results.csv")
    ranked = results[(results["resolution"] == "quarterly") & (results["method"] == "ts_cv")
                     & (results["model"] != "physics") & (results["site"] == site_name)]
    best = ranked.sort_values("r2").iloc[-1]
    X, y = build_features(site_name, "quarterly", feature_set="fair", add_lags=True)
    return make_model(best["model"]).fit(X, y)


def predict_forecast(site_name: str, hourly: pd.DataFrame) -> pd.Series:
    """15-min predictions (kWh per quarter) over a forecast window, from the
    site's best model. The first two hours fall in the lag warm-up window and
    count as 0, which is exact for a window starting at midnight UTC."""
    from features import build_forecast_features

    grid = _to_quarter_grid(hourly)
    X = build_forecast_features(grid)
    pred = _forecast_model(site_name).predict(X)
    pred = pd.Series([_clip_pred(site_name, p) for p in pred], index=X.index)
    return pred.reindex(grid.index, fill_value=0.0)


@st.cache_data
def get_day_backtest(site_name: str, day) -> dict | None:
    """Predicted vs actual 15-min series for one historical day, from the site's
    best model on that day's recorded weather. None if the site has no output
    data on that day."""
    from features import load_clean, build_forecast_features, TARGET

    df = load_clean(site_name, "quarterly")
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    day_rows = df.loc[day_start:day_end]
    # require at least one real reading; night fill alone would fake an all-zero day
    if day_rows.empty or day_rows[TARGET].notna().sum() == 0:
        return None

    actual = day_rows[TARGET].copy()
    night_gap = (day_rows["is_day ()"] == 0) & actual.isna()
    actual[night_gap] = 0.0  # unreported night quarters are zero, same rule as training

    # 2-hour warm-up before midnight so the lag features are defined at 00:00
    window = df.loc[day_start - pd.Timedelta(hours=2):day_end]
    X = build_forecast_features(window.drop(columns=[TARGET]))
    pred = _forecast_model(site_name).predict(X)
    pred = pd.Series([_clip_pred(site_name, p) for p in pred], index=X.index)
    pred = pred.reindex(day_rows.index, fill_value=0.0)

    return {"actual": actual, "pred": pred,
            "irr": day_rows["shortwave_radiation (W/m²)"]}


# WMO code -> (emoji, short label) for the forecast cards
WMO_ICONS = {
    0: ("☀️", "Clear"), 1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"), 3: ("☁️", "Overcast"),
    45: ("🌫️", "Fog"), 48: ("🌫️", "Rime fog"),
    51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Drizzle"), 55: ("🌧️", "Dense drizzle"),
    61: ("🌦️", "Slight rain"), 63: ("🌧️", "Rain"), 65: ("🌧️", "Heavy rain"),
    71: ("🌨️", "Slight snow"), 73: ("🌨️", "Snow"), 75: ("❄️", "Heavy snow"),
    80: ("🌦️", "Showers"), 81: ("🌧️", "Showers"), 82: ("⛈️", "Violent showers"),
    95: ("⛈️", "Thunderstorm"),
}


def weather_icon(code: int) -> tuple:
    """Emoji and label for a WMO code, falling back to the nearest known bucket."""
    return WMO_ICONS.get(code, ("🌡️", WMO_LABELS.get(code, f"Code {code}")))


def _fetch_empty_state(message: str):
    """Friendly placeholder shown on the forecast pages before the first fetch."""
    with st.container(border=True):
        st.markdown(
            "<div style='text-align:center;padding:26px 12px'>"
            "<div style='font-size:34px'>🛰️</div>"
            "<div style='font-size:16px;font-weight:600;margin-top:6px'>No weather loaded yet</div>"
            f"<div style='color:#9ca3af;font-size:13px;margin-top:4px;'>{message}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


OPEN_METEO_CREDIT = ("Weather data by [Open-Meteo](https://open-meteo.com) "
                     "([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), lat=50.908 lon=3.248.")


@st.cache_data
def predict_day(site_name: str, irr_profile: tuple, temp: float) -> list:
    """Predicted output for each of 24 hours given an irradiance profile."""
    model = _compact_model(site_name)
    hours = list(range(24))
    df = pd.DataFrame({
        "shortwave_radiation (W/m²)": list(irr_profile),
        "hour_sin": [math.sin(2 * math.pi * h / 24) for h in hours],
        "hour_cos": [math.cos(2 * math.pi * h / 24) for h in hours],
        "temperature_2m (°C)": [temp] * 24,
    })
    return [_clip_pred(site_name, p) for p in model.predict(df)]


@st.cache_data
def get_daily_profile(site_name: str) -> pd.Series:
    """Average output (kWh) per hour of day over sunny days, for the time-of-day
    and orientation views."""
    df = get_joined(site_name)
    daily_rad = df[RAD_COL].resample("D").sum()
    sunny = daily_rad[daily_rad > daily_rad.quantile(0.75)].index.normalize()
    lit = df[df.index.normalize().isin(sunny)]
    return lit.groupby(lit.index.hour)["energy"].mean()


def date_bounds():
    starts, ends = [], []
    for name in SITES:
        idx = get_daily_energy(name).index
        starts.append(idx.min())
        ends.append(idx.max())
    return min(starts).date(), max(ends).date()


def period_delta(series: pd.Series, date_range):
    """Return a delta string like '+12%' comparing the selected range to the equal-length period before it."""
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    length = end - start
    prev_start = start - length
    current = series[in_range(series.index, date_range)].sum()
    previous = series[(series.index >= prev_start) & (series.index < start)].sum()
    if previous == 0 or pd.isna(previous):
        return None
    pct = (current - previous) / previous * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


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


def in_range(index, date_range) -> pd.Series:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    return (index >= start) & (index < end)


# === PAGES ==================================================================

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
        if resolution == "Day" and len(agg) >= 7:
            rolling = agg.rolling(7, center=True, min_periods=4).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values, name=f"{name} 7d avg",
                mode="lines", line=dict(color=color, width=2, dash="dash"),
                showlegend=False, legendgroup=name, hoverinfo="skip",
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
    orient = get_orientations().set_index("site").loc[name]
    c4.metric("Orientation", orient["facing"],
              help=f"Estimated from the daily output profile. Azimuth {orient['azimuth_deg']}° — 180° is due south, below 180° is east of south, above 180° is west of south.")
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
        f"array facing **{orient['facing']}** (morning peak = east, midday = south, evening = west)."
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
        orient = get_orientations().set_index("site")
        rows = []
        for name in SITES:
            daily = get_daily_energy(name)[lambda s: in_range(s.index, date_range)]
            rows.append({
                "site": SITE_INFO[name]["label"], "kWp": SITES[name]["kwp"],
                "inverter (kW)": SITE_INFO[name]["inverter_kw"], "DC/AC": SITE_INFO[name]["dcac"],
                "orientation": orient.loc[name, "facing"],
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
    st.caption("Each curve is scaled to its own peak, so only the timing differs. The reactor peaks "
               "earliest (near solar noon, due south); the houses peak later (south-west).")


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
    st.caption("Averaged over sunny days. Times are UTC; local solar noon is around 12-13h UTC.")


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
        "margin at the near-south Reactor. The baseline itself improved with the night fill: predicting "
        "zero at zero irradiance is trivially right, which is exactly why the honest comparison keeps it in."
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

    st.subheader("Inferred panel orientation")
    st.markdown("Estimated from the daily output profile: east peaks in the morning, west in the evening, south at noon.")
    orient = get_orientations()
    st.dataframe(orient[["site", "peak_hour_utc", "centre_of_mass_hour", "azimuth_deg", "facing"]],
                 width="stretch", hide_index=True)
    st.caption("180° = due south, >180° = west of south. Reactor is near-south; the houses lean south-west.")


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
    st.caption("Times are UTC — local solar noon is around 12–13h. The peak shifts with orientation: reactor peaks earlier (south), houses later (south-west).")

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


# --- navigation -------------------------------------------------------------

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
nav.run()
