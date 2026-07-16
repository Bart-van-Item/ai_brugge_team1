import math

import requests
import pandas as pd
import streamlit as st

from dashboard_app.config import SITE_INFO, WMO_LABELS
from dashboard_app.data import get_ml_csv


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
