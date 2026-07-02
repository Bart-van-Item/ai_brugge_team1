"""
Feature engineering for the PV forecasting models.

Builds an (X, y) feature matrix from a site's cleaned quarterly or daily CSV:
- X: weather columns + clear-sky index + cyclical time features
- y: energy_kwh

Target handling (quarterly): the houses' PV loggers do not report at night, so
those quarters are missing rather than zero. A panel at night produces exactly
0, so missing night quarters (is_day == 0) are filled with 0 instead of being
dropped. This roughly doubles the usable rows and teaches the model that nights
are zero, which the forecast pages rely on. Missing daytime quarters stay NaN:
for house1 a large share of them are real logging outages at productive hours,
so zero-filling those would inject false zeros.

The clear-sky index (shortwave / terrestrial radiation) tells the model "how
cloudy" independent of season and hour, a small but consistent gain.

Time features are encoded as sin/cos pairs so the model sees that e.g. hour 23
and hour 0 are adjacent, and December and January are adjacent.

Optionally adds lag and rolling features (past irradiance) via add_lags=True.
PV output is strongly autocorrelated, so recent irradiance is a strong predictor.
Lags are built on the full continuous 15-min grid, before rows with a missing
energy reading are dropped, so a positional shift is always a real time shift.

This module has no side effects; train.py imports build_features() from it.
"""

from pathlib import Path
import numpy as np
import pandas as pd

CLEAN = Path(__file__).resolve().parent.parent / "data" / "clean"

TARGET = "energy_kwh"

# weather columns used as predictors (everything in the clean file except target)
WEATHER_FEATURES = [
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "dew_point_2m (°C)",
    "apparent_temperature (°C)",
    "shortwave_radiation (W/m²)",
    "direct_radiation (W/m²)",
    "diffuse_radiation (W/m²)",
    "direct_normal_irradiance (W/m²)",
    "global_tilted_irradiance (W/m²)",
    "terrestrial_radiation (W/m²)",
    "weather_code (wmo code)",
    "wind_speed_10m (km/h)",
    "visibility (m)",
    "is_day ()",
]

# Leaky/unfair predictor: global_tilted_irradiance is irradiance in the panel
# plane, which is orientation-specific and almost the answer itself. The "fair"
# feature set drops it so the model has to learn output from horizontal weather.
LEAKY_FEATURE = "global_tilted_irradiance (W/m²)"

# Lean, physically motivated set: one horizontal irradiance measure + temperature.
# This avoids six overlapping irradiance columns the model spreads weight across.
LEAN_WEATHER = [
    "shortwave_radiation (W/m²)",
    "temperature_2m (°C)",
    "is_day ()",
]

# Inverter AC capacity per site (kW). The inverter caps output, so a quarterly
# prediction can never exceed kW * 0.25 kWh; used to clip predictions.
INVERTER_KW = {"house1": 4.0, "house2": 2.2, "reactor": 22.0}


def quarter_cap_kwh(site_name: str) -> float:
    return INVERTER_KW[site_name] * 0.25


# Base irradiance signal used for lag/rolling features.
LAG_BASE = "shortwave_radiation (W/m²)"
# Lags in number of steps. Quarterly = 15 min per step, daily = 1 day per step.
QUARTERLY_LAGS = [1, 2, 4]          # 15, 30, 60 minutes ago
QUARTERLY_ROLL = [4, 8]             # 1-hour and 2-hour rolling mean
DAILY_LAGS = [1, 2]                 # yesterday, two days ago
DAILY_ROLL = [3, 7]                 # 3-day and 7-day rolling mean


def _cyclical(values: pd.Series, period: int) -> pd.DataFrame:
    radians = 2 * np.pi * values / period
    return pd.DataFrame(
        {f"{values.name}_sin": np.sin(radians), f"{values.name}_cos": np.cos(radians)},
        index=values.index,
    )


def _lag_features(df: pd.DataFrame, lags, rolls) -> pd.DataFrame:
    """Lag and rolling-mean features of the base irradiance signal, built on the
    full grid so shifts respect the real time order. Rolling means are shifted by
    one step so a row never sees its own value (no leakage)."""
    base = df[LAG_BASE]
    out = {}
    for k in lags:
        out[f"irr_lag_{k}"] = base.shift(k)
    for w in rolls:
        out[f"irr_roll_{w}"] = base.shift(1).rolling(w).mean()
    return pd.DataFrame(out, index=df.index)


def load_clean(site_name: str, resolution: str) -> pd.DataFrame:
    """resolution: 'quarterly' or 'daily'."""
    index_col = "datetime" if resolution == "quarterly" else "date"
    return pd.read_csv(
        CLEAN / f"{site_name}_{resolution}.csv",
        parse_dates=[index_col], index_col=index_col,
    )


def _assemble_features(df: pd.DataFrame, resolution: str, feature_set: str,
                       add_lags: bool) -> pd.DataFrame:
    """The X matrix from a weather frame on a continuous time grid. Shared by
    training (build_features) and forecasting (build_forecast_features) so the
    column set and order always match what the models were fit on."""
    if feature_set == "all":
        weather_cols = [c for c in WEATHER_FEATURES if c in df.columns]
    elif feature_set == "fair":
        weather_cols = [c for c in WEATHER_FEATURES if c in df.columns and c != LEAKY_FEATURE]
    elif feature_set == "lean":
        weather_cols = [c for c in LEAN_WEATHER if c in df.columns]
    else:
        raise ValueError(feature_set)

    X = df[weather_cols].copy()

    # clear-sky index: measured irradiance vs the theoretical (cloudless) maximum,
    # i.e. "how cloudy", independent of season and hour. 0 at night by definition.
    terrestrial = df["terrestrial_radiation (W/m²)"]
    X["clearsky_index"] = (
        (df[LAG_BASE] / terrestrial.where(terrestrial > 10)).clip(0, 1.5).fillna(0.0)
    )

    if add_lags:
        lags = QUARTERLY_LAGS if resolution == "quarterly" else DAILY_LAGS
        rolls = QUARTERLY_ROLL if resolution == "quarterly" else DAILY_ROLL
        X = pd.concat([X, _lag_features(df, lags, rolls)], axis=1)

    idx = df.index
    time_parts = [_cyclical(pd.Series(idx.dayofyear, index=idx, name="dayofyear"), 366),
                  _cyclical(pd.Series(idx.month, index=idx, name="month"), 12)]
    if resolution == "quarterly":
        time_parts.append(_cyclical(pd.Series(idx.hour, index=idx, name="hour"), 24))
    return pd.concat([X] + time_parts, axis=1)


def build_features(site_name: str, resolution: str, feature_set: str = "all",
                   add_lags: bool = False):
    """Return (X, y) with rows that have a real energy reading and complete
    features.

    feature_set:
      - "all"   : every weather column (default, backward compatible)
      - "fair"  : all weather columns except the leaky tilted irradiance
      - "lean"  : one horizontal irradiance + temperature only
    add_lags: also add lag/rolling irradiance features (see module docstring).

    Time features depend on resolution: quarterly gets hour + month + day-of-year,
    daily gets month + day-of-year (no hour)."""
    df = load_clean(site_name, resolution)

    # nights produce exactly 0; fill unreported night quarters instead of dropping
    if resolution == "quarterly":
        night_gap = (df["is_day ()"] == 0) & df[TARGET].isna()
        df.loc[night_gap, TARGET] = 0.0

    # lag features are computed on the full grid before dropping NaN energy rows
    X = _assemble_features(df, resolution, feature_set, add_lags)

    X = X.dropna()                      # drop rows with any missing feature
    y = df.loc[X.index, TARGET]
    keep = y.notna()                    # only rows with a real energy reading
    return X.loc[keep], y.loc[keep]


def build_forecast_features(weather: pd.DataFrame, feature_set: str = "fair",
                            add_lags: bool = True) -> pd.DataFrame:
    """X matrix for prediction from forecast weather on a continuous 15-min grid,
    using the training weather column names. Defaults to the fair_lag setup the
    best models use. The first rows fall inside the lag warm-up window and are
    dropped; with a forecast starting at midnight those are night quarters."""
    return _assemble_features(weather, "quarterly", feature_set, add_lags).dropna()
