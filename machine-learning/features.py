"""
Feature engineering for the PV forecasting models.

Builds an (X, y) feature matrix from a site's cleaned quarterly or daily CSV:
- X: weather columns + cyclical time features (hour, month, day-of-year)
- y: energy_kwh

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

    if feature_set == "all":
        weather_cols = [c for c in WEATHER_FEATURES if c in df.columns]
    elif feature_set == "fair":
        weather_cols = [c for c in WEATHER_FEATURES if c in df.columns and c != LEAKY_FEATURE]
    elif feature_set == "lean":
        weather_cols = [c for c in LEAN_WEATHER if c in df.columns]
    else:
        raise ValueError(feature_set)

    X = df[weather_cols].copy()

    # lag features are computed on the full grid before dropping NaN energy rows
    if add_lags:
        lags = QUARTERLY_LAGS if resolution == "quarterly" else DAILY_LAGS
        rolls = QUARTERLY_ROLL if resolution == "quarterly" else DAILY_ROLL
        X = pd.concat([X, _lag_features(df, lags, rolls)], axis=1)

    idx = df.index
    time_parts = [_cyclical(pd.Series(idx.dayofyear, index=idx, name="dayofyear"), 366),
                  _cyclical(pd.Series(idx.month, index=idx, name="month"), 12)]
    if resolution == "quarterly":
        time_parts.append(_cyclical(pd.Series(idx.hour, index=idx, name="hour"), 24))
    X = pd.concat([X] + time_parts, axis=1)

    X = X.dropna()                      # drop rows with any missing feature
    y = df.loc[X.index, TARGET]
    keep = y.notna()                    # only rows with a real energy reading
    return X.loc[keep], y.loc[keep]
