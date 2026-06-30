"""
Feature engineering for the PV forecasting models.

Builds an (X, y) feature matrix from a site's cleaned quarterly or daily CSV:
- X: weather columns + cyclical time features (hour, month, day-of-year)
- y: energy_kwh

Time features are encoded as sin/cos pairs so the model sees that e.g. hour 23
and hour 0 are adjacent, and December and January are adjacent.

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


def _cyclical(values: pd.Series, period: int) -> pd.DataFrame:
    radians = 2 * np.pi * values / period
    return pd.DataFrame(
        {f"{values.name}_sin": np.sin(radians), f"{values.name}_cos": np.cos(radians)},
        index=values.index,
    )


def load_clean(site_name: str, resolution: str) -> pd.DataFrame:
    """resolution: 'quarterly' or 'daily'."""
    index_col = "datetime" if resolution == "quarterly" else "date"
    return pd.read_csv(
        CLEAN / f"{site_name}_{resolution}.csv",
        parse_dates=[index_col], index_col=index_col,
    )


def build_features(site_name: str, resolution: str):
    """Return (X, y) with rows that have a real energy reading and complete
    weather. Time features depend on resolution: quarterly gets hour + month +
    day-of-year, daily gets month + day-of-year (no hour)."""
    df = load_clean(site_name, resolution)
    df = df.dropna(subset=[TARGET])  # only rows with a real energy reading

    weather_cols = [c for c in WEATHER_FEATURES if c in df.columns]
    X = df[weather_cols].copy()

    idx = df.index
    time_parts = [_cyclical(pd.Series(idx.dayofyear, index=idx, name="dayofyear"), 366),
                  _cyclical(pd.Series(idx.month, index=idx, name="month"), 12)]
    if resolution == "quarterly":
        time_parts.append(_cyclical(pd.Series(idx.hour, index=idx, name="hour"), 24))
    X = pd.concat([X] + time_parts, axis=1)

    X = X.dropna()              # drop rows with any missing weather feature
    y = df.loc[X.index, TARGET]
    return X, y
