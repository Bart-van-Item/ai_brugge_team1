import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import SITES, daily_energy
from weather_correlation import joined_data, RAD_COL, TEMP_COL
from anomalies import daily_yield_ratio, flag_anomalies

ML_DIR = Path(__file__).resolve().parent.parent / "machine-learning"


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
    return pd.read_csv(ML_DIR / "results" / name)


# the machine-learning/ dir has a hyphen (not importable as a package), so add it
# to the path and import its modules directly
sys.path.insert(0, str(ML_DIR))
from orientation import estimate_azimuth  # noqa: E402


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


def in_range(index, date_range) -> pd.Series:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    return (index >= start) & (index < end)
