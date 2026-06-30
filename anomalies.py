"""
Find underperforming days per site.

Approach: for each site, compute a daily yield ratio (kWh/kWp output per unit of
that day's total irradiance). That ratio is compared against the site's own
median ratio (over days with enough irradiance, so cloudy/winter days don't skew
the baseline). Days that fall far below are candidates for soiling, shading,
snow, or a fault.

Run: python anomalies.py
"""

import sys
import pandas as pd
from analysis import SITES
from weather_correlation import joined_data, RAD_COL

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

MIN_DAILY_RADIATION = 1000  # W/m² daily sum, threshold to skip "too cloudy to judge" days
Z_THRESHOLD = -1.5  # how many std below the median counts as an anomaly


def daily_yield_ratio(site_name: str, min_daily_radiation: float = MIN_DAILY_RADIATION) -> pd.DataFrame:
    df = joined_data(site_name)
    daily = df.resample("D").agg(
        energy_per_kwp=("energy_per_kwp", "sum"),
        radiation_sum=(RAD_COL, "sum"),
    )
    daily = daily[daily["radiation_sum"] >= min_daily_radiation].copy()
    daily["ratio"] = daily["energy_per_kwp"] / daily["radiation_sum"]
    median = daily["ratio"].median()
    std = daily["ratio"].std()
    daily["z_score"] = (daily["ratio"] - median) / std
    return daily


def flag_anomalies(
    site_name: str,
    z_threshold: float = Z_THRESHOLD,
    min_daily_radiation: float = MIN_DAILY_RADIATION,
) -> pd.DataFrame:
    daily = daily_yield_ratio(site_name, min_daily_radiation)
    return daily[daily["z_score"] <= z_threshold].sort_values("z_score")


def summary():
    for name in SITES:
        daily = daily_yield_ratio(name)
        anomalies = flag_anomalies(name)
        print(f"\n===== {name} =====")
        print(f"Days with enough irradiance to judge: {len(daily)}")
        print(f"Median yield (kWh/kWp per W/m² daily total): {daily['ratio'].median():.6f}")
        print(f"Underperforming days (z <= {Z_THRESHOLD}): {len(anomalies)}")
        if not anomalies.empty:
            print(anomalies[["energy_per_kwp", "radiation_sum", "ratio", "z_score"]])


if __name__ == "__main__":
    summary()
