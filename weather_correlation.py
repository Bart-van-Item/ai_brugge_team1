"""
Correlation between PV output and weather data per site.

Join PV energy (per quarter-hour) with weather data (per quarter-hour) and look at:
- correlation between irradiance (shortwave_radiation) and energy
- yield: how much Wh/kWp is produced per unit of irradiance (~ performance ratio)
- effect of temperature on that yield (panels lose efficiency at high temperature)

Run: python weather_correlation.py
"""

import sys
import pandas as pd
from analysis import SITES

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

RAD_COL = "shortwave_radiation (W/m²)"
TEMP_COL = "temperature_2m (°C)"


def joined_data(site_name: str) -> pd.DataFrame:
    site = SITES[site_name]
    pv_col = site["pv_unit"]
    pv = site["pv"][[pv_col]].rename(columns={pv_col: "energy"})
    if pv_col == "energy_wh":
        pv["energy"] = pv["energy"] / 1000  # -> kWh per quarter-hour
    weather = site["weather"][[RAD_COL, TEMP_COL]]

    df = pv.join(weather, how="inner")
    df["energy_per_kwp"] = df["energy"] / site["kwp"]  # kWh/kWp per quarter-hour
    return df.dropna()


def correlation_summary():
    print("===== correlation irradiance <-> output (per quarter-hour) =====")
    for name in SITES:
        df = joined_data(name)
        corr = df["energy"].corr(df[RAD_COL])
        print(f"{name:10s} corr(energy, radiation) = {corr:.3f}   (n={len(df)})")


def performance_ratio():
    # only quarter-hours with meaningful irradiance, otherwise dividing by near-zero
    print("\n===== yield: kWh/kWp per (W/m²) irradiance, where irradiance > 100 W/m² =====")
    for name in SITES:
        df = joined_data(name)
        lit = df[df[RAD_COL] > 100].copy()
        lit["yield_per_irradiance"] = lit["energy_per_kwp"] / lit[RAD_COL]
        print(f"\n{name} (n={len(lit)} quarter-hours)")
        print(lit["yield_per_irradiance"].describe())


def temperature_effect():
    print("\n===== effect of temperature on yield (irradiance 400-600 W/m², comparable light) =====")
    for name in SITES:
        df = joined_data(name)
        band = df[(df[RAD_COL] >= 400) & (df[RAD_COL] <= 600)].copy()
        if band.empty:
            print(f"\n{name}: no data in this band")
            continue
        band["temp_bin"] = pd.cut(band[TEMP_COL], bins=[-10, 5, 10, 15, 20, 25, 30, 40])
        grouped = band.groupby("temp_bin", observed=True)["energy_per_kwp"].agg(["mean", "count"])
        print(f"\n{name} (irradiance 400-600 W/m², n={len(band)})")
        print(grouped)


if __name__ == "__main__":
    correlation_summary()
    performance_ratio()
    temperature_effect()
