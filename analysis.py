"""
Quick exploratory analysis of all PV/weather data in this project.

Reads the cleaned per-site CSVs from data/clean/ (produced by prep_data.py). Run
prep_data.py first if that folder is empty.

Sites:
- house1: 4 kW inverter, 3 arrays (4 + 1.5 + 0.75 kWp), 2 directions
- house2: 2.2 kW inverter, 2.4 kWp, 1 direction
- reactor: 22 kW inverter, 2 arrays (16.35 + 16.35 kWp)

Run: python analysis.py
"""

import sys
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
CLEAN = ROOT / "data" / "clean"
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# energy is normalized to kWh for every site in the clean data
ENERGY_COL = "energy_kwh"

SITE_KWP = {
    "house1": 4 + 1.5 + 0.75,
    "house2": 2.4,
    "reactor": 16.35 + 16.35,
}


def load_clean(site_name: str) -> dict:
    """Load a site's cleaned quarterly CSV and split it into a pv frame (energy
    column only) and a weather frame (everything else), matching the shape the
    rest of the code expects."""
    df = pd.read_csv(
        CLEAN / f"{site_name}_quarterly.csv",
        parse_dates=["datetime"], index_col="datetime",
    )
    weather_cols = [c for c in df.columns if c != ENERGY_COL]
    return {
        "pv": df[[ENERGY_COL]],
        "weather": df[weather_cols],
        "pv_unit": ENERGY_COL,
        "kwp": SITE_KWP[site_name],
    }


SITES = {name: load_clean(name) for name in SITE_KWP}


def overview():
    for name, site in SITES.items():
        pv, weather = site["pv"], site["weather"]
        print(f"\n===== {name} =====")
        print(f"PV data:      {pv.index.min()} -> {pv.index.max()}  ({len(pv)} rows)")
        print(f"Weather data: {weather.index.min()} -> {weather.index.max()}  ({len(weather)} rows)")
        print(f"Total energy: {pv[ENERGY_COL].sum():.1f} kWh")
        print("PV sample:")
        print(pv.head(3))
        print("Weather sample:")
        print(weather[["temperature_2m (°C)", "shortwave_radiation (W/m²)"]].head(3))


def daily_energy(site_name: str) -> pd.Series:
    # min_count=1 so a day with only NaN readings stays NaN instead of summing to 0
    return SITES[site_name]["pv"][ENERGY_COL].resample("D").sum(min_count=1)


def daily_summary():
    print("\n===== daily energy (kWh) per site =====")
    combined = pd.concat(
        {name: daily_energy(name) for name in SITES}, axis=1, sort=True
    )
    print(combined.describe())
    print(combined.tail(10))
    missing = combined[combined.isna().any(axis=1)]
    if not missing.empty:
        print("\nDays with missing data for at least one site:")
        print(missing)


def specific_yield_summary():
    print("\n===== specific yield (kWh/kWp per day) =====")
    yields = {}
    for name, site in SITES.items():
        daily = daily_energy(name)
        yields[name] = daily / site["kwp"]
    combined = pd.concat(yields, axis=1, sort=True)
    print(combined.describe())
    print("\nkWp per site:", {name: site["kwp"] for name, site in SITES.items()})


if __name__ == "__main__":
    overview()
    daily_summary()
    specific_yield_summary()
