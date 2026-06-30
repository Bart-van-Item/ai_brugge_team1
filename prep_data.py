"""
Data preparation: read raw CSVs from data/raw/, clean and join them, and write
ML-ready clean CSVs to data/clean/.

This is the only script that touches the raw data. Everything downstream reads
from data/clean/. Run it once (or whenever the raw data changes):

    python prep_data.py

For each site it writes two files to data/clean/:
- <site>_quarterly.csv : PV energy + weather joined per 15 minutes
- <site>_daily.csv      : aggregated per day

Energy is normalized to kWh for every site. Missing readings stay empty (NaN),
not zero, so e.g. the reactor "Geen gegevens" gaps are not mistaken for zero
production downstream.
"""

import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# weather columns we keep, in a fixed order (raw files differ in column order per site)
WEATHER_COLUMNS = [
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


def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=3)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    return df[WEATHER_COLUMNS]


def load_pv_energy(*paths: Path) -> pd.Series:
    """house1/house2 PV files: datetime,energy_wh -> kWh series named energy_kwh."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    energy = df["energy_wh"] / 1000.0
    energy.name = "energy_kwh"
    return energy


def load_reactor_energy(path: Path) -> pd.Series:
    """Reactor meter file: semicolon-separated, comma decimals, one row per
    register per timestamp. We keep the active production register, in kWh."""
    df = pd.read_csv(path, sep=";")
    df["datetime"] = pd.to_datetime(
        df["Van (datum)"] + " " + df["Van (tijdstip)"], dayfirst=True
    )
    # empty Volume means "Geen gegevens" (no reading), not zero -> keep as NaN
    df["Volume"] = df["Volume"].astype(str).str.replace(",", ".", regex=False)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    active = df[df["Register"] == "Productie Actief"]
    energy = active.set_index("datetime")["Volume"].sort_index()
    # restore timestamps where the reading was missing (dropped by the filter above)
    full_index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="15min")
    energy = energy.reindex(full_index)
    energy.name = "energy_kwh"
    return energy


SITES = {
    "house1": {
        "weather": RAW / "house1" / "weer_data_house1.csv",
        "energy": lambda: load_pv_energy(
            RAW / "house1" / "PV_data_house1_2025.csv",
            RAW / "house1" / "PV_data_house1_2026.csv",
        ),
    },
    "house2": {
        "weather": RAW / "house2" / "weer_data_house2.csv",
        "energy": lambda: load_pv_energy(
            RAW / "house2" / "PV_data_house2_2025.csv",
            RAW / "house2" / "PV_data_house2_2026.csv",
        ),
    },
    "reactor": {
        "weather": RAW / "reactor" / "weer_reactor.csv",
        "energy": lambda: load_reactor_energy(
            RAW / "reactor"
            / "Historiek_submeting_elektriciteit_541454897100239158_20251201_20260624_kwartiertotalen.csv"
        ),
    },
}


def build_quarterly(site_name: str) -> pd.DataFrame:
    site = SITES[site_name]
    energy = site["energy"]()
    weather = load_weather(site["weather"])
    df = pd.concat([energy, weather], axis=1, sort=True)
    df.index.name = "datetime"
    return df


def build_daily(quarterly: pd.DataFrame) -> pd.DataFrame:
    # energy: sum per day; min_count=1 so all-missing days stay NaN, not 0
    daily_energy = quarterly["energy_kwh"].resample("D").sum(min_count=1)
    # weather: daily means, plus the irradiance sum which is what the analysis uses
    weather_cols = [c for c in quarterly.columns if c != "energy_kwh"]
    daily_weather = quarterly[weather_cols].resample("D").mean()
    daily = pd.concat([daily_energy, daily_weather], axis=1)
    daily["radiation_sum"] = quarterly["shortwave_radiation (W/m²)"].resample("D").sum(min_count=1)
    daily.index.name = "date"
    return daily


def main():
    for name in SITES:
        quarterly = build_quarterly(name)
        daily = build_daily(quarterly)

        q_path = CLEAN / f"{name}_quarterly.csv"
        d_path = CLEAN / f"{name}_daily.csv"
        quarterly.to_csv(q_path)
        daily.to_csv(d_path)

        print(f"\n===== {name} =====")
        print(f"quarterly: {len(quarterly)} rows -> {q_path.name}")
        print(f"  range {quarterly.index.min()} .. {quarterly.index.max()}")
        print(f"  energy NaN: {quarterly['energy_kwh'].isna().sum()}, total {quarterly['energy_kwh'].sum():.1f} kWh")
        print(f"daily: {len(daily)} rows -> {d_path.name}")
        print(f"  days with energy: {daily['energy_kwh'].notna().sum()}, NaN days: {daily['energy_kwh'].isna().sum()}")


if __name__ == "__main__":
    main()
