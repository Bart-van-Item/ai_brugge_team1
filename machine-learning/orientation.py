"""
Estimate panel orientation (azimuth) per site from the shape of the average
daily output profile.

We don't have the panel azimuth as metadata, so we infer it: an east-facing
array peaks in the morning, a west-facing one in the evening, south around solar
noon. We take the centre-of-mass hour of the normalized daytime output profile
(over sunny days) and map its offset from solar noon to an azimuth in degrees,
where 180 = due south, <180 = east of south, >180 = west of south.

Times in the data are UTC; local solar noon in this region (~3.3°E) is about
12:13 UTC, shifting with daylight saving. We use a fixed reference of 13.0h UTC
as "solar noon" since the profiles are averaged across the year. The result is a
rough estimate, used as a feature and as a talking point, not a precise survey.

Run: python machine-learning/orientation.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import load_clean, TARGET  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SOLAR_NOON_UTC = 13.0   # reference hour for "south" (year-averaged, region ~3.3°E)
DEGREES_PER_HOUR = 15.0  # sun moves 15° of azimuth per hour around noon


def daily_profile(site_name: str) -> pd.Series:
    """Normalized average output per hour of day, over sunny days only (so the
    orientation signal isn't washed out by overcast days)."""
    df = load_clean(site_name, "quarterly").dropna(subset=[TARGET])
    daily_rad = df["shortwave_radiation (W/m²)"].resample("D").sum()
    sunny_days = daily_rad[daily_rad > daily_rad.quantile(0.75)].index.normalize()
    sunny = df[df.index.normalize().isin(sunny_days)]
    profile = sunny.groupby(sunny.index.hour)[TARGET].mean()
    return profile / profile.max()


def estimate_azimuth(site_name: str) -> dict:
    profile = daily_profile(site_name)
    peak_hour = int(profile.idxmax())
    centre_of_mass = float((profile.index * profile).sum() / profile.sum())
    # hours after solar noon -> degrees west of south
    azimuth = 180.0 + (centre_of_mass - SOLAR_NOON_UTC) * DEGREES_PER_HOUR
    return {
        "site": site_name,
        "peak_hour_utc": peak_hour,
        "centre_of_mass_hour": round(centre_of_mass, 2),
        "azimuth_deg": round(azimuth, 1),
        "facing": describe_facing(azimuth),
    }


def describe_facing(azimuth: float) -> str:
    if azimuth < 157.5:
        return "south-east"
    if azimuth < 172.5:
        return "south, slightly east"
    if azimuth <= 187.5:
        return "south"
    if azimuth <= 202.5:
        return "south, slightly west"
    if azimuth <= 225:
        return "south-west"
    return "west"


def main():
    print("Estimated orientation per site (azimuth: 180 = south, >180 = west):\n")
    for site in ["house1", "house2", "reactor"]:
        est = estimate_azimuth(site)
        print(f"  {est['site']:8s} peak={est['peak_hour_utc']:>2}h UTC  "
              f"com={est['centre_of_mass_hour']:>5}h  "
              f"azimuth={est['azimuth_deg']:>6}°  ({est['facing']})")


if __name__ == "__main__":
    main()
