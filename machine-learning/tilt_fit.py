"""
Estimate panel tilt and azimuth per site from solar geometry.

The weather data carries the irradiance components (direct-normal, diffuse,
global horizontal), so we can compute what a plane with a given tilt and
azimuth would receive (plane-of-array transposition, isotropic sky model) and
grid-search which plane best explains the measured output.

Two-stage method: each parameter is estimated from the subset of the data that
identifies it most cleanly.

Stage A — azimuth from the daily shape. Solar position per 15-min interval
midpoint (site's own lat/lon, UTC), plane-of-array irradiance for every
candidate plane, scored on clear, unclipped daylight samples (sun elevation
> 15 deg, clearness index > 0.55, real reading, below 98% of the inverter cap).
The morning half of the day independently confirms the azimuth, so it is not
an artefact of the evening bias described below.

Stage B — tilt from the seasonal midday arc. Around solar noon (sun azimuth
160-200 deg) the sun's elevation sweeps 16 deg (winter) to 62 deg (summer);
how midday output scales along that arc is almost purely a function of tilt,
and the morning/evening asymmetries that contaminate a full-day tilt fit drop
out. Tilt is scanned at the stage-A azimuth on the midday subset.

Both stages report the *ridge*: every candidate within 0.01 of the best score.
A narrow ridge means the parameter is well determined.

Known systematic: all three sites show a shared morning deficit / evening
surplus versus any fixed plane (haze, dew or weather-model bias; a PVWatts
temperature correction was tested and does not remove it). This biases
full-day fits westward and steep, which is why tilt comes from midday only.
For the reactor (two equal arrays) a low midday tilt combined with a
west-leaning daily shape is the fingerprint of an east-west tent pair; the
two-plane decomposition (two_plane_fit.py) cannot separate such faces because
they are nearly collinear at low tilt. Writes machine-learning/tilt_results.csv.

Run: python machine-learning/tilt_fit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import load_clean, quarter_cap_kwh, TARGET  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RAW = HERE.parent / "data" / "raw"
WEATHER_FILES = {
    "house1": RAW / "house1" / "weer_data_house1.csv",
    "house2": RAW / "house2" / "weer_data_house2.csv",
    "reactor": RAW / "reactor" / "weer_reactor.csv",
}
RESULTS_PATH = HERE / "results" / "tilt_results.csv"

GHI = "shortwave_radiation (W/m²)"
DNI = "direct_normal_irradiance (W/m²)"
DHI = "diffuse_radiation (W/m²)"
EXTRA = "terrestrial_radiation (W/m²)"

MIN_ELEVATION_DEG = 15
MIN_CLEARNESS = 0.55
ALBEDO = 0.2
RIDGE_MARGIN = 0.01                      # candidates this close to the best form the ridge
TILT_GRID = np.arange(0, 62, 2)          # degrees from horizontal (roof planes)
AZIMUTH_GRID = np.arange(60, 302, 4)     # degrees from north: 90=E, 180=S, 270=W
MIDDAY_SUN_AZ = (160, 200)               # stage-B window around solar noon

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S",
           "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"]


def facing_label(azimuth_deg: float) -> str:
    return COMPASS[int(round(azimuth_deg / 22.5))]


def site_latlon(site: str) -> tuple:
    meta = pd.read_csv(WEATHER_FILES[site], nrows=1)
    return float(meta["latitude"].iloc[0]), float(meta["longitude"].iloc[0])


def solar_position(index: pd.DatetimeIndex, lat: float, lon: float):
    """Zenith and azimuth (radians) at each timestamp (UTC). Azimuth is from
    north, clockwise. Standard formulas: Cooper declination, Spencer equation
    of time."""
    n = index.dayofyear.to_numpy()
    b = 2 * np.pi * (n - 1) / 365.0
    declination = np.radians(23.45) * np.sin(2 * np.pi * (284 + n) / 365.0)
    eot_min = 229.18 * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b)
                        - 0.014615 * np.cos(2 * b) - 0.04089 * np.sin(2 * b))

    utc_minutes = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    solar_minutes = utc_minutes + 4.0 * lon + eot_min
    hour_angle = np.radians((solar_minutes / 60.0 - 12.0) * 15.0)

    phi = np.radians(lat)
    cos_zenith = (np.sin(phi) * np.sin(declination)
                  + np.cos(phi) * np.cos(declination) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1, 1)
    zenith = np.arccos(cos_zenith)

    # atan2 form: 0 at south, west positive; +pi makes it from-north clockwise
    azimuth = np.arctan2(np.sin(hour_angle),
                         np.cos(hour_angle) * np.sin(phi)
                         - np.tan(declination) * np.cos(phi)) + np.pi
    return zenith, azimuth


def poa_irradiance(df: pd.DataFrame, zenith, sun_az, tilt_deg: float, az_deg: float):
    """Plane-of-array irradiance (W/m²) for one candidate plane, isotropic sky."""
    beta = np.radians(tilt_deg)
    gamma = np.radians(az_deg)
    cos_aoi = (np.cos(zenith) * np.cos(beta)
               + np.sin(zenith) * np.sin(beta) * np.cos(sun_az - gamma))
    beam = df[DNI].to_numpy() * np.clip(cos_aoi, 0, None)
    sky = df[DHI].to_numpy() * (1 + np.cos(beta)) / 2
    ground = df[GHI].to_numpy() * ALBEDO * (1 - np.cos(beta)) / 2
    return beam + sky + ground


def explained_variance(y: np.ndarray, poa: np.ndarray) -> float:
    """Variance of y explained by the physical model y = k * POA (no intercept)."""
    denom = np.sum(poa * poa)
    if denom == 0:
        return -np.inf
    k = np.sum(y * poa) / denom
    ss_res = np.sum((y - k * poa) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def clear_samples(site: str):
    """Clear, unclipped daylight quarters with a real reading, plus the solar
    position at each interval midpoint."""
    df = load_clean(site, "quarterly")
    lat, lon = site_latlon(site)
    midpoints = df.index + pd.Timedelta(minutes=7.5)
    zenith, sun_az = solar_position(midpoints, lat, lon)

    elevation_ok = zenith < np.radians(90 - MIN_ELEVATION_DEG)
    with np.errstate(divide="ignore", invalid="ignore"):
        clearness = df[GHI].to_numpy() / df[EXTRA].to_numpy()
    clear = np.nan_to_num(clearness) > MIN_CLEARNESS
    unclipped = df[TARGET].to_numpy() < 0.98 * quarter_cap_kwh(site)
    keep = elevation_ok & clear & df[TARGET].notna().to_numpy() & unclipped

    return df[keep], zenith[keep], sun_az[keep]


def fit_site(site: str) -> dict:
    df, zenith, sun_az = clear_samples(site)
    y = df[TARGET].to_numpy()

    # stage A: azimuth from the full daily shape (tilt here is the *effective*
    # full-day plane, reported only as reference)
    scored = []
    for tilt in TILT_GRID:
        for az in AZIMUTH_GRID:
            score = explained_variance(y, poa_irradiance(df, zenith, sun_az, tilt, az))
            scored.append((score, tilt, az))
    scored.sort(reverse=True)
    flat = explained_variance(y, df[GHI].to_numpy())
    score_a, _, azimuth = scored[0]
    ridge_a = [a for s, _, a in scored if s >= score_a - RIDGE_MARGIN]

    # stage B: tilt from the seasonal midday arc, at the stage-A azimuth
    midday = ((np.degrees(sun_az) > MIDDAY_SUN_AZ[0])
              & (np.degrees(sun_az) < MIDDAY_SUN_AZ[1]))
    dfm, zenm, sazm = df[midday], zenith[midday], sun_az[midday]
    ym = y[midday]
    tilt_scores = [(explained_variance(ym, poa_irradiance(dfm, zenm, sazm, t, azimuth)), t)
                   for t in TILT_GRID]
    score_b, tilt = max(tilt_scores)
    ridge_b = [t for s, t in tilt_scores if s >= score_b - RIDGE_MARGIN]

    return {"site": site,
            "azimuth_deg": int(azimuth), "facing": facing_label(azimuth),
            "azimuth_ridge": f"{min(ridge_a)}-{max(ridge_a)}",
            "tilt_deg": int(tilt), "tilt_ridge": f"{min(ridge_b)}-{max(ridge_b)}",
            "r2_shape": round(score_a, 4), "r2_midday": round(score_b, 4),
            "r2_flat": round(flat, 4),
            "n_samples": len(df), "n_midday": int(midday.sum())}


def main():
    rows = []
    for site in WEATHER_FILES:
        row = fit_site(site)
        rows.append(row)
        print(f"{site}: azimuth {row['azimuth_deg']}° {row['facing']} "
              f"(ridge {row['azimuth_ridge']}), tilt {row['tilt_deg']}° "
              f"(ridge {row['tilt_ridge']}, midday fit R²={row['r2_midday']:.3f}), "
              f"daily-shape R²={row['r2_shape']:.3f}, flat GHI {row['r2_flat']:.3f}, "
              f"n={row['n_samples']}/{row['n_midday']} midday")
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
