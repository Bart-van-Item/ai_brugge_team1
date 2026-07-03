"""
General vs specific (theme) models.

Keep the full-feature model as the "general" allrounder, and compare it against
specific models that each train on one theme of features only. This shows how
much each aspect explains on its own, which is useful for the map tool (where
the user picks a direction, so we want to know how much "direction" alone buys).

Themes:
- general     : all features (the allrounder we keep)
- direction   : irradiance components + hour of day (the daily profile reveals
                panel orientation: east peaks morning, west evening)
- time_season : hour, day-of-year, month only (when, without knowing weather)
- weather     : weather conditions (irradiance, temp, humidity, cloud), no time
- irradiance  : the radiation measures only, the most direct driver

Evaluated on the realistic time_split, per site. Forest model (best on quarterly).

Run: python machine-learning/theme_experiment.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
from features import build_features  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SITES = ["house1", "house2", "reactor"]

IRRADIANCE = [
    "shortwave_radiation (W/m²)", "direct_radiation (W/m²)", "diffuse_radiation (W/m²)",
    "direct_normal_irradiance (W/m²)", "global_tilted_irradiance (W/m²)",
    "terrestrial_radiation (W/m²)",
]
WEATHER = IRRADIANCE + [
    "temperature_2m (°C)", "relative_humidity_2m (%)", "dew_point_2m (°C)",
    "apparent_temperature (°C)", "weather_code (wmo code)", "wind_speed_10m (km/h)",
    "visibility (m)", "is_day ()",
]
HOUR = ["hour_sin", "hour_cos"]
SEASON = ["dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos"]

THEMES = {
    "general": None,  # all features
    "direction": IRRADIANCE + HOUR,
    "time_season": HOUR + SEASON,
    "weather": WEATHER,
    "irradiance": IRRADIANCE,
}


def select(X: pd.DataFrame, cols) -> pd.DataFrame:
    if cols is None:
        return X
    return X[[c for c in cols if c in X.columns]]


def time_split_score(model, X, y):
    cut = int(len(X) * 0.8)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    return r2_score(y.iloc[cut:], pred), mean_absolute_error(y.iloc[cut:], pred)


def main():
    rows = []
    for site in SITES:
        X_full, y = build_features(site, "quarterly")
        for theme, cols in THEMES.items():
            X = select(X_full, cols)
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            r2, mae = time_split_score(model, X, y)
            rows.append({
                "site": site, "theme": theme, "n_features": X.shape[1],
                "r2": round(r2, 4), "mae_kwh": round(mae, 4),
            })

    results = pd.DataFrame(rows)
    # pivot so each theme is a column, easy to compare general vs specific
    pivot = results.pivot(index="theme", columns="site", values="r2")
    pivot = pivot.reindex(list(THEMES))
    print("R² per theme per site (forest, time_split):\n")
    print(pivot.to_string())
    print("\nFull detail:")
    print(results.to_string(index=False))

    results.to_csv(ML / "results" / "theme_results.csv", index=False)
    print(f"\nWrote theme_results.csv")


if __name__ == "__main__":
    main()
