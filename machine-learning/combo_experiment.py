"""
Combination experiment: find a strong-but-compact specific model.

The theme experiment showed `direction` (irradiance + hour) is almost as good as
the full `general` set. Here we test combinations on top of direction to see
whether adding season and/or temperature buys anything, and how small we can go.

Combinations (compared against general and the plain direction theme):
- general          : all features (reference allrounder)
- direction        : irradiance components + hour
- dir_season       : direction + day-of-year + month
- dir_temp         : direction + temperature + humidity
- dir_season_temp  : direction + season + temperature (near-general, drops the
                     least useful: visibility, wind, weather_code, dew point, etc.)
- core_hour        : shortwave irradiance + hour only (3 features, the floor test)

Evaluated on the realistic time_split, per site, forest model.

Run: python machine-learning/combo_experiment.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import build_features  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SITES = ["house1", "house2", "reactor"]

IRRADIANCE = [
    "shortwave_radiation (W/m²)", "direct_radiation (W/m²)", "diffuse_radiation (W/m²)",
    "direct_normal_irradiance (W/m²)", "global_tilted_irradiance (W/m²)",
    "terrestrial_radiation (W/m²)",
]
HOUR = ["hour_sin", "hour_cos"]
SEASON = ["dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos"]
TEMP = ["temperature_2m (°C)", "relative_humidity_2m (%)"]

DIRECTION = IRRADIANCE + HOUR

COMBOS = {
    "general": None,
    "direction": DIRECTION,
    "dir_season": DIRECTION + SEASON,
    "dir_temp": DIRECTION + TEMP,
    "dir_season_temp": DIRECTION + SEASON + TEMP,
    "core_hour": ["shortwave_radiation (W/m²)"] + HOUR,
}


def select(X, cols):
    return X if cols is None else X[[c for c in cols if c in X.columns]]


def time_split_score(model, X, y):
    cut = int(len(X) * 0.8)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    return r2_score(y.iloc[cut:], pred), mean_absolute_error(y.iloc[cut:], pred)


def main():
    rows = []
    for site in SITES:
        X_full, y = build_features(site, "quarterly")
        for combo, cols in COMBOS.items():
            X = select(X_full, cols)
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            r2, mae = time_split_score(model, X, y)
            rows.append({
                "site": site, "combo": combo, "n_features": X.shape[1],
                "r2": round(r2, 4), "mae_kwh": round(mae, 4),
            })

    results = pd.DataFrame(rows)
    pivot = results.pivot(index="combo", columns="site", values="r2").reindex(list(COMBOS))
    n_feat = results.groupby("combo")["n_features"].first().reindex(list(COMBOS))
    pivot.insert(0, "n_features", n_feat)
    print("R² per combination per site (forest, time_split):\n")
    print(pivot.to_string())

    results.to_csv(HERE / "combo_results.csv", index=False)
    print("\nWrote combo_results.csv")


if __name__ == "__main__":
    main()
