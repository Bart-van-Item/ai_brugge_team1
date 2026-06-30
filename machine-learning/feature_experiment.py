"""
Feature-set experiment: which features actually matter for predicting PV output?

The default feature set has six overlapping irradiance measures, and feature
importance showed the model spreading its weight across them inconsistently and
leaning on proxies (humidity, hour). This script compares feature sets on the
realistic time_split, per site, to find a leaner, fairer set.

Feature sets tested:
- all        : every feature (the current set, 20 columns)
- no_tilted  : drop global_tilted_irradiance (irradiance ON the panel plane -- almost
               the answer, and orientation-specific, so unfair for a general model)
- core       : one irradiance measure (shortwave) + temperature + time features
- minimal    : shortwave irradiance + time features only

Run: python machine-learning/feature_experiment.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import build_features  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SITES = ["house1", "house2", "reactor"]

TIME_FEATURES = ["dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "hour_sin", "hour_cos"]

FEATURE_SETS = {
    "all": None,  # None means "use every column"
    "no_tilted": "drop:global_tilted_irradiance (W/m²)",
    "core": ["shortwave_radiation (W/m²)", "temperature_2m (°C)", "is_day ()"] + TIME_FEATURES,
    "minimal": ["shortwave_radiation (W/m²)", "is_day ()"] + TIME_FEATURES,
}


def select(X: pd.DataFrame, spec) -> pd.DataFrame:
    if spec is None:
        return X
    if isinstance(spec, str) and spec.startswith("drop:"):
        col = spec[len("drop:"):]
        return X.drop(columns=[col])
    return X[[c for c in spec if c in X.columns]]


def time_split_score(model, X, y):
    cut = int(len(X) * 0.8)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    return r2_score(y.iloc[cut:], pred), mean_absolute_error(y.iloc[cut:], pred)


def main():
    rows = []
    for site in SITES:
        X_full, y = build_features(site, "quarterly")
        for set_name, spec in FEATURE_SETS.items():
            X = select(X_full, spec)
            for model_name, model in [
                ("linear", LinearRegression()),
                ("forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ]:
                r2, mae = time_split_score(model, X, y)
                rows.append({
                    "site": site, "feature_set": set_name, "n_features": X.shape[1],
                    "model": model_name, "r2": round(r2, 4), "mae_kwh": round(mae, 4),
                })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    print("\n=== Best feature set per site (forest, time_split R²) ===")
    forest = results[results["model"] == "forest"]
    best = forest.loc[forest.groupby("site")["r2"].idxmax()]
    print(best[["site", "feature_set", "n_features", "r2", "mae_kwh"]].to_string(index=False))


if __name__ == "__main__":
    main()
