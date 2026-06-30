"""
Train and compare PV forecasting models.

Experiment grid: 3 models x 3 evaluation methods x 3 sites x 2 resolutions.
For each combination we record R² and MAE (kWh) on held-out data, so the data
itself shows which model/method wins.

Models:
- linear   : LinearRegression (benchmark, fully interpretable)
- forest   : RandomForestRegressor (robust, handles non-linearity, no scaling)
- boosting : HistGradientBoostingRegressor (often most accurate)

Evaluation methods (this is time-series data, so splits matter):
- time_split : train on oldest 80%, test on newest 20% (realistic, no leakage)
- random     : random 80/20 split (optimistic baseline, ignores time order)
- ts_cv      : TimeSeriesSplit 5-fold, scores averaged (most robust estimate)

Writes machine-learning/results.csv with one row per combination, and saves the
single best model (by time_split R²) per site+resolution to models/.

Run: python machine-learning/train.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import build_features  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SITES = ["house1", "house2", "reactor"]
RESOLUTIONS = ["quarterly", "daily"]
RESULTS_PATH = HERE / "results.csv"
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)


def make_models():
    # fresh instances each time so nothing leaks between fits
    return {
        "linear": LinearRegression(),
        "forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "boosting": HistGradientBoostingRegressor(random_state=42),
    }


def evaluate(model, X, y, method):
    """Return (r2, mae, fitted_model_on_train). For ts_cv the scores are averaged
    over folds and the returned model is refit on all data for saving."""
    if method == "time_split":
        n = len(X)
        cut = int(n * 0.8)
        X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
        y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        return r2_score(y_te, pred), mean_absolute_error(y_te, pred), model

    if method == "random":
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        return r2_score(y_te, pred), mean_absolute_error(y_te, pred), model

    if method == "ts_cv":
        splitter = TimeSeriesSplit(n_splits=5)
        r2s, maes = [], []
        for tr_idx, te_idx in splitter.split(X):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[te_idx])
            r2s.append(r2_score(y.iloc[te_idx], pred))
            maes.append(mean_absolute_error(y.iloc[te_idx], pred))
        model.fit(X, y)  # refit on everything for the saved artifact
        return float(np.mean(r2s)), float(np.mean(maes)), model

    raise ValueError(method)


def main():
    rows = []
    best_per_target = {}  # (site, resolution) -> (r2, model, model_name)

    for site in SITES:
        for resolution in RESOLUTIONS:
            X, y = build_features(site, resolution)
            print(f"\n{site} / {resolution}: {len(X)} samples, {X.shape[1]} features")

            for method in ["time_split", "random", "ts_cv"]:
                for model_name, _ in make_models().items():
                    model = make_models()[model_name]
                    r2, mae, fitted = evaluate(model, X, y, method)
                    rows.append({
                        "site": site, "resolution": resolution,
                        "model": model_name, "method": method,
                        "r2": round(r2, 4), "mae_kwh": round(mae, 4),
                        "n_samples": len(X),
                    })
                    print(f"  {method:11s} {model_name:9s}  R2={r2:6.3f}  MAE={mae:.3f} kWh")

                    # track the best model by the realistic time_split metric
                    if method == "time_split":
                        key = (site, resolution)
                        if key not in best_per_target or r2 > best_per_target[key][0]:
                            best_per_target[key] = (r2, fitted, model_name)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {RESULTS_PATH.name} ({len(results)} rows)")

    for (site, resolution), (r2, model, model_name) in best_per_target.items():
        path = MODELS_DIR / f"{site}_{resolution}_{model_name}.joblib"
        joblib.dump(model, path)
        print(f"  best {site}/{resolution}: {model_name} (R2={r2:.3f}) -> {path.name}")


if __name__ == "__main__":
    main()
