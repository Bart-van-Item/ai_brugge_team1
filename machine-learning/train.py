"""
Train and compare PV forecasting models.

Experiment grid: models x evaluation methods x sites x resolutions x feature
variants. For each combination we record R² and MAE (kWh) on held-out data, so
the data itself shows which model/variant wins.

Models:
- physics  : baseline, output proportional to horizontal irradiance (a linear fit
             of energy on shortwave radiation). Shows what plain physics buys, so
             we can tell whether ML actually adds anything.
- linear   : LinearRegression (fully interpretable)
- forest   : RandomForestRegressor (robust, handles non-linearity, no scaling)
- boosting : HistGradientBoostingRegressor (often most accurate)

Feature variants (see features.build_features):
- all       : every weather column (original set)
- fair      : drops the leaky tilted-irradiance column
- fair_lag  : fair set + lag/rolling irradiance features (recent past)

Evaluation methods (this is time-series data, so splits matter):
- time_split : train on oldest 80%, test on newest 20% (realistic, no leakage)
- random     : random 80/20 split (optimistic baseline, ignores time order)
- ts_cv      : TimeSeriesSplit 5-fold, scores averaged (most robust estimate)

Writes machine-learning/results.csv with one row per model/method/site/resolution
(best feature variant per model, so the dashboard's existing view is unchanged),
and machine-learning/variant_results.csv with the full feature-variant detail.
Saves the single best model (by time_split R²) per site+resolution to models/.

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
from features import build_features, LAG_BASE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SITES = ["house1", "house2", "reactor"]
RESOLUTIONS = ["quarterly", "daily"]
RESULTS_PATH = HERE / "results.csv"
VARIANT_PATH = HERE / "variant_results.csv"
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)

# feature variant -> kwargs for build_features
VARIANTS = {
    "all": dict(feature_set="all", add_lags=False),
    "fair": dict(feature_set="fair", add_lags=False),
    "fair_lag": dict(feature_set="fair", add_lags=True),
}


class PhysicsBaseline:
    """Predict energy from horizontal irradiance only (single-feature linear fit).
    A floor to beat: if a model can't clear this, it isn't earning its complexity."""

    def __init__(self):
        self._lr = LinearRegression()

    def fit(self, X, y):
        self._lr.fit(X[[LAG_BASE]], y)
        return self

    def predict(self, X):
        return np.clip(self._lr.predict(X[[LAG_BASE]]), 0, None)


def make_model(name):
    if name == "physics":
        return PhysicsBaseline()
    if name == "linear":
        return LinearRegression()
    if name == "forest":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "boosting":
        return HistGradientBoostingRegressor(random_state=42)
    raise ValueError(name)


MODEL_NAMES = ["physics", "linear", "forest", "boosting"]


def evaluate(name, X, y, method):
    """Return (r2, mae, fitted_model). For ts_cv the scores are averaged over
    folds and the returned model is refit on all data for saving."""
    if method == "time_split":
        cut = int(len(X) * 0.8)
        model = make_model(name).fit(X.iloc[:cut], y.iloc[:cut])
        pred = model.predict(X.iloc[cut:])
        return r2_score(y.iloc[cut:], pred), mean_absolute_error(y.iloc[cut:], pred), model

    if method == "random":
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = make_model(name).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        return r2_score(y_te, pred), mean_absolute_error(y_te, pred), model

    if method == "ts_cv":
        splitter = TimeSeriesSplit(n_splits=5)
        r2s, maes = [], []
        for tr_idx, te_idx in splitter.split(X):
            model = make_model(name).fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[te_idx])
            r2s.append(r2_score(y.iloc[te_idx], pred))
            maes.append(mean_absolute_error(y.iloc[te_idx], pred))
        model = make_model(name).fit(X, y)  # refit on everything for the saved artifact
        return float(np.mean(r2s)), float(np.mean(maes)), model

    raise ValueError(method)


def main():
    variant_rows = []          # full detail: one row per variant too
    best_variant = {}          # (site, res, model, method) -> (r2, mae, variant, model_obj)

    for site in SITES:
        for resolution in RESOLUTIONS:
            print(f"\n=== {site} / {resolution} ===")
            for variant, kwargs in VARIANTS.items():
                X, y = build_features(site, resolution, **kwargs)
                print(f"  [{variant}] {len(X)} samples, {X.shape[1]} features")
                for method in ["time_split", "random", "ts_cv"]:
                    for name in MODEL_NAMES:
                        r2, mae, fitted = evaluate(name, X, y, method)
                        variant_rows.append({
                            "site": site, "resolution": resolution, "variant": variant,
                            "model": name, "method": method,
                            "r2": round(r2, 4), "mae_kwh": round(mae, 4), "n_samples": len(X),
                        })
                        key = (site, resolution, name, method)
                        if key not in best_variant or r2 > best_variant[key][0]:
                            best_variant[key] = (r2, mae, variant, fitted, len(X))

    variants_df = pd.DataFrame(variant_rows)
    variants_df.to_csv(VARIANT_PATH, index=False)
    print(f"\nWrote {VARIANT_PATH.name} ({len(variants_df)} rows)")

    # results.csv keeps its original schema (one row per model/method/site/res),
    # using the best feature variant for each, so the dashboard view still works.
    results_rows = []
    for (site, res, name, method), (r2, mae, variant, _, n) in best_variant.items():
        results_rows.append({
            "site": site, "resolution": res, "model": name, "method": method,
            "r2": round(r2, 4), "mae_kwh": round(mae, 4), "n_samples": n,
            "best_variant": variant,
        })
    results = pd.DataFrame(results_rows)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {RESULTS_PATH.name} ({len(results)} rows)")

    # save the best model per site+resolution by the realistic time_split metric
    best_saved = {}
    for (site, res, name, method), (r2, mae, variant, model, _) in best_variant.items():
        if method != "time_split":
            continue
        key = (site, res)
        if key not in best_saved or r2 > best_saved[key][0]:
            best_saved[key] = (r2, name, variant, model)
    for (site, res), (r2, name, variant, model) in best_saved.items():
        path = MODELS_DIR / f"{site}_{res}_{name}.joblib"
        joblib.dump(model, path)
        print(f"  best {site}/{res}: {name} [{variant}] (R2={r2:.3f}) -> {path.name}")


if __name__ == "__main__":
    main()
