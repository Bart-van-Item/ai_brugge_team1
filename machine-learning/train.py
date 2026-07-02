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

All models are trained on quarterly (15-min) data only. Daily scores come from
summing the quarterly model's predictions per day: the standalone daily models
this replaced had only 196-534 samples and lost to aggregation nearly everywhere
(reactor daily R² 0.32 -> 0.81). Daily totals are sums over the quarters that
have a reading, the same definition the daily CSVs use.

Predictions are clipped to [0, inverter cap]: a site can never produce negative
energy or more than its inverter passes (features.quarter_cap_kwh).

Feature variants (see features.build_features):
- all       : every weather column (original set)
- fair      : drops the leaky tilted-irradiance column
- fair_lag  : fair set + lag/rolling irradiance features (recent past)
Every variant also gets the clear-sky index (see features.py).

Evaluation methods (this is time-series data, so splits matter):
- time_split : train on oldest 80%, test on newest 20% (realistic, no leakage)
- random     : random 80/20 split (optimistic baseline, ignores time order);
               for daily scores the split is over whole days, so a day's quarters
               never straddle train and test
- ts_cv      : TimeSeriesSplit 5-fold, scores averaged (most robust estimate)

The winning feature variant per site/resolution/model is picked by ts_cv R²,
not by its own test score: picking by test R² was mild selection-on-test leakage.

Writes machine-learning/results.csv with one row per model/method/site/resolution
(the ts_cv-chosen variant per model), and machine-learning/variant_results.csv
with the full feature-variant detail. Saves the best quarterly model per site
(by ts_cv R², refit on all data) to models/; the daily forecast is that same
model summed per day, so there are no separate daily artifacts.

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
from features import build_features, quarter_cap_kwh, LAG_BASE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SITES = ["house1", "house2", "reactor"]
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
        return self._lr.predict(X[[LAG_BASE]])


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


def fit_predict(name, site, X_tr, y_tr, X_te):
    """Fit a fresh model and return clipped predictions as a Series on X_te's index."""
    model = make_model(name).fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 0.0, quarter_cap_kwh(site))
    return model, pd.Series(pred, index=X_te.index)


def quarterly_scores(pred, y_true):
    return r2_score(y_true, pred), mean_absolute_error(y_true, pred)


def daily_scores(pred, y_true, trim_edges):
    """Score predictions summed per calendar day. trim_edges drops the first and
    last day of the window, which a row-level split may have cut in half."""
    days = y_true.index.normalize()
    if trim_edges:
        keep = (days > days.min()) & (days < days.max())
        pred, y_true, days = pred[keep], y_true[keep], days[keep]
    day_true = y_true.groupby(days).sum()
    day_pred = pred.groupby(days).sum()
    return r2_score(day_true, day_pred), mean_absolute_error(day_true, day_pred)


def evaluate(name, site, X, y):
    """Return {(resolution, method): (r2, mae)} for one model on one feature set."""
    scores = {}
    days = X.index.normalize()
    unique_days = days.unique().sort_values()

    # time_split: one fit serves both resolutions
    cut = int(len(X) * 0.8)
    _, pred = fit_predict(name, site, X.iloc[:cut], y.iloc[:cut], X.iloc[cut:])
    scores[("quarterly", "time_split")] = quarterly_scores(pred, y.iloc[cut:])
    scores[("daily", "time_split")] = daily_scores(pred, y.iloc[cut:], trim_edges=True)

    # random over quarters (quarterly) and over whole days (daily)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    _, pred = fit_predict(name, site, X_tr, y_tr, X_te)
    scores[("quarterly", "random")] = quarterly_scores(pred, y_te)

    _, test_days = train_test_split(unique_days, test_size=0.2, random_state=42)
    te_mask = days.isin(test_days)
    _, pred = fit_predict(name, site, X[~te_mask], y[~te_mask], X[te_mask])
    scores[("daily", "random")] = daily_scores(pred, y[te_mask], trim_edges=False)

    # ts_cv: each fold serves both resolutions, scores averaged over folds
    fold_scores = {"quarterly": [], "daily": []}
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(X):
        _, pred = fit_predict(name, site, X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[te_idx])
        fold_scores["quarterly"].append(quarterly_scores(pred, y.iloc[te_idx]))
        fold_scores["daily"].append(daily_scores(pred, y.iloc[te_idx], trim_edges=True))
    for res, pairs in fold_scores.items():
        scores[(res, "ts_cv")] = tuple(float(np.mean(v)) for v in zip(*pairs))

    return scores


def main():
    variant_rows = []
    data = {}  # (site, variant) -> (X, y), kept for the final refit of saved models

    for site in SITES:
        print(f"\n=== {site} ===")
        for variant, kwargs in VARIANTS.items():
            X, y = build_features(site, "quarterly", **kwargs)
            data[(site, variant)] = (X, y)
            n_days = X.index.normalize().nunique()
            print(f"  [{variant}] {len(X)} quarters / {n_days} days, {X.shape[1]} features")
            for name in MODEL_NAMES:
                for (resolution, method), (r2, mae) in evaluate(name, site, X, y).items():
                    variant_rows.append({
                        "site": site, "resolution": resolution, "variant": variant,
                        "model": name, "method": method,
                        "r2": round(r2, 4), "mae_kwh": round(mae, 4),
                        "n_samples": len(X) if resolution == "quarterly" else n_days,
                    })

    variants_df = pd.DataFrame(variant_rows)
    variants_df.to_csv(VARIANT_PATH, index=False)
    print(f"\nWrote {VARIANT_PATH.name} ({len(variants_df)} rows)")

    # results.csv: per site/resolution/model the variant that wins on ts_cv R²,
    # reported across all methods (picking by test score would leak).
    results_rows = []
    for (site, resolution, name), group in variants_df.groupby(["site", "resolution", "model"]):
        cv = group[group["method"] == "ts_cv"]
        winner = cv.loc[cv["r2"].idxmax(), "variant"]
        chosen = group[group["variant"] == winner].copy()
        chosen["best_variant"] = winner
        results_rows.append(chosen.drop(columns="variant"))
    results = pd.concat(results_rows, ignore_index=True)
    results = results[["site", "resolution", "model", "method",
                       "r2", "mae_kwh", "n_samples", "best_variant"]]
    results.to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {RESULTS_PATH.name} ({len(results)} rows)")

    # save the best quarterly model per site (ts_cv R²), refit on all data;
    # daily forecasts reuse this same model summed per day
    cv = variants_df[(variants_df["resolution"] == "quarterly") &
                     (variants_df["method"] == "ts_cv") &
                     (variants_df["model"] != "physics")]
    for site, group in cv.groupby("site"):
        best = group.loc[group["r2"].idxmax()]
        X, y = data[(site, best["variant"])]
        model = make_model(best["model"]).fit(X, y)
        path = MODELS_DIR / f"{site}_quarterly_{best['model']}.joblib"
        joblib.dump(model, path)
        print(f"  best {site}: {best['model']} [{best['variant']}] "
              f"(ts_cv R2={best['r2']:.3f}) -> {path.name}")


if __name__ == "__main__":
    main()
