# Machine learning notes

How the PV prediction is built, which models were tried, and the reasoning behind the choices.
This is the source for the dashboard's "Machine Learning" tab.

## Goal

Predict PV output from weather, and ultimately let a user place a panel on a map, pick an
orientation, and get an output estimate.

## Installation differences matter for the ML

The three sites are not the same kind of installation, which is important: a model trained
across sites could confuse hardware differences with weather/orientation differences. We have
no panel brand/type/model in the data (only kWp and inverter size), but the configurations
differ clearly:

| site    | kWp   | inverter | DC/AC ratio | arrays                    | peak seen (kW) |
|---------|-------|----------|-------------|---------------------------|----------------|
| house1  | 6.25  | 4.0 kW   | 1.56        | 3 (4 + 1.5 + 0.75), 2 dirs | 5.15          |
| house2  | 2.40  | 2.2 kW   | 1.09        | 1, 1 direction            | 2.33           |
| reactor | 32.70 | 22.0 kW  | 1.49        | 2 (16.35 + 16.35)         | 21.95          |

Why this affects the model:
- **DC/AC ratio** (panel kWp vs inverter kW) differs a lot. house2 (1.09) has an oversized
  inverter, so output scales almost linearly with irradiance. house1 and reactor (~1.5) have
  panels over-sized relative to the inverter, so at high irradiance the inverter **clips**
  (caps) the output. The peaks (5.15 kW for house1's 6.25 kWp) sit well below the panel rating.
- That clipping is a per-site non-linearity caused by hardware, not by weather or orientation.
  A single cross-site model would mistake it for something else.
- house1 has three differently-sized arrays across two directions -- the most mixed profile,
  which is why it is the hardest to predict (R² 0.75 vs reactor 0.86 after the target fixes below).
- The EAN code in the reactor file (541454897100239158) is a Belgian grid connection ID
  (Fluvius/Flanders), identifying the metering point, not the panel type.

Implication: per-site models are fair. A general (map-tool) model should normalize by kWp and,
ideally, account for the DC/AC clipping, or at least flag it as a known limitation.

## Two model tracks

1. **Per-site forecasting** (`train.py`) -- predicts a site's own output from weather + time
   features. Good accuracy, and lag features (previous quarter-hour) push it higher. But lag
   features need a "previous output", which you don't have for a hypothetical new panel, so this
   track does not transfer to the map tool. Kept as a benchmark / forecasting experiment.
2. **Generalizable model** (the map tool) -- trained across all three sites together on weather +
   location + orientation + capacity, predicting output **per kWp** (so it learns efficiency, not
   just size). No lag features. This is what the map tool uses.

## Models tried (per-site track)

- physics (single-feature linear fit on horizontal irradiance) -- the floor to beat
- linear (LinearRegression) -- benchmark, fully interpretable
- forest (RandomForestRegressor) -- robust, non-linear, no scaling needed
- boosting (HistGradientBoostingRegressor) -- often most accurate

Each was evaluated three ways, because this is time-series data:
- time_split -- train oldest 80%, test newest 20% (realistic, no leakage) -> we rank on this
- random -- random 80/20 (looks better but leaks: same-day quarters in train and test).
  For daily scores the random split is over whole days, so a day's quarters never straddle
  train and test.
- ts_cv -- TimeSeriesSplit 5-fold (robust, but early folds have little data -> can go negative)

All models train on quarterly (15-min) data only. Daily figures come from summing the
quarterly model's predictions per calendar day; see the next section for why. The feature
variant reported per model is chosen by ts_cv R² (choosing by the test score itself was mild
selection-on-test leakage), and predictions are clipped to [0, inverter capacity].

## Target handling: the fixes that mattered (July 2026)

An audit of the training data revealed that the biggest accuracy problems were in the target,
not the models. Three changes were validated on the honest time split and adopted:

1. **Missing night quarters are filled with 0** (`features.py`). The houses' PV loggers do not
   report at night, so roughly half of all quarters had no reading and were dropped. A panel at
   night produces exactly 0, so those rows are free, perfectly-labeled training data. This
   roughly doubles the usable rows and teaches the model that nights are zero, which the
   forecast pages depend on. Missing *daytime* quarters stay excluded: house2's daytime gaps sit
   at median 25 W/m² (inverter asleep at dawn/dusk), but house1's sit at median 91 / mean
   183 W/m², i.e. real logging outages at productive hours. Zero-filling those was tested and
   made house1 clearly worse (daily R² 0.58 -> 0.37), so it was rejected.
2. **Clear-sky index feature** (`shortwave_radiation / terrestrial_radiation`, clipped to
   [0, 1.5], 0 at night). It expresses "how cloudy" as one number, independent of season and
   hour. Small but consistent gain on every site.
3. **Daily = aggregated quarterly.** The standalone daily models had only 196-534 samples and
   lost to summing the quarterly model's predictions per day nearly everywhere (reactor daily
   R² 0.32 -> 0.81). They were removed; `results.csv` daily rows now score the aggregation.
   Note the daily targets themselves are slightly undercounted for the houses (median daytime
   coverage ~90% due to logging gaps), so daily metrics reward matching an imperfect total.
4. **Physical clipping.** Predictions are clipped to the inverter capacity (kW x 0.25 = kWh per
   quarter): house1 1.0, house2 0.55, reactor 5.5 kWh. house1 clips for real -- ~5% of its
   readings sit at >=98% of the cap (6.25 kWp on a 4 kW inverter), which is also why purely
   linear models overshoot its sunny middays.

Validated effect (boosting, time_split R², before -> after):

| resolution | house1 | house2 | reactor |
|------------|--------------|--------------|--------------|
| 15-min     | 0.60 -> 0.75 | 0.78 -> 0.84 | 0.86 -> 0.86 |
| daily      | 0.49 -> 0.58 | 0.69 -> 0.77 | 0.32 -> 0.81 |

The reactor's 15-min score is flat because its meter already records nights, so the night fill
adds nothing there. Linear regression gained even more than boosting from the fixes: on the
fair_lag variant it now leads house1 (quarterly 0.78, daily 0.69) -- with nights filled and the
clear-sky index the relationship is close to linear, and the inverter-cap clip patches its one
blind spot. The best model per site (ts_cv): house1 linear, house2 forest, reactor boosting,
all on fair_lag.

The theme/combination experiment tables below predate these fixes, so their absolute scores run
lower; the relative comparisons (which features matter) still hold and are what drove the
feature choices.

The dashboard's forecast pages (Today / This week) run these best models directly on the full
Open-Meteo hourly forecast (same variables, units and UTC timestamps as the training data),
interpolated onto the 15-min training grid; lag features are built from the forecast irradiance
series itself. Only the Predict page keeps the compact irradiance + hour + temperature model,
because its three sliders cannot drive 25 features.

Also tested, **not** adopted:
- Zero-filling low-light daytime gaps (irradiance < 50 W/m²): helps house2 slightly, hurts
  house1 badly (its gaps are real outages, not inverter sleep).
- A pooled cross-site model on per-kWp output: reactor R² dropped 0.86 -> 0.80. The sites'
  per-kWp behavior differs too much (DC/AC ratio, orientation), confirming the per-site choice.
- Hyperparameter tuning of the boosting model: no measurable gain over defaults.

Known limitation: the reactor only spans Dec 2025 - Jun 2026, so every held-out window is an
unseen season. Its daily ts_cv scores stay unstable until a full year of data exists.

A feature-set experiment (`feature_experiment.py`) confirmed the full feature set is fine:
dropping global_tilted_irradiance barely changes scores (it is not a "cheat" feature), and
trimming to fewer features makes things slightly worse. So the scores above are roughly the
ceiling for these weather features; tuning features further is not where the gains are.

## General vs specific (theme) models

We keep the full-feature model as the **general** allrounder, and compare it against models
trained on one theme of features each (`theme_experiment.py`). R² per theme per site (forest,
time_split):

| theme        | features | house1 | house2 | reactor | what it isolates                          |
|--------------|----------|--------|--------|---------|-------------------------------------------|
| general      | 20       | 0.62   | 0.77   | 0.87    | everything (the allrounder we keep)       |
| direction    | 8        | 0.65   | 0.71   | 0.82    | irradiance components + hour of day       |
| time_season  | 6        | 0.20   | 0.17   | 0.68    | hour + day-of-year + month only           |
| weather      | 14       | 0.41   | 0.49   | 0.74    | weather conditions, no time               |
| irradiance   | 6        | 0.19   | 0.24   | 0.65    | radiation measures only                   |

Takeaways (this is the story for the dashboard reasoning section):
- **direction is almost as good as general with far fewer features** (and beats it on house1).
  This is the compact model the map tool wants, since direction is the variable the user picks.
- **irradiance alone is weak** (0.19-0.65). Surprising, but without the hour of day the model
  can't tell how that radiation falls on a tilted, oriented panel. Adding the hour (-> direction
  theme) is what makes irradiance useful, which is itself evidence the orientation matters.
- **time/season alone works for the reactor** (0.68, a large near-flat array whose output tracks
  the sun's daily arc closely) but not for the houses (0.17-0.20, more irregular profiles).

### Combinations (`combo_experiment.py`)

Building on `direction`, we tested adding season and/or temperature. R² per combination
per site (forest, time_split):

| combination      | features | house1 | house2 | reactor |
|------------------|----------|--------|--------|---------|
| general          | 20       | 0.62   | 0.77   | 0.87    |
| direction        | 8        | 0.65   | 0.71   | 0.82    |
| dir_season       | 12       | 0.59   | 0.68   | 0.81    |
| dir_temp         | 10       | 0.60   | 0.74   | 0.87    |
| dir_season_temp  | 14       | 0.59   | 0.76   | 0.87    |
| core_hour        | 3        | 0.54   | 0.62   | 0.77    |

Findings:
- **dir_season_temp matches general with fewer features** (house2 0.76 vs 0.77, reactor 0.87 vs
  0.87). It drops the least useful columns (visibility, wind, weather_code, dew point) for free.
  This is the best compact allrounder.
- **Adding season to direction hurts** (dir_season < direction everywhere). Under a time_split you
  test on a different season than you trained on, so season features mislead the model toward the
  wrong season. Temperature (dir_temp) helps instead -- it's a direct physical driver.
- **core_hour (just shortwave irradiance + hour, 3 features) already reaches 0.54-0.77** -- a good
  minimal demo model.
- Best compact model per site: house1 -> direction (simpler is better), house2 and reactor ->
  dir_season_temp (matches general).

## Key limitation: location does not generalize

The three sites sit within ~10 km of each other (lat 50.81-50.91, long 3.25-3.39, the Bruges/
Kortrijk area). Their weather is nearly identical, so the model never sees variation in location.
Consequences for the map tool:

- **Works**: orientation/direction and capacity (kWp) -- we have real variation in those.
- **Does not work reliably**: placing a marker far away. The model only knows Bruges-region weather,
  so lat/long add almost nothing. The map tool is therefore a regional, zoomed-in proof of concept.
- Honest framing belongs in the dashboard's reasoning section.

A real "anywhere" version would fetch irradiance for the marker's location at prediction time
(e.g. the Open-Meteo API this data came from). Out of scope for now.

## Orientation and tilt estimate

We don't have the panel azimuth/tilt as metadata, so we estimate both from the data. The quick
method (`orientation.py`) reads direction from the balance point of the average daily profile;
the full method (`tilt_fit.py`, `two_plane_fit.py`) matches simulated plane-of-array irradiance
against the measured output and also answers the two-direction question. Method, results and
caveats: see [orientation-tilt.md](orientation-tilt.md).
