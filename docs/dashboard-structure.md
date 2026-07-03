# Dashboard structure

Reference of every page and function in `dashboard.py`, for design work.
The app is a Streamlit multipage app with sidebar navigation, grouped into
four sections. All data is per-15-minute PV output and weather for three
sites: House 1, House 2, Reactor.

## Navigation

Sidebar groups and their pages (in order):

| Group | Page | Function |
|---|---|---|
| Start | Overview | `page_overview` |
| Start | Data guide | `page_data_guide` |
| Sites | House 1 | `page_house1` → `render_site` |
| Sites | House 2 | `page_house2` → `render_site` |
| Sites | Reactor | `page_reactor` → `render_site` |
| Sites | Compare | `page_compare` |
| Analysis | Time of day | `page_time_of_day` |
| Analysis | Weather | `page_weather` |
| Analysis | Anomalies | `page_anomalies` |
| Machine learning | Models | `page_ml_models` |
| Machine learning | Predict | `page_predict` |
| Machine learning | Today | `page_today` |
| Machine learning | This week | `page_this_week` |
| Machine learning | Replay a day | `page_replay` |

Every page except Data guide and Today shows a collapsible **Filters** expander
(`filter_controls`): a date-range slider, and on most pages a site multiselect.

## Pages

### Start

**Overview** (`page_overview`)
- Title + intro paragraph.
- One `st.metric` per site: total kWh over the selected range, with a delta vs
  the previous equal-length period.
- "Energy output over time" area chart, with a Day/Week/Month aggregation toggle.

**Data guide** (`page_data_guide`)
- Explains the three raw data sources (PV output, reactor meter, weather) in a table.
- Notes on what is and isn't known about the panels (no brand/model, EAN code meaning).
- Table of the weather predictor columns and their meaning.
- "Sources & attribution": Open-Meteo (CC BY 4.0) and the PV data collection.

### Sites

**House 1 / House 2 / Reactor** (`render_site`)
- Four metrics: Installed (kWp), Inverter (kW), DC/AC ratio, Orientation (fitted
  facing + tilt from `tilt_results.csv`). All have explanatory tooltips.
- "Daily output" line chart with 7-day average.
- "Average day shape": mean output per 15-min slot across sunny days, showing the
  typical production curve, with a peak-hour + facing callout underneath.

**Compare** (`page_compare`)
- "Compare by" selector with four views:
  - Specific yield (kWh/kWp) — grouped monthly bar chart, size-normalized.
  - Output over time — line chart, Day/Week/Month toggle.
  - Average day shape — all sites overlaid, each normalized to its own peak so
    only the timing differs (reactor peaks earliest, due south).
  - Characteristics table — kWp, inverter, DC/AC, orientation, estimated tilt,
    mean daily kWh, mean kWh/kWp.

### Analysis

**Time of day** (`page_time_of_day`)
- Single site or all sites selector.
- Line chart of average output by hour of day.

**Weather** (`page_weather`)
- "Irradiance vs output" scatter, colored by month, with a corr(energy, irradiance) metric.
- "Temperature effect at fixed irradiance": bar chart of mean kWh/kWp per temperature bin,
  with an irradiance-band slider. Includes a note that the upward trend is a seasonal artefact.

**Anomalies** (`page_anomalies`)
- Two sliders: anomaly threshold (z-score) and min daily irradiance.
- Yield-ratio line chart per site with flagged underperforming days marked.
- Table of flagged day-site combinations.
- Note on the shared bad days (drizzle/fog, not snow).

### Machine learning

**Models** (`page_ml_models`)
- "Why per-site models": clipping curve (max output vs irradiance, normalized per site).
- "Model comparison": bar chart of R² per model per site (time_split), with best-model metrics.
- "What made the models better": the target fixes (night fill, clear-sky index, inverter cap,
  daily = summed quarterly) with the before/after R² table.
- "Physics baseline": irradiance-only fit vs best ML model per site.
- "What lag features add": fair vs fair_lag comparison table.
- "General vs specific models": pivot table comparing feature-set variants by R².
- "Panel orientation and tilt": compass polar chart (direction = angle, tilt = radius,
  shaded arcs = plausible ranges), results table with per-column tooltips, the
  two-direction verdict table (`two_plane_results.csv`), and an expander with a
  plain-language explanation of the method plus the balance-point cross-check.
  Method details: [orientation-tilt.md](orientation-tilt.md).

**Predict** (`page_predict`)
- Three sliders: irradiance, hour of day, temperature.
- Predicted daily curve for all three sites, with a marker at the selected hour.
- Per-site metric of predicted output at that point.
- "Irradiance sweep": output vs sun strength at the fixed hour/temperature.
- "Simulated full day": pick a sky condition (clear/cloudy/overcast) for a realistic
  irradiance profile, shows predicted curve and estimated daily total per site.

**Today** (`page_today`)
- Button to fetch today's full weather picture live from the Open-Meteo API
  (friendly empty-state card before the first fetch).
- Runs each site's best model (ts_cv winner from results.csv) on the 15-min grid;
  the Predict page keeps the compact slider model.
- Peak irradiance and average temperature metrics.
- Predicted output curve for all sites, with actual output overlaid (dashed) if the
  dataset already contains today.
- Estimated daily total per site as metric cards, with Open-Meteo CC BY 4.0 credit.

**This week** (`page_this_week`)
- Button to fetch the 7-day forecast (full weather set) from Open-Meteo.
- Weather outlook cards (icon, label, min/max temperature per day).
- Runs each site's best model on the 15-min grid: 7-day totals, daily bar chart,
  and the predicted 15-min output curve across the week.

**Replay a day** (`page_replay`)
- Date picker over the dataset range; runs each site's best model on that day's
  recorded weather (`get_day_backtest`).
- Chart of predicted (solid) vs actual (dashed) 15-min output plus irradiance.
- Day-totals table: actual, predicted, error in kWh and %, missing quarters.
  Totals only cover quarters with a known actual, so logging gaps stay fair.
- Caveat shown on the page: models are trained on the full history including the
  chosen day, so this is a reproduction check, not a blind forecast.

## Functions

### Data loading (cached with `@st.cache_data`)
- `get_daily_energy(site)` — daily energy series per site.
- `get_joined(site)` — PV joined with weather, per 15 min.
- `get_yield_ratio(site, min_rad)` — daily yield ratio for anomaly detection.
- `get_anomalies(site, z, min_rad)` — days flagged below the z-score threshold.
- `get_ml_csv(name)` — reads a machine-learning results CSV.
- `get_orientations()` — estimated azimuth/facing per site.
- `get_clipping_curve()` — max output per irradiance bin, normalized per site.
- `get_daily_profile(site)` — average output per hour of day over sunny days.

### Prediction — compact model (Predict page sliders)
- `_compact_model(site)` — small RandomForest on irradiance + hour (sin/cos) + temperature,
  cached with `@st.cache_resource`. Used only where the user drives 3 sliders.
- `predict_compact(site, irradiance, hour, temp)` — single prediction.
- `predict_sweep(site, hour, temp)` — predictions across all irradiance levels.
- `predict_day(site, irr_profile, temp)` — 24-hour prediction from an irradiance profile.
- `_irr_profile(peak, ...)` — synthetic Gaussian irradiance curve for the day simulator.
- `_clip_pred(site, value)` — clips any prediction to [0, inverter capacity per quarter].

### Prediction — best model (Today / This week pages)
- `FORECAST_VARS` — Open-Meteo hourly variable -> training column name (full fair set).
- `fetch_forecast(days)` — full hourly weather set + daily summary from Open-Meteo (UTC).
- `_to_quarter_grid(hourly)` — interpolates the hourly forecast onto the 15-min training grid.
- `_forecast_model(site)` — the site's best model type (ts_cv winner in results.csv) trained
  on the fair_lag features, cached with `@st.cache_resource`.
- `predict_forecast(site, hourly)` — 15-min prediction series (kWh/quarter) over the window.

### Layout and helpers
- `render_site(name)` — shared body for the three site pages.
- `filter_controls(key, with_sites)` — the Filters expander (date range + site multiselect);
  the collapsed header shows the active date range.
- `_fetch_empty_state(message)` — placeholder card on Today/This week before the first fetch.
- Sidebar footer caption with the dataset summary and Open-Meteo attribution,
  plus a QR code (`assets/qrcode_ai-brugge-team1.png`) linking to the deployed app.
- `in_range(index, date_range)` / `date_bounds()` — date filtering helpers.
- `period_delta(series, date_range)` — percentage change vs the previous equal-length period.

## Styling constants
- `SITE_COLORS` — blue (house1), orange (house2), green (reactor).
- `SITE_FILL` — translucent versions for area fills.
- `SITE_INFO` — per-site label, inverter kW, DC/AC ratio, array layout.
- `PLOTLY_LAYOUT` — shared Plotly theme (white template, Arial, transparent legend).
- Page config: wide layout, title "PV Dashboard — AI Brugge Team 1".
