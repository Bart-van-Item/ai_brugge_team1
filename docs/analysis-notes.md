# Analysis notes

Observations and open questions from the exploratory analysis. Data-prep specifics live in
`data-prep.md`.

## Open questions

- **Temperature appears to increase yield instead of decreasing it** (`weather_correlation.py`, temperature_effect).
  With irradiance fixed at 400-600 W/m², average kWh/kWp output rises with temperature for all three sites.
  This is physically unexpected (panels normally lose efficiency at higher panel temperature).
  Likely cause: season/sun angle isn't fully decoupled from temperature, even with irradiance fixed to a band.
  For example, a spring day at 500 W/m² around 10am has a different sun angle than a summer day at 500 W/m²
  early morning or late evening.
  To refine later: filter on a fixed time around noon, or factor in sun angle/is_day.

## Anomalies (underperforming days, `anomalies.py`)

- Days flagged as underperforming at **multiple sites at once**: **2026-01-10, 2025-12-23, 2025-11-20**.
  This points to a shared, external cause rather than a site-specific issue like soiling or a faulty string,
  which would be expected on random, non-overlapping days.
  **Verified**: WMO weather codes on these days are 0-3 (clear to overcast), 51/53/55 (drizzle), 61/63/65 (rain),
  and 45 (fog) -- no snow codes (71-77). Average relative humidity was 84-95% on all three days at all sites,
  consistent with persistent drizzle/fog/heavy overcast. Likely explanation: under hazy/drizzly conditions,
  light is diffuse rather than direct, and `shortwave_radiation` (global horizontal irradiance) may not capture
  the same yield-per-W/m² as on clear days, since panel response to diffuse vs. direct light differs.
- Site-specific anomalies (only at 1 site, not the others) are more interesting to investigate further, since
  those do point to a local issue (shading, soiling, technical fault). Still to identify which days those are
  relative to the shared list above.
