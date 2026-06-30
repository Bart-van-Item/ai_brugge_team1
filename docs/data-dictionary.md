# Data dictionary

What every dataset and column means. For how the raw data is cleaned into `data/clean/`, see
`data-prep.md`. For why installation differences matter to the modelling, see `machine-learning.md`.

## Sites

| site    | folder              | inverter | kWp   | arrays                         | directions |
|---------|---------------------|----------|-------|--------------------------------|------------|
| house1  | `data/raw/house1/`  | 4 kW     | 6.25  | 3 (4 + 1.5 + 0.75 kWp)         | 2          |
| house2  | `data/raw/house2/`  | 2.2 kW   | 2.40  | 1 (2.4 kWp)                    | 1          |
| reactor | `data/raw/reactor/` | 22 kW    | 32.70 | 2 (16.35 + 16.35 kWp)          | (south)    |

Source for these numbers: the `.txt` file in each raw site folder.

## What we know (and don't) about the panels

- **No panel brand, model, or type** is recorded anywhere in the data. We only have installed
  capacity (kWp), inverter size, and array layout.
- Installations clearly differ in scale, array count, and DC/AC ratio (see `machine-learning.md`).
- The **EAN code** in the reactor meter file (`541454897100239158`) is a Belgian grid connection
  identifier (EAN-18; the `5414549` prefix is Fluvius, the Flanders grid operator). It identifies
  the **metering point / connection**, not the panel. The houses have no EAN in their PV files
  (those come from the inverter export, not the grid meter).

## Dataset 1: PV output (house1, house2)

Files: `PV_data_house{1,2}_{2025,2026}.csv`. Comma-separated, dot decimals, one row per 15 minutes.

| column      | type     | unit | meaning                                           |
|-------------|----------|------|---------------------------------------------------|
| `datetime`  | datetime | -    | start of the 15-minute interval (`YYYY-MM-DD HH:MM:SS`) |
| `energy_wh` | float    | Wh   | energy produced in that 15-minute interval        |

Note: in the clean data this is converted to **kWh** and the column is renamed `energy_kwh`.

## Dataset 2: Reactor meter

File: `Historiek_submeting_elektriciteit_..._kwartiertotalen.csv`. Semicolon-separated, **comma
decimals**, UTF-8 BOM. **Three rows per timestamp** (one per register). This is a grid-meter
export, not an inverter export.

| column            | type     | unit       | meaning                                                  |
|-------------------|----------|------------|----------------------------------------------------------|
| `Van (datum)`     | date     | -          | interval start date (`DD-MM-YYYY`)                       |
| `Van (tijdstip)`  | time     | -          | interval start time                                      |
| `Tot (datum)`     | date     | -          | interval end date                                        |
| `Tot (tijdstip)`  | time     | -          | interval end time (15 min after start)                  |
| `EAN-code`        | string   | -          | grid connection ID, quoted as `="..."` (same every row) |
| `Meter`           | string   | -          | meter id (empty in this export)                          |
| `Metertype`       | string   | -          | `AMR-meter` (Automatic Meter Reading, i.e. smart meter)  |
| `Register`        | string   | -          | which quantity: see below                                |
| `Volume`          | float    | kWh/kVArh  | the measured value; **empty = no reading** (not zero)   |
| `Eenheid`         | string   | -          | unit: `kWh` for active, `kVArh` for reactive             |
| `Validatiestatus` | string   | -          | `Gevalideerd` (validated) or `Geen gegevens` (no data)   |
| `Omschrijving`    | string   | -          | `KWE_A_AP2 - Submeting 4 | Productie` (a production submeter) |

Register values:
- `Productie Actief` (kWh) -- **active production**, the actual energy generated. This is the one
  we use as `energy_kwh`.
- `Productie Capacitief` (kVArh) -- capacitive reactive power. Not real energy output; ignored.
- `Productie Inductief` (kVArh) -- inductive reactive power. Ignored.

`Validatiestatus = Geen gegevens` always coincides with an empty `Volume`. Counts in this export:
56436 `Gevalideerd`, 2592 `Geen gegevens`.

## Dataset 3: Weather (all sites)

Files: `weer_data_house{1,2}.csv`, `weer_reactor.csv`. Source: Open-Meteo. Comma-separated, dot
decimals. **Rows 1-2 are metadata** (the site's coordinates), row 3 is blank, row 4 is the header,
data starts at row 5. One row per 15 minutes. Times are **UTC** (timezone GMT, offset 0).

Metadata rows (1-2):

| column                  | meaning                              |
|-------------------------|--------------------------------------|
| `latitude`, `longitude` | site coordinates (Bruges/Kortrijk region) |
| `elevation`             | metres above sea level               |
| `utc_offset_seconds`    | 0 (data is in UTC)                   |
| `timezone`, `timezone_abbreviation` | GMT                     |

Data columns (row 4 onward). **Column order differs between house and reactor files**; the clean
data fixes this to one order.

| column                            | unit     | meaning                                              |
|-----------------------------------|----------|------------------------------------------------------|
| `time`                            | datetime | interval start (`YYYY-MM-DDTHH:MM`); renamed `datetime` in clean |
| `temperature_2m`                  | °C       | air temperature at 2 m                               |
| `relative_humidity_2m`            | %        | relative humidity at 2 m                             |
| `dew_point_2m`                    | °C       | dew point at 2 m                                     |
| `apparent_temperature`           | °C       | "feels like" temperature                             |
| `shortwave_radiation`             | W/m²     | global horizontal irradiance (total sun on a flat surface) -- main driver |
| `direct_radiation`                | W/m²     | direct beam component on horizontal                  |
| `diffuse_radiation`               | W/m²     | scattered/sky component on horizontal                |
| `direct_normal_irradiance`        | W/m²     | direct beam perpendicular to the sun                 |
| `global_tilted_irradiance`        | W/m²     | irradiance on a tilted plane (closest to what a panel sees) |
| `terrestrial_radiation`           | W/m²     | top-of-atmosphere radiation (clear-sky theoretical max) |
| `weather_code`                    | WMO code | weather type (0 clear ... 45 fog, 51-55 drizzle, 61-65 rain, 71-75 snow) |
| `wind_speed_10m`                  | km/h     | wind speed at 10 m                                   |
| `visibility`                      | m        | horizontal visibility                                |
| `is_day`                          | 0/1      | 1 during daylight, 0 at night                        |

## Clean data: `data/clean/`

Produced by `prep_data.py`. Per site, two files.

**`<site>_quarterly.csv`** -- PV energy joined with weather, per 15 minutes:
- `datetime` (index), `energy_kwh` (target), then all weather columns above in a fixed order.
- `energy_kwh` is empty (NaN) where there was no reading.

**`<site>_daily.csv`** -- aggregated per day:
- `date` (index), `energy_kwh` (daily sum), weather columns as daily **means**, plus
  `radiation_sum` (daily total of `shortwave_radiation`, used by the anomaly analysis).
