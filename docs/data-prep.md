# Data preparation notes

How the raw data is cleaned into `data/clean/`. The single script that does this is
`prep_data.py`; this document explains the why behind it.

## Pipeline

```
data/raw/   ->  prep_data.py  ->  data/clean/
```

- `data/raw/` holds the original files, untouched (read-only).
- `prep_data.py` is the only code that reads raw data. Run it once (or when raw data changes).
- `data/clean/` holds ML-ready CSVs: per site `<site>_quarterly.csv` and `<site>_daily.csv`.

Energy is normalized to **kWh** for every site. Missing readings stay empty (NaN), never zero.

## Raw data formats

The three sources do not share a format. Each difference is handled in `prep_data.py`; this table
is the reference for why the cleaning does what it does.

| Aspect            | PV house1/house2        | Reactor meter                     | Weather files (all)            |
|-------------------|-------------------------|-----------------------------------|--------------------------------|
| Field separator   | comma `,`               | semicolon `;`                     | comma `,`                      |
| Decimal separator | dot `.` (`0.0`)         | comma `,` (`0,000`)              | dot `.`                        |
| Date format       | `2025-04-27 06:15:00`   | `01-12-2025` + separate time col  | `2025-01-01T00:00` (ISO + T)   |
| Header row        | row 1                   | row 1                             | row 4 (2 metadata rows + blank)|
| BOM               | no                      | yes (UTF-8 BOM)                   | no                             |
| Energy unit       | Wh                      | kWh                               | n/a                            |
| Structure         | 1 row per timestamp     | 3 rows/timestamp (Actief/Cap/Ind) | 1 row per timestamp            |
| Quoting           | none                    | EAN-code as `="..."`              | none                           |

## How each difference is handled

- **Reactor semicolons**: `read_csv(sep=";")`.
- **Reactor comma decimals**: `str.replace(",", ".")` then `to_numeric`. Verified: raw `0,025` -> clean `0.025`,
  and the max quarter-hour value (~5.5 kWh) is realistic for 32.7 kWp, so no factor-1000 mistake.
- **Reactor date** is DD-MM-YYYY in a column separate from the time; joined and parsed with `dayfirst=True`.
- **Reactor registers**: 3 rows per timestamp (Actief / Capacitief / Inductief). We keep only
  `Productie Actief` (active production).
- **Reactor BOM** is stripped automatically by pandas. The `="..."` EAN quoting sits in a column we drop.
- **Weather files** have 2 metadata rows + a blank line before the header -> `skiprows=3`.
- **Weather column order differs** between house1/2 and reactor (e.g. global_tilted_irradiance and
  terrestrial_radiation are swapped). We select columns by name in a fixed order, so clean output is uniform.
- **Energy unit**: house1/2 are divided by 1000 (Wh -> kWh); reactor is already kWh.

## Missing values, not zeros

- Reactor `Geen gegevens` (empty Volume) means "no reading", not 0 kWh. Kept as NaN, written as an empty
  field in the clean CSV. The first rows of the reactor dataset (early 2025-12-01) are such gaps -- real, not a bug.
- house1 PV data starts 2025-04-27 while its weather starts 2025-01-01, so the clean quarterly file has
  ~4 months of NaN energy with valid weather at the start. This is correct: the join is complete, energy is
  simply absent before the panels started reporting.
- Daily aggregation uses `sum(min_count=1)`, so a day with no readings stays NaN instead of summing to 0.

## Clean output reference

| File                      | Rows  | Notes                                            |
|---------------------------|-------|--------------------------------------------------|
| `house1_quarterly.csv`    | 51744 | energy NaN before 2025-04-27 (PV starts late)    |
| `house1_daily.csv`        | 539   | 418 days with energy                             |
| `house2_quarterly.csv`    | 51744 | most complete site                               |
| `house2_daily.csv`        | 539   | 534 days with energy                             |
| `reactor_quarterly.csv`   | 19680 | 9 NaN days ("Geen gegevens") kept                |
| `reactor_daily.csv`       | 205   | 196 days with energy                             |

Each daily file adds a `radiation_sum` column (daily total shortwave radiation), which the anomaly
analysis uses; the other weather columns are daily means.
