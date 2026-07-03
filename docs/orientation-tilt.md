# Panel orientation and tilt estimation

How we estimate each site's panel direction (azimuth) and angle (tilt) from the
output and weather data alone, with no metadata and no site visit. This covers
assignment goal 3, including the multi-direction question. Code:
`machine-learning/tilt_fit.py` (single plane, two-stage) and
`machine-learning/two_plane_fit.py` (two-direction decomposition). Results land
in `tilt_results.csv` and `two_plane_results.csv`, shown on the dashboard's
Models page under "Panel orientation and tilt".

## Results

| Site | Facing | Azimuth | Plausible directions | Tilt | Plausible tilts | Midday fit R² |
|---|---|---|---|---|---|---|
| house1 | SW | 236° | 232-248° | 56° | 46-60° | 0.40 |
| house2 | WSW | 252° | 248-260° | 60° | 56-60° | 0.47 |
| reactor | W | 264° | 248-280° | 16° | 0-28° | 0.84 |

Reading guide: azimuth is degrees from north (90 = east, 180 = south,
270 = west). The "plausible" columns are the ridge, meaning every candidate
that explains the data within 0.01 R² of the best one. A narrow ridge means
the parameter is well determined.

Interpretation per site:

- **All three sites face west of south.** The direction signal is strong and
  is independently confirmed by morning-only fits and by the simpler
  balance-point method (`orientation.py`), so it is not an artefact.
- **Reactor: nearly flat, east-west tent.** The fitted tilt is very low (every
  value in the ridge is a low tilt) and the fit quality is high. A very low
  effective tilt combined with a west-leaning daily shape is the fingerprint
  of two opposite low-tilt rows, which matches the known layout (2 equal
  arrays, tent mounting).
- **Houses: steep, roughly 45-60°, read as a range.** Their tilt signal
  competes with the systematic morning deficit (below), so we report "steep"
  with an honest band rather than an exact figure.

## Method: two-stage single-plane fit

The weather data carries the irradiance components (direct-normal, diffuse,
global horizontal), so for any candidate plane we can compute what it would
receive (plane-of-array transposition, isotropic sky model) and grid-search
which plane best explains the measured output.

1. **Solar position** per 15-minute interval midpoint at each site's own
   lat/lon (UTC): Cooper declination, Spencer equation of time, then zenith
   and azimuth from standard formulas. Sanity-checked: solar noon comes out
   at ~11:47 UTC, correct for longitude 3.3°E.
2. **Sample selection.** Only clear, unclipped daylight quarters with a real
   reading: sun elevation above 15°, clearness index above 0.55, output below
   98% of the inverter cap. This removes clouds, horizon effects and the
   inverter ceiling, which would all blur the geometry signal.
3. **Stage A, azimuth from the daily shape.** Grid over tilt 0-60° and azimuth
   60-300°, scored by explained variance of the zero-intercept fit
   `output = k * POA`. The daily shape (morning vs afternoon weight) pins the
   azimuth; a morning-only refit confirms it.
4. **Stage B, tilt from the seasonal midday arc.** Around solar noon (sun
   azimuth 160-200°) the sun's elevation sweeps from 16° (winter) to 62°
   (summer). How midday output scales along that arc is almost purely a
   function of tilt: a steep panel does relatively well at winter noon and
   poorly at summer noon, a flat panel the opposite. Tilt is scanned at the
   stage-A azimuth on the midday subset only.

Why two stages: all three sites share a systematic morning deficit / evening
surplus versus any fixed plane (haze, dew or weather-model bias; a PVWatts
temperature correction was tested and does not remove it). A naive full-day
fit lets that systematic drag the tilt steep and the azimuth west. Splitting
the estimation gives each parameter the data slice that identifies it most
cleanly, and the bias caveat is reported instead of hidden.

## Two-direction separation

House 1 (3 arrays, 2 directions) and the reactor (2 arrays, tent) are known to
have panels facing two ways, so a single plane can only be the effective mix.
`two_plane_fit.py` models the output as a non-negative combination of two
planes:

    output ~ k1 * POA(tilt1, azimuth1) + k2 * POA(tilt2, azimuth2)

Every pair of candidate planes is scored (tilts and azimuths independent, so
tent pairs are included but not assumed). The two-variable non-negative least
squares has a closed form via the Gram matrix, so all pairs are evaluated
exactly; when the unconstrained solution goes negative the answer collapses to
the better single plane.

| Site | One plane R² | Two planes R² | Gain |
|---|---|---|---|
| house1 | 0.549 | 0.550 | +0.000 |
| house2 | 0.771 | 0.771 | +0.000 |
| reactor | 0.753 | 0.753 | +0.000 |

Verdict: **the data does not support a separable second direction anywhere.**
House 2, the known single-direction site, is the built-in control and
correctly collapses to one plane. The reason the true two-direction sites do
not split: panels at low tilt facing opposite ways see almost the same sky,
so their sum is mathematically near-indistinguishable from one nearly-flat
panel (the candidate planes are collinear). That is a property of the physics,
not a bug, and it is itself the answer to the assignment's separation
question: we built the method, validated it on the control, and showed the
output data contains no separable signal, while the tent layout is still
identifiable by its fingerprint (very low midday tilt plus west-leaning
daily shape).

## Cross-check: the balance-point method

`orientation.py` is the independent quick method: take the centre of mass of
the average production day (sunny days only) and map its offset from solar
noon (~11:47 UTC at 3.3°E) to an azimuth at 15° per hour. After fixing an
earlier bug (it used 13:00 UTC as solar noon, biasing every azimuth about 19°
south), it agrees with the full fit in direction: house1 219°, house2 218°,
reactor 200°, all west of south. It understates how far west, because the
centre of mass is damped by the fixed day length, so the full fit's numbers
are the ones we report.

## Bonus finding: house2 produces too much for its label

Specific-yield arithmetic, all from house2's own 2025 data (PR 0.85 assumed,
POA sums computed with the same transposition code):

| Scenario | POA 2025 | Implied specific yield |
|---|---|---|
| Optimal placement (south, 35°) | 1,331 kWh/m² | ~1,132 kWh/kWp |
| Fitted plane (WSW 252°, 60°) | 1,072 kWh/m² | ~911 kWh/kWp |
| Flat | 1,169 kWh/m² | ~994 kWh/kWp |

House2's actual 2025 yield is 2,670 kWh, which at the labelled 2.4 kWp is
1,113 kWh/kWp: essentially the optimal-south figure, and impossible for any
west-of-south plane, let alone a steep one. Either the real capacity is
larger than the metadata says (2,670 / 911 implies roughly 2.9 kWp) or the
2.4 kWp label is otherwise off. Supporting clue: house2 sometimes hits its
2.2 kW inverter cap, which a true 2.4 kWp array facing steep WSW would
almost never do.

## Plain-language version (the story for the demo)

A solar panel produces the most when it points straight at the sun, so the
shape of its production over the day and over the seasons betrays which way
it points. Where the sun is at any moment is pure math, like knowing where
the hands of a clock are. So we built thousands of virtual panels (every
direction, every angle), computed what each would have produced every 15
minutes, and kept the one that matches the real meter best.

The direction comes from the daily rhythm: east panels have their best hours
in the morning, west panels in the afternoon. All three sites are
afternoon-heavy, so they face west of south. The angle comes from the
seasons: at noon the winter sun hangs low and the summer sun high, and a
steep panel loves winter noons while a flat panel loves summer noons. The
reactor behaves like a nearly flat panel; the houses behave like steep ones.

For roofs with panels both ways we also mixed two virtual panels and let the
math pick the blend. The mix never beat a single panel, which is itself an
answer: both faces of those roofs are so alike to the sun that only their
combination is visible in the data.

## Reproducing

```
python machine-learning/tilt_fit.py        # writes tilt_results.csv
python machine-learning/two_plane_fit.py   # writes two_plane_results.csv
python machine-learning/orientation.py     # prints the cross-check
```
