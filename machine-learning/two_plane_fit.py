"""
Decompose a site's output into two panel directions.

house1 and the reactor have arrays in two directions. A single-plane fit
(tilt_fit.py) can only return the effective mix; here we model the output as a
non-negative combination of two planes:

    output ~ k1 * POA(tilt1, azimuth1) + k2 * POA(tilt2, azimuth2)

Every pair of candidate planes is scored (tilts and azimuths independent, so
tent pairs like east+west are included but not assumed). The two-variable
non-negative least squares has a closed form via the Gram matrix, so all pairs
are evaluated exactly: if the unconstrained solution goes negative, the best
answer is one of the single planes, which is how a one-direction site (house2,
the control) collapses to k2 = 0.

Reported per site: the plane pair, each direction's share of production, the
explained variance, and the gain over the single-plane fit. A gain of ~0 means
the data does not support a second direction. Writes two_plane_results.csv.

Run: python machine-learning/two_plane_fit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tilt_fit import (clear_samples, poa_irradiance, facing_label,  # noqa: E402
                      WEATHER_FILES, TARGET)

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_PATH = HERE / "results" / "two_plane_results.csv"

TILT_GRID = np.arange(4, 64, 4)          # 0 excluded: a flat pair is one plane
AZIMUTH_GRID = np.arange(60, 304, 8)     # degrees from north: 90=E, 180=S, 270=W


def candidate_planes(df, zenith, sun_az):
    """POA matrix (n_samples x n_planes) for every (tilt, azimuth) candidate."""
    planes = [(t, a) for t in TILT_GRID for a in AZIMUTH_GRID]
    poa = np.column_stack([poa_irradiance(df, zenith, sun_az, t, a) for t, a in planes])
    return planes, poa


def fit_site(site: str) -> dict:
    df, zenith, sun_az = clear_samples(site)
    y = df[TARGET].to_numpy()
    planes, P = candidate_planes(df, zenith, sun_az)

    # Gram matrix form: for any pair (i, j) the least-squares fit follows from
    # G = P'P and b = P'y, no per-pair regression needed.
    G = P.T @ P
    b = P.T @ y
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_y = np.sum(y * y)

    diag = np.diag(G)
    # single-plane residuals (k = b/G_ii clamped at 0)
    k_single = np.clip(b / diag, 0, None)
    ss_single = ss_y - k_single * b
    best_single = float(1 - ss_single.min() / ss_tot)

    G11 = diag[:, None]
    G22 = diag[None, :]
    G12 = G
    b1 = b[:, None]
    b2 = b[None, :]
    det = G11 * G22 - G12 ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        k1 = (b1 * G22 - b2 * G12) / det
        k2 = (b2 * G11 - b1 * G12) / det
    valid = (k1 > 0) & (k2 > 0) & (det > 1e-9)
    ss_pair = np.where(valid, ss_y - (k1 * b1 + k2 * b2), np.inf)
    # a pair is never worse than its best member (boundary solution)
    ss_bound = np.minimum(ss_single[:, None], ss_single[None, :])
    ss_pair = np.minimum(ss_pair, ss_bound)

    i, j = np.unravel_index(np.argmin(ss_pair), ss_pair.shape)
    score = float(1 - ss_pair[i, j] / ss_tot)
    if valid[i, j]:
        w1, w2 = float(k1[i, j]), float(k2[i, j])
    else:  # collapsed to the better single plane
        first = ss_single[i] <= ss_single[j]
        w1, w2 = (float(k_single[i]), 0.0) if first else (0.0, float(k_single[j]))

    e1 = w1 * P[:, i].sum()
    e2 = w2 * P[:, j].sum()
    share1 = e1 / (e1 + e2) if (e1 + e2) > 0 else 1.0
    (t1, a1), (t2, a2) = planes[i], planes[j]
    if share1 < 0.5:
        (t1, a1), (t2, a2), share1 = (t2, a2), (t1, a1), 1 - share1

    return {
        "site": site,
        "tilt_1": int(t1), "azimuth_1": int(a1), "facing_1": facing_label(a1),
        "share_1": round(share1, 2),
        "tilt_2": int(t2), "azimuth_2": int(a2), "facing_2": facing_label(a2),
        "share_2": round(1 - share1, 2),
        "r2_two_plane": round(score, 4), "r2_single_plane": round(best_single, 4),
        "gain": round(score - best_single, 4), "n_samples": len(df),
    }


def main():
    rows = []
    for site in WEATHER_FILES:
        row = fit_site(site)
        rows.append(row)
        print(f"{site}: {row['facing_1']} {row['azimuth_1']}°/{row['tilt_1']}° "
              f"({row['share_1']:.0%}) + {row['facing_2']} {row['azimuth_2']}°/"
              f"{row['tilt_2']}° ({row['share_2']:.0%})  "
              f"R²={row['r2_two_plane']:.3f}, single {row['r2_single_plane']:.3f}, "
              f"gain {row['gain']:+.3f}")
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
