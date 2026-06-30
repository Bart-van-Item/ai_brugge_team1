"""
Step 3: dashboard visuals for daily/monthly output and a typical-day power curve.

Run: python -m plots.daily_output_plots
"""

import sys
import matplotlib.pyplot as plt
from analysis import SITES, daily_energy
from plots.common import save, SITE_COLORS

sys.stdout.reconfigure(encoding="utf-8")

TYPICAL_DAY = "2025-06-15"  # clear summer day, used for the intraday power curve


def plot_daily_energy():
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in SITES:
        daily = daily_energy(name)
        ax.plot(daily.index, daily.values, label=name, color=SITE_COLORS[name], linewidth=1)
    ax.set_title("Daily energy output per site")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "daily_energy")


def plot_monthly_energy():
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    offsets = [-width, 0, width]
    monthly_by_site = {}
    for name in SITES:
        daily = daily_energy(name)
        monthly_by_site[name] = daily.resample("MS").sum(min_count=1)

    months = sorted(set().union(*[s.index for s in monthly_by_site.values()]))
    x = range(len(months))
    for offset, (name, monthly) in zip(offsets, monthly_by_site.items()):
        values = [monthly.get(m, float("nan")) for m in months]
        ax.bar([i + offset for i in x], values, width=width, label=name, color=SITE_COLORS[name])

    ax.set_xticks(list(x))
    ax.set_xticklabels([m.strftime("%Y-%m") for m in months], rotation=45, ha="right")
    ax.set_title("Monthly energy output per site")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save(fig, "monthly_energy")


def plot_typical_day_power_curve():
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in SITES:
        site = SITES[name]
        pv_col = site["pv_unit"]
        try:
            day = site["pv"].loc[TYPICAL_DAY][pv_col]
        except KeyError:
            continue
        if pv_col == "energy_kwh":
            day = day * 1000  # -> Wh, so all sites share a unit
        times = day.index.strftime("%H:%M")
        ax.plot(times, day.values, label=name, color=SITE_COLORS[name])

    ax.set_title(f"Intraday power curve — {TYPICAL_DAY}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy per 15 min (Wh)")
    ax.set_xticks(ax.get_xticks()[::4])
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "typical_day_power_curve")


if __name__ == "__main__":
    plot_daily_energy()
    plot_monthly_energy()
    plot_typical_day_power_curve()
