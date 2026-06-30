"""
Step 3: visualize the underperforming days found in anomalies.py.

Run: python -m plots.anomaly_plots
"""

import sys
import matplotlib.pyplot as plt
from analysis import SITES
from anomalies import daily_yield_ratio, flag_anomalies
from plots.common import save, SITE_COLORS

sys.stdout.reconfigure(encoding="utf-8")


def plot_yield_ratio_with_anomalies():
    fig, axes = plt.subplots(len(SITES), 1, figsize=(12, 10), sharex=True)
    for ax, name in zip(axes, SITES):
        daily = daily_yield_ratio(name)
        anomalies = flag_anomalies(name)
        ax.plot(daily.index, daily["ratio"], color=SITE_COLORS[name], linewidth=1, label=name)
        if not anomalies.empty:
            ax.scatter(anomalies.index, anomalies["ratio"], color="red", zorder=5, label="anomaly")
        ax.set_ylabel("kWh/kWp per W/m²")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    fig.suptitle("Daily yield ratio with flagged underperforming days")
    fig.tight_layout()
    save(fig, "yield_ratio_anomalies")


if __name__ == "__main__":
    plot_yield_ratio_with_anomalies()
