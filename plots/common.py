"""
Shared plotting setup, used by every plot script in this folder
(daily_output_plots.py, anomaly_plots.py, and later forecast_plots.py).

Plots are saved to plots/output/ instead of shown interactively, so they can
be reviewed as files and reused in reports.
"""

from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SITE_COLORS = {
    "house1": "tab:blue",
    "house2": "tab:orange",
    "reactor": "tab:green",
}


def save(fig, name: str):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
