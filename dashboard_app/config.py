SITE_COLORS = {"house1": "#1f77b4", "house2": "#ff7f0e", "reactor": "#2ca02c"}
SITE_FILL = {"house1": "rgba(31,119,180,0.12)", "house2": "rgba(255,127,14,0.12)", "reactor": "rgba(44,160,44,0.12)"}
SITE_DOT = {"house1": "🔵", "house2": "🟠", "reactor": "🟢"}  # matches SITE_COLORS for text labels

# installation metadata, used by the site pages and Compare
SITE_INFO = {
    "house1": {"label": "House 1", "inverter_kw": 4.0, "dcac": 1.56,
               "arrays": "3 arrays (4 + 1.5 + 0.75 kWp), 2 directions"},
    "house2": {"label": "House 2", "inverter_kw": 2.2, "dcac": 1.09,
               "arrays": "1 array (2.4 kWp), 1 direction"},
    "reactor": {"label": "Reactor", "inverter_kw": 22.0, "dcac": 1.49,
                "arrays": "2 arrays (16.35 + 16.35 kWp)"},
}

# per-site nuance for the fitted orientation (see Models page for the method)
TILT_NOTE = {
    "house1": "The house has panels in 2 directions, but they are too similar to separate from the "
              "output data, so this is the effective combined plane.",
    "house2": "Single direction, confirmed by the two-plane check collapsing to one plane.",
    "reactor": "A very low tilt plus a west-leaning day shape is the fingerprint of an east-west "
               "'tent' pair: the two faces look nearly identical to the sun, so only their "
               "combination is identifiable.",
}

# WMO weather codes present in this dataset, grouped for readability
WMO_LABELS = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Rain", 65: "Heavy rain",
    73: "Moderate snow", 75: "Heavy snow",
}

RESAMPLE_RULES = {"Day": "D", "Week": "W", "Month": "MS"}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(t=60, b=40),
)
