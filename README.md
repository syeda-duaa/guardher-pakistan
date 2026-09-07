# GuardHer Pakistan 🛡️

**AI-Powered Route Safety Intelligence for Women Commuters**

GuardHer Pakistan is a Streamlit web app that helps women plan safer walking/commuting routes in Pakistani cities. Given a start point, a destination, and a time of travel, it scores multiple route options using a machine learning model trained on incident reports and safety-relevant points of interest (police stations, hospitals, bus stops), then recommends the safest one.

> ⚠️ **Disclaimer:** This tool is for decision support only and does not guarantee safety. In an emergency, contact **Police: 15** | **Women Helpline: 1099** | **Ambulance: 115** | **Rescue: 1122**.

---

## Features

- **Route comparison** – Generates and scores three candidate routes (shortest, main roads, alternative) between a start and end point.
- **Safety scoring (0–100)** – Each route gets a safety score, letter grade (A–F), and risk label (Safe / Moderate / High Risk / Very High Risk).
- **ML-driven predictions** – Uses a pre-trained Random Forest classifier (`guardher_model.pkl`) when available, with a rule-based fallback if the model isn't found.
- **Interactive map view** – Routes and an incident-density heatmap rendered with Folium, color-coded by safety.
- **Risk breakdown** – Per-route incident risk, isolation risk, "help availability" risk, and time-of-day risk.
- **Safety tips & emergency numbers** – Contextual advice based on time of travel (day/evening/night).
- **ML explanation tab** – Shows the exact features (distance to police/hospital/bus stop, incident counts, isolation score, etc.) behind a prediction.
- **Anonymous area reporting (demo UI)** – A form to report unsafe areas by type and severity.
- **City selector** – Preset locations for Lahore, Islamabad, and Karachi, plus custom lat/long input.

## How It Works

1. **Data** – Uses local incident reports (`incident_reports.csv`) and safety points of interest — police stations, hospitals, bus stops (`lahore_pois.csv`). If these files are missing, the app falls back to synthetic sample data so it still runs.
2. **Feature engineering** – For any given location and hour, 11 features are computed: distance to nearest police station/hospital/bus stop, incident counts within 500m/1km, average incident severity nearby, hour of day, time-of-day risk score, night/evening flags, and an isolation score.
3. **Prediction** – The Random Forest model (`guardher_model.pkl`) classifies the location into one of four risk classes and produces a safety score. If no model file is present, a rule-based heuristic estimates risk instead.
4. **Route ranking** – The same scoring is applied along waypoints of each candidate route, routes are ranked by average safety score, and the safest is highlighted as recommended.

## Tech Stack

- [Streamlit](https://streamlit.io/) – web app framework
- [Folium](https://python-visualization.github.io/folium/) + `streamlit-folium` – interactive maps and heatmaps
- [scikit-learn](https://scikit-learn.org/) – Random Forest model
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) – data handling
- [geopy](https://geopy.readthedocs.io/) – geodesic distance calculations
- [joblib](https://joblib.readthedocs.io/) – model serialization

## Project Structure

```
guardher-pakistan/
├── app.py                 # Main Streamlit application
├── guardher_model.pkl     # Pre-trained Random Forest model
├── incident_reports.csv   # Sample/collected incident report data
├── lahore_pois.csv        # Points of interest (police, hospitals, bus stops)
├── requirements.txt       # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/syeda-duaa/guardher-pakistan.git
cd guardher-pakistan
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Pick a city (Lahore, Islamabad, or Karachi) from the sidebar.
2. Choose a start location and destination from the presets, or enter custom coordinates.
3. Select your time of travel (morning / afternoon / evening / night).
4. Click **Analyze Route Safety** to view:
   - The recommended (safest) route with its score, grade, distance, and estimated walk time
   - A map with all routes plotted and an incident heatmap overlay
   - Safety tips tailored to the time of day
   - A breakdown of the ML features behind the prediction

## Model Details

- **Algorithm:** Random Forest Classifier (100 trees)
- **Classes:** Safe / Moderate / High Risk / Very High Risk
- **Input features (11):** distance to police, distance to hospital, distance to bus stop, incident count within 500m, incident count within 1km, average incident severity within 500m, hour of travel, time-risk score, is-night flag, is-evening flag, isolation score

If `guardher_model.pkl` isn't present in the directory, the app automatically switches to a transparent rule-based scoring method so the interface remains usable.

## Emergency Numbers (Pakistan)

| Service | Number |
|---|---|
| Police | 15 |
| Ambulance | 115 |
| Women Helpline | 1099 |
| Rescue | 1122 |

## Disclaimer

GuardHer Pakistan provides AI-generated risk estimates based on limited, self-reported, and/or synthetic incident data. It is **not** a substitute for personal judgment, local knowledge, or official safety guidance, and should not be relied on as the sole basis for travel decisions.

## Contributing

Issues and pull requests are welcome. If you have real incident or POI data for additional Pakistani cities, contributions to expand coverage beyond Lahore are especially appreciated.

## Author

Built by [Syeda Dua](https://github.com/syeda-duaa).
