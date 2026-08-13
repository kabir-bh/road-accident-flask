# Road Accident Analysis — Interactive Flask Web App

A full-stack interactive web application built on MoRTH road accident
data, allowing users to explore state-wise accident trends, fatality
rates, and Prophet-based forecasts through 2026.

**Live App:** [road-accident-flask.onrender.com](https://road-accident-flask.onrender.com)

---

## Features

| Page | Description |
|---|---|
| **Dashboard** | National trend, top 10 dangerous states, headline KPI stats |
| **State Explorer** | Select any state — view trend, fatality rate, and 2026 forecast |
| **Compare States** | Compare up to 3 states side by side with forecast table |

---

## Tech Stack

- **Backend** — Flask, Python
- **Data Processing** — Pandas, NumPy
- **Forecasting** — Facebook Prophet
- **Visualization** — Plotly (interactive), Matplotlib (forecast charts)
- **Frontend** — Bootstrap 5, Jinja2
- **Deployment** — Render

---

## Project Structure

```
accidents_flask/
│
├── app.py                  ← Flask routes and data logic
├── requirements.txt
│
├── data/
│   ├── state_accidents.csv
│   ├── state_fatalities.csv
│   ├── road_user_fatalities.csv
│   └── safety_devices.csv
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── state.html
│   └── compare.html
│
└── static/
    └── css/
        └── style.css
```

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/kabir-bh/road-accident-flask.git
cd road-accident-flask

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Dataset

**Source:** [OpenCity India](https://data.opencity.in/dataset/road-accidents-in-india-2023)
**Publisher:** Ministry of Road Transport and Highways (MoRTH), India

---

## Key Insights Surfaced by the App

- Tamil Nadu leads in accident volume — projected to reach **73,915**
  accidents by 2026
- Kerala shows the steepest projected rise (+8,752 by 2026) despite
  being among India's most developed states
- Andhra Pradesh is the only top-10 state showing a consistent decline
- Two-wheelers account for **44.8%** of all road fatalities
- **54,000+** deaths in 2023 linked to helmet absence

---

## Related

- [Road Accident Analysis Notebook](https://github.com/kabir-bh/road-accident-analysis)
  — EDA, forecasting notebook, and full findings

---

## Author

**Kabir Bhatia**
[LinkedIn](https://www.linkedin.com/in/kabirbh) |
[GitHub](https://github.com/kabir-bh)
