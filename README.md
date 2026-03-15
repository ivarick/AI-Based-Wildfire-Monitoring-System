
<h1 align="center">Firewatch — Tech4Connect 2025</h1>
<h3 align="center">Active Wildfire Intelligence & Prevention Operations Center</h3>

<p align="center">
  Firewatch is an end-to-end, multi-agent AI platform built for the <strong>Huawei Tech4Connect 2025 Algeria Hackathon</strong>. It provides live, proactive monitoring by combining extreme-weather mathematical forecasting, machine learning, and deep learning into a single glassmorphic command dashboard.
</p>

---

## 📖 Overview

Algeria loses tens of thousands of hectares of forest every year to wildfire, particularly in the Kabylie region. Firewatch was developed as a real-time wildfire intelligence and response command room to address this gap, shifting crisis response from reactive to proactive.

The system fuses live satellite fire detections, current weather data, Canadian FWI index computation, AI-driven risk scoring, deep learning satellite fire segmentation, and a structured response layer into a single operational interface.

## ✨ Key Features

*   **Live Sensor Fusion:** Autonomously pulls live datasets from NASA FIRMS (VIIRS active hotspots) and Open-Meteo (high-res meteorology).
*   **Predictive Engine (XGBoost):** Evaluates 15 micro-climate features (including Canadian FWI metrics) to output risk probabilities *before* ignition.
*   **Semantic Vision (PyTorch U-Net):** Segment satellite imagery (Dice 0.936) to instantly map precise burned area perimeters.
*   **Explainable AI (SHAP):** Built-in SHAP values explain every high-risk AI prediction by breaking down the specific variables driving the alert.
*   **Live Spread Projection:** Wind-vector mathematics that draw 1h, 3h, and 6h forward-spread geographical projections for active fires.
*   **Historical Hindcast Validation:** Pre-validated against real historical crisis events to verify model accuracy based on archived scenarios.
*   **Simulated 5G & Drone Integration:** Features an interactive simulated response topology encompassing 5G-connected UAV drones and a bilingual (FR/AR) Cell Broadcast warning system.

---

## 🧠 System Architecture & Models

The architecture relies on multiple data pipelines and ML/DL models working synchronously:

### 1. Canadian FWI Engine
Computes the full Van Wagner 1987 formulation, capturing the Fine Fuel Moisture Code (FFMC), Duff Moisture Code (DMC), Drought Code (DC), Initial Spread Index (ISI), Build-Up Index (BUI), and overall Fire Weather Index (FWI) from real-time Open-Meteo data.

### 2. XGBoost Fire Risk Predictor
Trained on the UCI Algeria Forest Fires dataset to predict binary fire probabilities. The dataset was balanced using SMOTE, and the model achieves over 0.85 Validation AUC.

### 3. PyTorch U-Net Satellite Segmentation
An encoder-decoder CNN explicitly trained on the Sentinel-2 Turkey Wildfire 2021 dataset. It generates spatial fire segmentation (perimeters of burning areas) which act as polygonal layers onto the dashboard's GIS interface.

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### Requirements

*   Python 3.9+

### 1. Clone the repository and setup the environment

```bash
git clone https://github.com/your-username/firewatch.git
cd firewatch
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data & Model Setup
Ensure the pre-trained weights and training datasets are properly populated within the `src/data` and `src/models` directories (or root `data` and `models` depends on your working directory):

```
data/
├── algerian_forest_fires.csv
└── train_balanced_features.csv
models/
├── feature_cols.pkl
├── xgboost_fire_risk.pkl
└── fire_segmentation_unet.pth
```
*(Note: If the XGBoost model is missing, the backend will autonomously synthesize one by dynamically training on the CSV datasets).*

### 3. Launch the Application

Navigate to your source directory and run the Streamlit app:

```bash
python -m streamlit run src/firewatch_app.py
```

The Command Center will open automatically in your browser at `http://localhost:8501`. The dashboard runs on a continuous 5-minute data-refresh cycle.

---

## 📂 Project Structure

```text
firewatch/
├── src/
│   ├── firewatch_app.py              # Streamlit command room UI & maps
│   ├── firewatch_pipeline.py         # NASA FIRMS + Open-Meteo + FWI + alerts logic
│   ├── firewatch_model.py            # XGBoost engine, exact SHAP explainability, hindcast validation
│   ├── firewatch_sim.py              # Simulated hardware (drones, 5G cells, etc)
│   ├── firewatch_theme.py            # Custom CSS for thermal ops design
│   ├── train_fire_segmentation.py    # Torch U-Net segmentation training script
│   ├── data/                         # CSV datasets for XGBoost fallback training
│   └── models/                       # Stored model `.pkl` and `.pth` weight files
├── design/                           # UI/UX design assets or mockups
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

<p align="center">
  <i>Built for the preservation of forests and the safety of citizens.</i>
<br/>
  <b>Huawei Tech4Connect Algeria — Track 2: Agritech & Environmental Protection</b>
</p>
