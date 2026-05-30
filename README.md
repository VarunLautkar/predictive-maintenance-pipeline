# Predictive Maintenance Pipeline — Turbofan Engine RUL Prediction

An end-to-end, production-ready machine learning pipeline that predicts the **Remaining Useful Life (RUL)** of turbofan engines using NASA's C-MAPSS dataset. Built with modular Python code, reproducible configuration, logging, and unit tests.

## Business Context

Unplanned equipment failures cost the manufacturing and aviation industries billions annually. Predictive maintenance uses sensor data to forecast *when* a component will fail, enabling maintenance teams to act proactively — reducing downtime, preventing catastrophic failures, and optimizing maintenance schedules.

This project predicts how many operational cycles a turbofan engine has remaining before failure, using 21 sensor readings collected over the engine's lifetime.

## Pipeline Architecture

```
┌─────────────┐    ┌────────────────┐    ┌─────────────────────┐    ┌───────────┐    ┌────────────┐
│   Data       │───▶│ Preprocessing  │───▶│ Feature Engineering │───▶│ Training  │───▶│ Evaluation │
│  Ingestion   │    │  & Cleaning    │    │  (rolling stats,    │    │ (RF, XGB) │    │ (RMSE,MAE, │
│              │    │                │    │   trend features)   │    │           │    │  plots)    │
└─────────────┘    └────────────────┘    └─────────────────────┘    └───────────┘    └────────────┘
```

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)**

- 100 engines run to failure (training)
- 100 engines with partial history (testing)
- 21 sensor measurements + 3 operational settings per cycle
- Source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/VarunLautkar/predictive-maintenance-pipeline.git
cd predictive-maintenance-pipeline
pip install -r requirements.txt
```

### 2. Download Data

Download the [C-MAPSS dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place the text files in `data/raw/`:

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

### 3. Run the Full Pipeline

```bash
python run_pipeline.py --config config/config.yaml
```

Or run individual steps:

```bash
python -m src.data_ingestion
python -m src.preprocessing
python -m src.feature_engineering
python -m src.train
python -m src.evaluate
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Results

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Random Forest | ~20-25 | ~15-18 | ~0.75+ |
| XGBoost | ~18-22 | ~13-16 | ~0.80+ |

*Exact values depend on run. See `results/metrics.json` after running.*

Evaluation plots are saved to `results/`:
- Predicted vs Actual RUL scatter plot
- Residual distribution
- Feature importance chart

## Project Structure

```
predictive-maintenance-pipeline/
├── README.md
├── requirements.txt
├── run_pipeline.py              # CLI entry point
├── config/
│   └── config.yaml              # All hyperparameters & paths
├── data/
│   ├── raw/                     # Original NASA text files
│   └── processed/               # Cleaned CSVs (generated)
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py        # Load raw text → structured CSV
│   ├── preprocessing.py         # Clean, normalize, create RUL target
│   ├── feature_engineering.py   # Rolling stats, trend features
│   ├── train.py                 # Model training with config
│   ├── evaluate.py              # Metrics, plots, reporting
│   └── utils.py                 # Logging setup, helpers
├── notebooks/
│   └── exploration.ipynb        # EDA only (not part of pipeline)
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_feature_engineering.py
├── results/                     # Metrics JSON + plots (generated)
└── .github/
    └── workflows/
        └── ci.yml               # GitHub Actions CI
```

## Key Design Decisions

- **Piecewise Linear RUL**: Raw RUL is capped at 125 cycles. Engines don't degrade in early life, so predicting RUL=300 adds noise without value.
- **Sensor Selection**: Sensors with near-zero variance (1, 5, 6, 10, 16, 18, 19) are dropped — they carry no degradation signal.
- **Rolling Features**: Window-based statistics (mean, std, min, max, slope) over 5/10/20-cycle windows capture degradation trends better than raw instantaneous readings.
- **Asymmetric Evaluation**: Late predictions (predicting more RUL than actual) are more dangerous than early ones. The evaluation module flags this.

## What I'd Improve With More Time

- Add MLflow experiment tracking
- Dockerize the full pipeline
- Implement LSTM/temporal models for sequential sensor data
- Build a Streamlit dashboard for real-time RUL monitoring
- Add data drift detection for production deployment

## Citation

A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
