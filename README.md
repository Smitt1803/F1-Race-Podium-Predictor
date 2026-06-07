## Team Members

- Smit Vaghani (@Smitt1803)
- Vidhi Gadge (@vidhigadge)

## F1 Race Podium Predictor

An end-to-end Machine Learning project that predicts whether a Formula 1 driver will finish on the podium (Top 3) using historical race results, qualifying performance, and engineered racing statistics. The project covers data collection, preprocessing, feature engineering, model training, prediction generation, and deployment through an interactive Streamlit web application.

## Overview

Formula 1 is a data-driven sport where qualifying pace, grid position, and historical performance strongly influence race outcomes. This project leverages Machine Learning techniques to analyze these factors and predict podium finishes with high accuracy.

The complete pipeline includes:

- Data Collection & Web Scraping
- Data Cleaning & Feature Engineering
- Model Training & Evaluation
- Podium Prediction
- Interactive Streamlit Dashboard

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Matplotlib
- BeautifulSoup
- Selenium

## Model Performance

### Model F1 Score

XGBoost 0.9918
Random Forest 0.9858
SVM 0.9268
Logistic Regression 0.8235

### Best Model: XGBoost

## Project Structure

```bash
F1-Race-Podium-Predictor/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── models/
│   ├── XGBoost.pkl
│   └── scaler.pkl
│
├── results/
│
├── src/
│   ├── scrape_f1_champions.py
│   ├── phase2_training.py
│   └── phase3_prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
git clone https://github.com/smitt1803/F1-Race-Podium-Predictor.git
```

```bash
cd F1-Race-Podium-Predictor
```

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python src/phase2_training.py
```

## Run Predictions

```bash
python src/phase3_prediction.py
```

## Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Key Features Used

- Grid Position
- Qualifying Position
- Driver Points
- Best Qualifying Time
- Rolling Average Track Points
- Rolling Average Qualifying Position
- Rolling Average Global Points
- Qualifying-to-Finish Delta

## Future Improvements

- Real-time race data integration
- Driver-specific prediction dashboard
- Race winner prediction
- Championship outcome forecasting
- Advanced feature engineering
