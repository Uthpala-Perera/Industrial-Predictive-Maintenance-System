# 🏭 Industrial Predictive Maintenance System

An end-to-end machine learning system for predicting engine failure using the NASA Turbofan (FD004) dataset.  
This project simulates a real-world industrial monitoring platform with anomaly detection, alert generation, and an interactive dashboard.

---

##  Live Demo

🔗 **App Link:** *https://industrial-predictive-maintenance-system-75ywehfpatxnexz5hdqfd.streamlit.app/*

---

## Overview

This system predicts the **Remaining Useful Life (RUL)** of aircraft engines using multivariate sensor data.  
It integrates machine learning models with a real-time dashboard to simulate an industrial maintenance environment.

---

## Key Features

- RUL Prediction using multiple ML models  
- Alert System (Healthy / Warning / Critical)  
- Anomaly Detection using Isolation Forest  
- Interactive Streamlit Dashboard  
- RUL Degradation Visualization  
- Predicted vs Actual Performance Tracking  
- Sensor Behavior Monitoring  

---

## Tech Stack

- **Languages:** Python  
- **ML Models:** XGBoost, LightGBM, Random Forest, Linear Regression  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Streamlit  
- **Model Storage:** Joblib
- **Deployed** Streamlit Cloud 

---

## 📊 Dataset

- https://data.phmsociety.org/nasa/  - NASA Turbofan Engine Degradation Dataset (FD004)  
 
## Model Performance

| Model              | RMSE |
|-------------------|------|
| Random Forest      | ~42  |
| XGBoost           | ~38  |
| LightGBM          | ~38  |
| Linear Regression | ~60  |

---

## Dashboard Features

- Engine selection panel  
- Real-time RUL prediction  
- Health status indicators  
- Sensor trend visualization  
- Anomaly detection alerts  
- Prediction vs actual comparison

---

## Authors

KAUS Perera
