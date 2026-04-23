import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.data_loader import load_data
from src.preprocessing import add_rul, drop_unused_sensors, normalize_by_engine
from src.features import create_features
from src.anomaly import detect_anomalies
from src.alerts import generate_alert

# =========================
# PAGE CONFIG (INDUSTRIAL UI)
# =========================
st.set_page_config(
    page_title="Industrial Predictive Maintenance System",
    page_icon="🏭",
    layout="wide"
)

# =========================
# HEADER (CONTROL ROOM STYLE)
# =========================
st.title("🏭 INDUSTRIAL PREDICTIVE MAINTENANCE DASHBOARD")
st.markdown("Real-time Engine Health Monitoring | FD004 Turbofan System")

st.markdown("---")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_processed_data():

    df = load_data("data/train_FD004.txt")

    df = add_rul(df)
    df = drop_unused_sensors(df)
    df = normalize_by_engine(df)
    df = create_features(df)
    df = detect_anomalies(df)

    return df

data = load_processed_data()

model = joblib.load("models/xgb.pkl")

# =========================
# SIDEBAR CONTROL PANEL
# =========================
st.sidebar.header("⚙️ CONTROL PANEL")

engine_id = st.sidebar.selectbox(
    "Select Engine ID",
    sorted(data["engine_id"].unique())
)

engine_data = data[data["engine_id"] == engine_id]

latest = engine_data.drop(["RUL", "engine_id"], axis=1).iloc[-1:]

# =========================
# PREDICTION
# =========================
pred_rul = model.predict(latest)[0]
alert = generate_alert(pred_rul)

# HEALTH SCORE
health_score = max(0, min(100, (pred_rul / 200) * 100))

# =========================
# TOP KPI DASHBOARD
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔧 Engine ID", engine_id)

with col2:
    st.metric("⏳ Predicted RUL", f"{int(pred_rul)} cycles")

with col3:
    st.metric("💚 Health Score", f"{int(health_score)}%")

with col4:
    st.metric("⚠️ Status", alert)

st.markdown("---")

# =========================
# INDUSTRIAL STATUS PANEL
# =========================
st.subheader("🏭 ENGINE STATUS PANEL")

if pred_rul > 100:
    st.success("🟢 SYSTEM HEALTHY - NORMAL OPERATIONS")
elif pred_rul > 30:
    st.warning("🟡 MAINTENANCE REQUIRED SOON")
else:
    st.error("🔴 CRITICAL FAILURE RISK - STOP ENGINE")

# =========================
# ANOMALY PANEL
# =========================
st.subheader("🚨 ANOMALY DETECTION")

if engine_data["anomaly"].iloc[-1] == 1:
    st.error("ANOMALY DETECTED IN ENGINE BEHAVIOR")
else:
    st.success("NO ANOMALIES DETECTED")

# =========================
# RUL DEGRADATION CURVE
# =========================
st.subheader("📉 RUL DEGRADATION CURVE")

st.line_chart(engine_data[["cycle", "RUL"]].set_index("cycle"))

# =========================
# PREDICTED VS ACTUAL (WOW FACTOR)
# =========================
st.subheader("🎯 Predicted vs Actual RUL")

engine_data["pred_rul"] = model.predict(
    engine_data.drop(["RUL", "engine_id"], axis=1)
)

st.line_chart(engine_data[["RUL", "pred_rul"]].tail(100))

# =========================
# SENSOR DRIFT
# =========================
st.subheader("📡 SENSOR BEHAVIOR OVER TIME")

sensor_cols = [c for c in engine_data.columns if "sensor" in c]

st.line_chart(engine_data[sensor_cols].tail(100))

# =========================
# ALERT SYSTEM DEMO
# =========================
st.subheader("🚨 ALERT SYSTEM")

st.write(f"Current Engine Status: **{alert}**")

# =========================
# RAW DATA VIEW
# =========================
with st.expander("📊 RAW ENGINE DATA"):
    st.dataframe(engine_data.tail(30))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("⚙️ Built with Machine Learning | FD004 Predictive Maintenance System")