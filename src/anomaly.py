from sklearn.ensemble import IsolationForest
import numpy as np

def detect_anomalies(df):

    features = df.drop(["engine_id", "RUL"], axis=1)

    iso = IsolationForest(contamination=0.05, random_state=42)
    df["iso_anomaly"] = iso.fit_predict(features)

    z = np.abs((features - features.mean()) / features.std())
    df["z_anomaly"] = (z > 3).any(axis=1).astype(int)

    df["anomaly"] = ((df["iso_anomaly"] == -1) | (df["z_anomaly"] == 1)).astype(int)

    return df