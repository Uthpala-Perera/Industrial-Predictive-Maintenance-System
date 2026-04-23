print("Starting...")

from src.data_loader import load_data
print("Loaded data_loader")

from src.preprocessing import add_rul, drop_unused_sensors, normalize_by_engine
print("Loaded preprocessing")

from src.features import create_features
print("Loaded features")

from src.models import train_models, evaluate
print("Loaded models")

from src.anomaly import detect_anomalies
print("Loaded anomaly detection")

from src.alerts import generate_alert
print("Loaded alerts")

from sklearn.model_selection import train_test_split
import joblib
import os

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
train = load_data("data/train_FD004.txt")

# =========================
# PREPROCESSING
# =========================
print("Adding RUL...")
train = add_rul(train)

print("Cleaning sensors...")
train = drop_unused_sensors(train)

print("Normalizing...")
train = normalize_by_engine(train)

print("Feature engineering...")
train = create_features(train)

# =========================
# ANOMALY DETECTION
# =========================
print("Running anomaly detection...")
train = detect_anomalies(train)

# =========================
# SPLIT DATA
# =========================
print("Splitting...")
X = train.drop(["RUL", "engine_id"], axis=1)
y = train["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODELS
# =========================
print("Training models...")
models = train_models(X_train, y_train)

# =========================
# EVALUATION
# =========================
print("Evaluating...")
results = evaluate(models, X_test, y_test)
print("Model Performance:", results)

# =========================
# ALERT DEMO
# =========================
print("\nGenerating alerts (sample)...")
preds = models["xgb"].predict(X_test[:10])

for i, p in enumerate(preds):
    print(f"Engine {i} | RUL: {int(p)} | Alert: {generate_alert(p)}")

# =========================
# SAVE MODEL
# =========================
print("Saving model...")

os.makedirs("models", exist_ok=True)
joblib.dump(models["xgb"], "models/xgb.pkl")

print("Model saved ✔")
print("DONE ")