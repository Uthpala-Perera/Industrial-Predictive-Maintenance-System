import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error


def train_models(X_train, y_train):

    models = {}

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models["rf"] = rf

    xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05)
    xgb_model.fit(X_train, y_train)
    models["xgb"] = xgb_model

    lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
    lgb_model.fit(X_train, y_train)
    models["lgb"] = lgb_model

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["lr"] = lr

    return models


def evaluate(models, X_test, y_test):

    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = rmse

    return results