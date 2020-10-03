"""Checks if RMSE of Model Prediction is as low as expected."""

import pandas as pd
from lifetimes import BetaGeoFitter
from sklearn.metrics import mean_squared_error

import pytest

@pytest.fixture
def load_data_and_model():
    """Loads Customer Lifetime Estimator Model"""
    model = BetaGeoFitter(penalizer_coef=0.0)
    model.load_model("../models/calibration_model.pkl")
    summary_cal_holdout = pd.read_csv("../datasets/summary_cal_holdout.csv")
    return model, summary_cal_holdout

def test_model_rmse(load_data_and_model):
    """Test RMSE of Predicted Frequency vs Holdout Frequency"""
    data = load_data_and_model[1]
    model = load_data_and_model[0]
    predictions = model.predict(t=243,
                                frequency=data["frequency_cal"],
                                recency=data["recency_cal"],
                                T=data["T_cal"])
    rmse = mean_squared_error(data["frequency_holdout"],
                              predictions,
                              squared=False)
    assert rmse < 0.15, "RMSE is greater than 0.15"
