"""Train BG/NBD Model"""

from lifetimes import BetaGeoFitter
import pandas as pd
from pathlib import Path


def calibration_model():
    """
    Trains BG/NBD Calibration Model.

    Yields
    ------
        calibration_model.pkl
    """
    summary_cal_holdout = pd.read_csv("datasets/summary_cal_holdout.csv")

    # Training Calibration Model.
    cal_bg_nbd = BetaGeoFitter(penalizer_coef=0.0)
    cal_bg_nbd.fit(frequency=summary_cal_holdout["frequency_cal"],
                   recency=summary_cal_holdout["recency_cal"],
                   T=summary_cal_holdout["T_cal"],
                   verbose=False)

    # Saving Model.
    file_path = Path.cwd() / "models/calibration_model.pkl"
    cal_bg_nbd.save_model(path=file_path)
    return


def clv_model():
    """
    Trains BG/NBD Model on entire RFM data, final fit.

    Yields
    ------
        customer_lifetime_estimator.pkl
    """
    summary = pd.read_csv("datasets/summary.csv")

    # Training Calibration Model.
    clv = BetaGeoFitter(penalizer_coef=0.0)
    clv.fit(frequency=summary["frequency"],
            recency=summary["recency"],
            T=summary["T"],
            verbose=False)

    # Saving Model.
    file_path = Path.cwd() / "models/customer_lifetime_estimator.pkl"
    clv.save_model(path=file_path)
    return
