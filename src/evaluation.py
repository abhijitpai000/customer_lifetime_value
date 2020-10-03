"""Utility Functions for Evaluating Calibration Model and Holdout Data"""

from lifetimes import BetaGeoFitter
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path

from lifetimes.plotting import plot_cumulative_transactions
from lifetimes.plotting import plot_incremental_transactions
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
from lifetimes.plotting import plot_period_transactions

import matplotlib.pyplot as plt


def single_customer_evaluation(time_units=243):
    """
    Predicts Number of Purchases of a randomly chosen customer from the dataset.
    (conditional_expected_number_of_purchases_up_to_time)

    Parameters
    ----------
        time_units: int, default=243.
            Number of days for prediction.

    Returns
    -------
        (frequency_predicted, frequency_holdout)
    """
    # Loading Calibration Model.
    cal_bg_nbd = BetaGeoFitter(penalizer_coef=0.0)
    cal_bg_nbd.load_model(path="models/calibration_model.pkl")

    # Loading summary_cal_holdout dataset.
    summary_cal_holdout = pd.read_csv("datasets/summary_cal_holdout.csv")

    # Randomly sample single customer.
    individual = summary_cal_holdout.sample()
    frequency_prediction = cal_bg_nbd.predict(t=time_units,
                                              frequency=individual["frequency_cal"],
                                              recency=individual["recency_cal"],
                                              T=individual["T_cal"])
    frequency_holdout = individual["frequency_holdout"]

    return frequency_prediction, frequency_holdout


def root_mean_squared_error(time_units=243):
    """
    Calculates Root Mean Squared Error of all predictions.

    Parameters
    ----------
        time_units: int, default=243.
            Number of days for prediction.

    Yields
    ------
        summary_cal_holdout_preds.csv.

    Returns
    ------
        rmse
    """
    # Loading Calibration Model.
    cal_bg_nbd = BetaGeoFitter(penalizer_coef=0.0)
    cal_bg_nbd.load_model(path="models/calibration_model.pkl")

    # Loading summary_cal_holdout dataset.
    summary_cal_holdout = pd.read_csv("datasets/summary_cal_holdout.csv")
    frequency_holdout = summary_cal_holdout["frequency_holdout"].copy()

    # Predictions.
    frequency_predictions = cal_bg_nbd.predict(t=time_units,
                                               frequency=summary_cal_holdout["frequency_cal"],
                                               recency=summary_cal_holdout["recency_cal"],
                                               T=summary_cal_holdout["T_cal"])

    # Adding Predictions to Summary dataset.
    summary_cal_holdout["frequency_predictions"] = frequency_predictions.copy()
    file_path = Path.cwd() / "datasets/summary_cal_holdout_preds.csv"
    summary_cal_holdout.to_csv(file_path, index=False)

    rmse = mean_squared_error(frequency_holdout,
                              frequency_predictions,
                              squared=False)
    return rmse

def evaluation_plots(plot_type):
    """
    Evaluation Plots:
    - Tracking Cumulative Transactions
    - Tracking Daily Transactions
    - Frequency of Repeated Transactions
    - Calibration vs Holdout.

    Parameters
    ----------
        plot_type: str.
            "tracking" - Tracking Cumulative and Tracking Daily Transactions.
            "repeated" - Frequency of Repeated Transactions.
            "calibration_holdout" - Calibration vs Holdout Purchases.
    """
    # Loading Calibration Model.
    cal_bg_nbd = BetaGeoFitter(penalizer_coef=0.0)
    cal_bg_nbd.load_model(path="models/calibration_model.pkl")

    # Loading summary_cal_holdout dataset.
    summary_cal_holdout = pd.read_csv("datasets/summary_cal_holdout.csv")

    # Loading Transactions.
    transactions = pd.read_csv("datasets/transactions.csv")

    if plot_type == "tracking":
        fig = plt.figure(figsize=(20, 4))
        plot_cumulative_transactions(model=cal_bg_nbd,
                                     transactions=transactions,
                                     datetime_col="order_purchase_timestamp",
                                     customer_id_col="customer_unique_id",
                                     t=604,
                                     t_cal=512,
                                     freq="D",
                                     ax=fig.add_subplot(121))

        plot_incremental_transactions(model=cal_bg_nbd,
                                      transactions=transactions,
                                      datetime_col="order_purchase_timestamp",
                                      customer_id_col="customer_unique_id",
                                      t=604,
                                      t_cal=512,
                                      freq="D",
                                      ax=fig.add_subplot(122))
    elif plot_type == "repeated":
        plot_period_transactions(model=cal_bg_nbd)

    elif plot_type == "calibration_holdout":
        plot_calibration_purchases_vs_holdout_purchases(model=cal_bg_nbd,
                                                        calibration_holdout_matrix=summary_cal_holdout)
    return



