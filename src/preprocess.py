"""Pre-processing Raw data."""

import pandas as pd
from pathlib import Path

from lifetimes.utils import calibration_and_holdout_data
from lifetimes.utils import summary_data_from_transaction_data


def _create_transactions(customers_raw, orders_raw, payments_raw):
    """
    Structure data into transactions format using csv from zip.

    Parameters
    ----------
        customers_raw: "olist_customers_dataset.csv"
        orders_raw: "olist_orders_dataset.csv"
        payments_raw: "olist_payments_dataset.csv"

    Yields
    ------
        transactions.csv.

    Returns
    -------
        transactions with following features:

        ["customer_unique_id", "order_purchase_timestamp",
        "order_date", "avg_inter_purchase_time",
        "payment_type"]
    """

    # Extracting CustomerID and CustomerUniqueID
    transactions = customers_raw[["customer_unique_id", "customer_id"]].copy()

    # Adding Order ID.
    transactions = transactions.merge(orders_raw[["order_id", "customer_id", "order_purchase_timestamp"]],
                                      how="inner",
                                      on="customer_id")

    # Dropping CustomerID.
    transactions.drop("customer_id", axis=1, inplace=True)

    # Adding Payment Value.
    payment_value = payments_raw.groupby("order_id")[["payment_value"]].sum()
    transactions = transactions.merge(payment_value, how="left", on="order_id")

    # Changing to datetime.
    transactions["order_purchase_timestamp"] = pd.to_datetime(transactions["order_purchase_timestamp"])

    # Adding Payment type for analysis.
    payment_type = payments_raw.groupby("order_id", as_index=False).last()
    transactions = transactions.merge(payment_type[["order_id", "payment_type"]], on="order_id", how='left')

    # Dropping one customer_id which is present in list but has not made a purchase.
    transactions.dropna(axis=0, inplace=True)

    # Sorting.
    transactions.sort_values(by="order_purchase_timestamp", ascending=True, inplace=True)

    # Adding Year month to extract 2017-01 to 2018-08 data.
    transactions["year_month"] = transactions["order_purchase_timestamp"].dt.to_period("M")
    transactions = transactions[(transactions["year_month"] > "2016-12") & (transactions["year_month"] < "2018-09")]

    # Add order_date only.
    transactions["order_date"] = transactions["order_purchase_timestamp"].dt.date

    # Compute Average Inter Purchase time for each customer.
    transactions = _inter_purchase_time(transactions)

    # Saving file.
    file_path = Path.cwd() / "datasets/transactions.csv"

    transactions.to_csv(file_path, index=False)

    return transactions


def _inter_purchase_time(transactions):
    """
    Computes average time between repeated transactions in days for each customer.

    Parameters
    ----------
        transactions: transactions data with "order_date"

    Returns
    -------
        transactions with average inter purchase time for each customer.
    """
    # Customers with repeated transactions.
    repeated = transactions.groupby("customer_unique_id").count().sort_values(by="order_date", ascending=False)
    repeated = repeated[repeated["order_date"] > 2]
    repeated.reset_index(inplace=True)
    repeated_customer_list = repeated["customer_unique_id"].copy()

    # Separating customer id and order date.
    customer_id_order_date = transactions[["customer_unique_id", "order_date"]].copy()

    # Computation.
    avg_inter_purchase_time = {}
    for customer in repeated_customer_list:
        x = customer_id_order_date[customer_id_order_date["customer_unique_id"] == customer].sort_values(
            by="order_date")
        x["inter"] = x["order_date"] - x["order_date"].shift(-1)
        x["inter"] = -x["inter"].dt.days
        avg_inter_purchase_time[customer] = x["inter"].mean()

    # Converting dictionary to dataframe.
    inter_purchase_data = pd.DataFrame.from_dict(avg_inter_purchase_time, orient="index")
    inter_purchase_data.reset_index(inplace=True)
    inter_purchase_data.columns = ["customer_unique_id", "avg_inter_purchase_time"]

    # Merging into Transactions dataset.
    transactions = transactions.merge(inter_purchase_data, on="customer_unique_id", how="left")
    transactions.fillna(0, axis=1, inplace=True)
    return transactions


def _summary_calibration_and_holdout(transactions):
    """
    Creates Summary (RFM) data for Model Training and Evaluation using transactions data.

    Parameters
    ---------
        transactions: transaction data with customer_unique_id and order_purchase_timestamp.

    Yields
    ------
        summary_cal_holdout.csv.
    """
    summary_cal_holdout = calibration_and_holdout_data(transactions=transactions,
                                                       customer_id_col="customer_unique_id",
                                                       datetime_col="order_purchase_timestamp",
                                                       calibration_period_end="2017-12-31",
                                                       observation_period_end="2018-08-31",
                                                       freq="D",
                                                       freq_multiplier=1)

    # Saving file.
    file_path = Path.cwd() / "datasets/summary_cal_holdout.csv"
    summary_cal_holdout.to_csv(file_path, index=False)
    return


def _summary(transactions):
    """
    Create Summary (RFM) data for final model fit using transactions data.

    Yields
    ------
        summary.csv - RFM data.
        customer_mapping.csv - Summary Index - Customer ID from Transactions mapping.
    """
    summary = summary_data_from_transaction_data(transactions=transactions,
                                                 customer_id_col="customer_unique_id",
                                                 datetime_col="order_purchase_timestamp",
                                                 observation_period_end="2018-08-31",
                                                 freq="D")
    # Customer ID - Summary Index mapping.
    customer_mapping = transactions.groupby("customer_unique_id", as_index=False).count()[["customer_unique_id"]]
    file_path = Path.cwd() / "datasets/customer_mapping.csv"
    customer_mapping.to_csv(file_path, index=False)

    # Saving file.
    file_path = Path.cwd() / "datasets/summary.csv"
    summary.to_csv(file_path, index=False)
    return


def make_dataset():
    """
    Prepares data for Training and Evaluation.

    Yields
    ------
        transactions.csv: Transactions data using "customers, orders and payments" data from zip file.
        summary_cal_holdout.csv: RFM Data from Transactions with Calibration and Holdout Separated.
        summary.csv: RFM data of the entire dataset for Training after evaluation.

    Returns
    -------
        transactions dataset.
    """
    customers_raw = pd.read_csv("datasets/olist_customers_dataset.csv")
    orders_raw = pd.read_csv("datasets/olist_orders_dataset.csv")
    payments_raw = pd.read_csv("datasets/olist_order_payments_dataset.csv")

    # Creating Transactions dataset.
    transactions = _create_transactions(customers_raw, orders_raw, payments_raw)

    # Creating Summary data with Calibration and Holdout.
    _summary_calibration_and_holdout(transactions)

    # Summary data for final fit.
    _summary(transactions)

    return transactions
