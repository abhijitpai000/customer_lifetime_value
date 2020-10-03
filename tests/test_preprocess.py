"""Test Datasets generated during pre-processing"""

import pytest
import pandas as pd


@pytest.fixture
def load_datasets():
    """
    Load transactions.csv, summary_cal_holdout.csv & summary.csv.
    """
    transactions = pd.read_csv("../datasets/transactions.csv")
    summary_cal_holdout = pd.read_csv("../datasets/summary_cal_holdout.csv")
    summary = pd.read_csv("../datasets/summary.csv")
    return transactions, summary_cal_holdout, summary

def test_transactions(load_datasets):
    """Test Shape of Transactions dataset"""
    assert load_datasets[0].shape == (99092, 8), "Transactions Dataset Failed"

def test_summary_cal_holdout(load_datasets):
    """Test Shape of Transactions dataset"""
    assert load_datasets[1].shape == (43640, 5), "Summary Calibration Dataset Failed"

def test_summary(load_datasets):
    """Test Shape of Transactions dataset"""
    assert load_datasets[2].shape == (95774, 3), "Summary Dataset Failed"
