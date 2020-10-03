"""Run Scripts from Command Line"""

from src.preprocess import make_dataset
from src.train import train_model
from src.evaluation import single_customer_evaluation, root_mean_squared_error

if __name__ == '__main__':
    make_dataset()
    train_model()
    freq_predictions, freq_holdout = single_customer_evaluation(time_units=243)
    rmse = root_mean_squared_error(time_units=243)

    print(f"Single Customer Predictions:"
          f"\nPrediction:"
          f"\n {freq_predictions}"
          f"\nGround Truth: "
          f"\n {freq_holdout}"
          f"\n OVERALL RMSE: {rmse}")
