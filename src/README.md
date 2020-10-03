# Documentation.

| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| preprocess | make_dataset() | Pre-processes raw data | -- | transactions, summary, summary_cal_holdout, customer_mapping | transactions
| train | train_model() | Trains BG/NBD models on summary and summary_cal_holodut dataset | -- | calibration_model.pkl, customer_lifetime_estimator.pkl, summary_cal_preds.csv | --
| evaluation | -- | Utility functions for evaluation on calibration_holdout dataset | -- | -- | --
| predict | number_of_purchases(), probability_alive() | Predictions using final fit model | -- | -- | --
