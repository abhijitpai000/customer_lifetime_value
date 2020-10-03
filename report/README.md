# Customer Lifetime Value.
Forecasting Number of transactions a customer would make using Beta Geometric-Negative Binomial Distribution, a BTYD-Probabilistic Model.

## Overview
Trained a Beta Geometric-Negative Binomial Distribution (BG/NBD) model that explains how frequently customers make purchases while they are still "alive" and how likely a customer is to churn in any given time period, using customer transactions of E-Commerce store Olist [public dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)

**Model Outcome**


Trained Model Predicts Number of Purchases with a RMSE of 0.144 and is able to capture 99% of customer historical transactions with frequency less than or equal to 4, Less than 1% of customers have greater than 4 repeated purchases in the dataset.

![image.png](attachment:image.png)




*Model vs Actual - Cumulative Transactions and Daily Transactions*
![image-3.png](attachment:image-3.png)


## About Olist Dataset

The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil, the orders are divided into 9 .csv files in a relational database schema.

For this study, I have aggregrated data using the following 3 .csv files out of the 9 in the zip file.
* olist_customers_dataset.csv
* olist_orders_dataset.csv
* olist_payments_dataset.csv

**Source :** [Olist Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)


# Analysis Walk-through

**Table of Contents**

1. [Experiment](#experiment)
2. [Package Introduction](#introduction)
3. [Pre-Processing](#preprocess)
4. [Train BG/NBD Models](#train)
5. [Model Evaluation](#evaluation)
6. [Model Interpretation](#inter) 
    1. [Frequency-Recency-Expected Number of Purchases Analysis](#rf)
    2. [Customer Segmentation](#segment)
    3. [Probability-Alive-Matrix](#alive)
    4. [High Probability Customers](#highprob)
    5. [Future Forecast Random Sampled Customer](#forecast)

## Experiment <a name="experiment"></a>

*Model Selection*

Experimented with Pareto/NBD, Modified-Beta-Geometric/NBD, and Beta-Geometric/NBD model. Out of the 3 selected BG/NBD for further exploration as it presented faster training time and low prediction RMSE.

*Calibration-Holdout Cut-off Selection*

Transactions dataset year-month ranges from 2016-09 to 2018-08. Treating the calibration-holdout thershold date as hyperparameter, experimented with different dates. Selected 2017-01 to 2017-12 as calibration period and 2018-01 to 2018-08 as holdout period for model evaluation.



```python
# Setting Working Directory to Git-Clone-Path.

mydir = "E:\Data Science Portfolio\Git Projects\customer_lifetime_value"

%cd $mydir
```

    E:\Data Science Portfolio\Git Projects\customer_lifetime_value
    

## Package Introduction <a name="introduction"></a>

| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| preprocess | make_dataset() | Pre-processes raw data | -- | transactions, summary, summary_cal_holdout, customer_mapping | transactions
| train | train_model() | Trains BG/NBD models on summary and summary_cal_holodut dataset | -- | calibration_model.pkl, customer_lifetime_estimator.pkl, summary_cal_preds.csv | --
| evaluation | -- | Utility functions for evaluation on calibration_holdout dataset | -- | -- | --
| predict | number_of_purchases(), probability_alive() | Predictions using final fit model | -- | -- | --


```python
# Basic Imports.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Local Imports.
from src.preprocess import make_dataset
from src.train import train_model
```

## Pre-Processing.  <a name="preprocess"></a>

**make_dataset()** Pre-processes raw .csv files to following.
1. *Transactions data features* - 
        customer_unique_id - Customer ID
        order_id - Order ID
        order_purchase_timestamp - Timestamp when order was placed
        payment_value - Order payment value
        payment_type - Method used to make payment
        year_month - Year-Month from order timestamp
        order_date - Order date from order timestamp
        avg_inter_purchase_time - Average number of days between each orders for repeated customers

2. *Summary Calibration and Holdout data features* -
        frequency_cal - Frequency of Purchases: (Total Purchase Count) - 1 
        recency_cal - Age of customer: (first purchase) - (latest purchase) days
        T_cal - Total age of customer: (first purchase) - (closing date in dataset)
        frequency_holdout - Frequency after thershold
        duration_holdout - Number of days in holdout

3. *Summary data features* -
        frequency - Frequency of Purchases: (Total Purchase Count) - 1 
        recency -  Age of customer: (first purchase) - (latest purchase) days
        T - Total age of customer: (first purchase) - (closing date in dataset)


```python
# Pre-Processing raw data.

transactions = make_dataset()
```


```python
# Transactions dataset.

transactions.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_unique_id</th>
      <th>order_id</th>
      <th>order_purchase_timestamp</th>
      <th>payment_value</th>
      <th>payment_type</th>
      <th>year_month</th>
      <th>order_date</th>
      <th>avg_inter_purchase_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f7b981e8a280e455ac3cbe0d5d171bd1</td>
      <td>ec7a019261fce44180373d45b442d78f</td>
      <td>2017-01-05 11:56:06</td>
      <td>19.62</td>
      <td>credit_card</td>
      <td>2017-01</td>
      <td>2017-01-05</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>83e7958a94bd7f74a9414d8782f87628</td>
      <td>b95a0a8bd30aece4e94e81f0591249d8</td>
      <td>2017-01-05 12:01:20</td>
      <td>19.62</td>
      <td>boleto</td>
      <td>2017-01</td>
      <td>2017-01-05</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Summary Calibration and Holdout dataset.

summary_cal_holdout = pd.read_csv("datasets/summary_cal_holdout.csv")

summary_cal_holdout.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency_cal</th>
      <th>recency_cal</th>
      <th>T_cal</th>
      <th>frequency_holdout</th>
      <th>duration_holdout</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>296.0</td>
      <td>0.0</td>
      <td>243.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>80.0</td>
      <td>0.0</td>
      <td>243.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Summary dataset for final Fit.

summary = pd.read_csv("datasets/summary.csv")

summary.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
      <th>recency</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>116.0</td>
    </tr>
  </tbody>
</table>
</div>



## Training BG/NBD Models <a name="train"></a>

**train_model():**

Trains two BG/NBD models. One on summary_cal_holdout dataset for evaluation and another on summary as a final fit.


```python
# Training Models.

train_model()
```

    Optimization terminated successfully.
             Current function value: 0.070935
             Iterations: 61
             Function evaluations: 63
             Gradient evaluations: 63
    Optimization terminated successfully.
             Current function value: 0.086931
             Iterations: 62
             Function evaluations: 63
             Gradient evaluations: 63
    

## Model Evaluation <a name="evaluation"></a>


**evaluation**

* *single_customer_evaluation()* - Compares Model prediction to Ground Truth of randomly sampled customer from the dataset.



* *root_mean_squared_error()* - Computes Root Mean Squared Error of model frequency predictions vs frequency holdout.



* *evaluation_plots()* - 4 Plots for model evaluation.
        tracking - Tracking Cumulative transactions and Daily transactions.
        repeated - Frequency of Repeated Purchases.
        calibration_holdout - Calibration vs Holdout Repeated Purchases.
            


```python
# Evaluation utility functions.

from src.evaluation import single_customer_evaluation
from src.evaluation import root_mean_squared_error
from src.evaluation import evaluation_plots
```


```python
# Evaluation of an Individual customer predictions by the model.

frequency_predicted, frequency_holdout = single_customer_evaluation()


# Predicted vs Holdout.

print(f"SINGLE CUSTOMER PREDICTIONS:"
          f"\nPrediction:"
          f"\n {frequency_predicted}"
          f"\nGround Truth: "
          f"\n {frequency_holdout}")
```

    SINGLE CUSTOMER PREDICTIONS:
    Prediction:
     29881    0.008178
    dtype: float64
    Ground Truth: 
     29881    0.0
    Name: frequency_holdout, dtype: float64
    


```python
# Overall Root Mean Squared Error of Predictions.

rmse = root_mean_squared_error()

print(f"RMSE: {rmse}")
```

    RMSE: 0.14444759935762416
    


```python
# Calibration vs Holdout Plot.

evaluation_plots(plot_type="calibration_holdout");
```


![png](output_17_0.png)



```python
# Cumulative Transactions and Daily Transactions plot.

evaluation_plots(plot_type="tracking");
```


![png](output_18_0.png)



```python
# Repeated Frequency of Transactions plot.

evaluation_plots(plot_type="repeated");
```

    C:\Program Files\Anaconda\lib\site-packages\lifetimes\generate_data.py:54: RuntimeWarning: divide by zero encountered in double_scalars
      next_purchase_in = random.exponential(scale=1.0 / l)
    


![png](output_19_1.png)


## Model Interpretation <a name="inter"></a>


```python
# Imports for Model Interpretation.

from lifetimes import BetaGeoFitter

from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_probability_alive_matrix
```


```python
# Loading Dataset used Training.

summary = pd.read_csv("datasets/summary.csv")
customer_id_mapping = pd.read_csv("datasets/customer_mapping.csv")
transactions = pd.read_csv("datasets/transactions.csv", parse_dates=["order_purchase_timestamp", "order_date"])
```


```python
# Setting Trained Customer_Lifetime_Estimator.

model = BetaGeoFitter()
model.load_model("models/customer_lifetime_estimator.pkl")
```


### Frequency/Recency Analysis <a name="rf"></a>

Analyzing relation between frequency-recency-expected number of future purchases, 30 days forecast generated by the model using plot below.


*Frequency* - Repeated purchases the customer has made.


*Recency* - Age at last purchase viz., (first purchase - last purchase) days

### Customer Segmentation <a name="segment"></a>


*Best Customers*

The model predicts that the best set of customers are the ones in bottom right, with historical recency of 400-600, frequency of 10-15. Who are likely to make about 6 purchase in next 30 days.

*Coldest Customers*

The top right customers who have historical recency of 0-200, frequency of 10-15 are likely to almost no purchases. 


```python
# Frequency-Recency-Expected Number of Future Purchases.

plot_frequency_recency_matrix(model=model,
                              T=30, 
                              max_frequency=None);
```


![png](output_25_0.png)


### Probability Alive Matrix <a name="alive"></a>
This plot depicts relation between frequency-recency-probability a customer is Alive in the context of will they be placing an order in the future.

*Interpretation*

Customer who has made a purchase after 200 days of their first purchase and has been making about 7 purchases, has a probability of 0.2 of them coming back to make a purchase.


```python
# Probability the customer is alive.

plot_probability_alive_matrix(model=model);
```


![png](output_27_0.png)


### High Probability Customers. <a name="highprob"></a>

Finding Insights from historical transactions of Customers who are likely to make a purchase in next 30 days.


**Insight Obtained** 

The Customers who have high probability of making a purchase, have made purchases using "Creadit Card". This information could be used to devise marketing plans.


```python
# Predictions.

frequency_predictions = model.predict(t=30,
                                     frequency=summary["frequency"],
                                     recency=summary["recency"],
                                     T=summary["T"])

summary["frequency_predictions"] = frequency_predictions.copy()
```


```python
# Top 10 Likely to Purchase customers.

top_ten = summary.sort_values("frequency_predictions", ascending=False).head(10)

# Extracting IDs using Customer ID Mapping dataset.

top_ten_ids = customer_id_mapping.iloc[top_ten.index]

# List of ids.

top_ten_ids_list = list(top_ten_ids.customer_unique_id.values)

# Their Transactions.

historical_transactions_of_top_ten = transactions[transactions.customer_unique_id.isin(top_ten_ids_list)]
```


```python
historical_transactions_of_top_ten.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_unique_id</th>
      <th>order_id</th>
      <th>order_purchase_timestamp</th>
      <th>payment_value</th>
      <th>payment_type</th>
      <th>year_month</th>
      <th>order_date</th>
      <th>avg_inter_purchase_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9401</th>
      <td>8d50f5eadf50201ccdcedfb9e2ac8455</td>
      <td>5d848f3d93a493c1c8955e018240e7ca</td>
      <td>2017-05-15 23:30:03</td>
      <td>22.77</td>
      <td>credit_card</td>
      <td>2017-05</td>
      <td>2017-05-15</td>
      <td>28.875</td>
    </tr>
    <tr>
      <th>13399</th>
      <td>8d50f5eadf50201ccdcedfb9e2ac8455</td>
      <td>369634708db140c5d2c4e365882c443a</td>
      <td>2017-06-18 22:56:48</td>
      <td>51.75</td>
      <td>credit_card</td>
      <td>2017-06</td>
      <td>2017-06-18</td>
      <td>28.875</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Payment Methods Used Historical by Top 10 Customers.

sns.countplot(historical_transactions_of_top_ten["payment_type"]);
```


![png](output_32_0.png)


### Forecast for Randomly Selected Customer in dataset. <a name="forecast"></a>

**Insights Obtained**
* *Number of Purchases* - The randomly sampled customer is not very likely to make a purchase in next 30 days.
* *Probability Alive* - There is a very low probablity that they will be placing an order anytime soon.


```python
# Prediction Module for Local Package.

from src.predict import number_of_purchases, probability_alive
```


```python
# Customers with repeated purchases.

repeated_customers = summary.loc[summary["frequency"] >= 2]

# Randomly sampling one customer.

random_sampled_customer = repeated_customers.sample()

random_sampled_customer
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
      <th>recency</th>
      <th>T</th>
      <th>frequency_predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79299</th>
      <td>2.0</td>
      <td>125.0</td>
      <td>220.0</td>
      <td>0.044719</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Predicting Number of Purchases 30 days in future.

number_of_purchases(historical_rfm_data=random_sampled_customer,
                    time_units=30)
```




    79299    0.044719
    dtype: float64




```python
# Predicting Probability that they will be placing an order based on Historical Transactions.

probability_alive(historical_rfm_data=random_sampled_customer)
```




    array([0.22415287])



**END**
