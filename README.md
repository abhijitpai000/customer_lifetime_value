# Customer Lifetime Value.
Forecasting Number of transactions a customer would make using Beta Geometric-Negative Binomial Distribution, a BTYD-Probabilistic Model.

## Overview
Trained a Beta Geometric-Negative Binomial Distribution (BG/NBD) model that explains how frequently customers make purchases while they are still "alive" and how likely a customer is to churn in any given time period, using customer transactions of E-Commerce store Olist [public dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)

**Model Outcome**


Trained Model Predicts Number of Purchases with a RMSE of 0.144 and is able to capture 99% of customer historical transactions with frequency less than or equal to 4, Less than 1% of customers have greater than 4 repeated purchases in the dataset.

![image.png](https://github.com/abhijitpai000/customer_lifetime_value/blob/master/report/figures/output_17_0.png)




*Model vs Actual - Cumulative Transactions and Daily Transactions*
![image-3.png](https://github.com/abhijitpai000/customer_lifetime_value/blob/master/report/figures/output_18_0.png)


## About Olist Dataset

The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil, the orders are divided into 9 .csv files in a relational database schema.

For this study, I have aggregrated data using the following 3 .csv files out of the 9 in the zip file.
* olist_customers_dataset.csv
* olist_orders_dataset.csv
* olist_payments_dataset.csv

**Source :** [Olist Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)

## Final Report & Package Walk-Through

To reproduce this study, use modules in 'src' directory of this repo. (setup instructions below) and walk-through of the package is presented in the [final report](https://github.com/abhijitpai000/customer_lifetime_value/blob/master/report/README.md)

## Setup instructions

#### Creating Python environment

This repository has been tested on Python 3.7.6.

1. Cloning the repository:

`git clone https://github.com/abhijitpai000/customer_lifetime_value.git`

2. Navigate to the git clone repository.

`cd customer_lifetime_value`

3. Download raw data from the data source link and place in "datasets" directory

4. Install [virtualenv](https://pypi.org/project/virtualenv/)

`pip install virtualenv`

`virtualenv clv`

5. Activate it by running:

`clv/Scripts/activate`

6. Install project requirements by using:

`pip install -r requirements.txt`

**Note**
* For make_dataset(), please place the unzipped raw data (from data source) in the 'datasets' directory.
