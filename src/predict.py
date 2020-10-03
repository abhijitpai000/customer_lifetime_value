"""Predict Number of Purchases and Probability of Alive using customer_lifetime_estimator"""

from lifetimes import BetaGeoFitter


def number_of_purchases(historical_rfm_data, time_units=30):
    """
    Predicted Conditional Expected Number of Purchases.

    Parameters
    ----------
        historical_rfm_data: Historical Frequency, Recency & T of an individual

        time_units: int, default=30.
            Number of days for predictions.
    Returns
    -------
        expected number of purchases.
    """
    clv_model = BetaGeoFitter(penalizer_coef=0.0)
    clv_model.load_model(path="models/customer_lifetime_estimator.pkl")
    frequency_predictions = clv_model.predict(t=time_units,
                                              frequency=historical_rfm_data["frequency"],
                                              recency=historical_rfm_data["recency"],
                                              T=historical_rfm_data["T"])
    return frequency_predictions


def probability_alive(historical_rfm_data):
    """
    Predicted Conditional Probability Alive.

    Parameters
    ----------
        historical_rfm_data: Historical Frequency, Recency & T of an individual

    Returns
    -------
        Conditional Probability Alive.
    """
    clv_model = BetaGeoFitter(penalizer_coef=0.0)
    clv_model.load_model(path="models/customer_lifetime_estimator.pkl")

    alive_probability = clv_model.conditional_probability_alive(frequency=historical_rfm_data["frequency"],
                                                                recency=historical_rfm_data["recency"],
                                                                T=historical_rfm_data["T"])
    return alive_probability
