"""Prophet forecast."""

from prophet import Prophet


def predict_prophet(historic, holidays, forecast_length, metric):
    """Predict future demand using Prophet.

    Parameters
    ----------
    historic : pd.DataFrame
        DataFrame which must include two columns: "ds" (date) and
        "y" (value). This should have been filtered to the relevant metric
        and county.
    holidays : pd.DataFrame
        DataFrame which must include two columns: "ds" (date) and
        "holiday" (name of holiday). It can also include "lower_window" and
        "upper_window" which extend the holiday out to
        [lower_window, upper_window] days around the date. This should have
        been filtered to the relevant county.
    forecast_length : int
        Number of days to forecast.
    metric : str
        Name of metric being forecast - used to choose changepoint parameters.

    Returns
    -------
    pd.DataFrame
        Forecast dataframe returned by Prophet.predict().
    """
    if metric == "Responses":
        changepoint_range = 1
        changepoint_prior = 0.5
    else:
        # default values
        changepoint_range = 0.8
        changepoint_prior = 0.05

    prophet = Prophet(
        holidays=holidays,
        changepoint_range=changepoint_range,
        changepoint_prior_scale=changepoint_prior,
        interval_width=0.95,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )

    prophet.fit(historic)

    future = prophet.make_future_dataframe(
        freq="D", periods=forecast_length, include_history=False
    )
    return prophet.predict(future)
