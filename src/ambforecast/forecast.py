"""Forecasting functions (moved from notebook)."""

from prophet import Prophet


def predict_prophet_stf(
    df, forecast_length, metric, holiday, alter_forecast=1
):
    """Define a facebook prophet function.

    changepoint_prior_scale. By default, this parameter is set to 0.05
    https://facebook.github.io/prophet/docs/trend_changepoints.html

    We currently use an altered version of the forecast to include more
    changepoint, set alter_forecast to 0 for the default forecast.

    Keep alter_forecast equal to 1 to alter responses forecast.
    """
    if metric == "Responses" and alter_forecast == 1:
        changepoint_range_value = 1
        changepoint_prior_scale = 0.5
    else:
        # default values
        changepoint_range_value = 0.8
        changepoint_prior_scale = 0.05

    prophet = Prophet(
        holidays=holiday,
        changepoint_range=changepoint_range_value,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=1 - 0.05,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )

    prophet.fit(df)  # Fit historical data.

    future = prophet.make_future_dataframe(
        freq="D", periods=forecast_length, include_history=False
    )  # Change to true to for visualisations
    forecast = prophet.predict(future)
    return forecast, prophet
