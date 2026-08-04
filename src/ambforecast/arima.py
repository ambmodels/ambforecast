"""ARIMA forecast."""

from datetime import timedelta
import pandas as pd
import statsmodels.api as sm


def predict_arima(historic, holidays, forecast_length):
    """Predict future demand using ARIMA.

    Parameters
    ----------
    historic : pd.DataFrame
        DataFrame which must include two columns: "ds" (date) and
        "y" (value). This should have been filtered to the relevant metric
        and county.
    holidays : pd.DataFrame
        DataFrame which must include two columns: "ds" (date) and "holiday"
        (name of holiday). This should have been filtered to the relevant
        county.
    forecast_length : int
        Number of days to forecast.

    Returns
    -------
    forecast : pd.DataFrame
        Forecast dataframe.
    """
    # Set the dates as the index
    arima_historic = historic.set_index("ds")

    # Create dataframe where index is each date from historic and column is
    # "holiday" which is True when the date is listed as a holiday. This just
    # uses the date - it doesn't use lower_window and upper_window
    holiday_dummy = pd.DataFrame(
        {"holiday": arima_historic.index.isin(holidays["ds"])},
        index=arima_historic.index
    )

    # Fit ARIMA model
    model = sm.tsa.arima.ARIMA(
        endog=arima_historic,
        exog=holiday_dummy,
        order=(1, 1, 3),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        freq="D"
    )
    model = model.fit()

    # Create index of dates to make prediction for
    prediction_dates = pd.date_range(
        start=arima_historic.index[-1] + timedelta(days=1),
        periods=forecast_length,
    )

    # Encode holidays for prediction dates
    holiday_dummy = pd.DataFrame(
        {"holiday": prediction_dates.isin(holidays["ds"])},
        index=prediction_dates
    )

    # Get forecast for those dates and extract summary dataframe
    model_forecast = model.get_forecast(forecast_length, exog=holiday_dummy)
    forecast = model_forecast.summary_frame(alpha=0.05)

    # Rearranging/relabelling forecast dataframe
    forecast = forecast.rename_axis("ds").drop("mean_se", axis=1)
    forecast = forecast.rename(columns={
        "mean": "arima_mean",
        "mean_ci_lower": "arima_lower_95",
        "mean_ci_upper": "arima_upper_95"
    })
    return forecast
