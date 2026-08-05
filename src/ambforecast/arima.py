"""ARIMA forecast."""

from datetime import timedelta

import pandas as pd
import statsmodels.api as sm


def encode_holidays(dates, holiday_dates):
    """Create dataframe with encoded holidays for ARIMA.

    For each date, it is either marked as a holiday (1) or not (0).

    Parameters
    ----------
    dates : pd.Series | pd.Index
        Dates in the data that is being fit or predicted.
    holiday_dates : pd.Series | pd.Index
        Dates that holidays are on.

    Returns
    -------
    pd.DataFrame
        The index is each date from dates and then the column "holiday" marks
        whether each date was in holiday_dates or not.

    """
    return pd.DataFrame(
        {"holiday": dates.isin(holiday_dates)}, index=dates
    ).astype(int)


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
    arima_historic.index.freq = "D"

    # Create dataframe where index is each date from historic and column is
    # "holiday" which is 1 when the date is listed as a holiday or 0
    # otherwise. This just uses the date - it doesn't use lower_window and
    # upper_window
    holiday_dummy = encode_holidays(
        dates=arima_historic.index, holiday_dates=holidays["ds"]
    )

    # Fit ARIMA model
    model = sm.tsa.arima.ARIMA(
        endog=arima_historic,
        exog=holiday_dummy,
        order=(1, 1, 3),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        freq="D",
    )
    model = model.fit()

    # Create index of dates to make prediction for
    prediction_dates = pd.date_range(
        start=arima_historic.index[-1] + timedelta(days=1),
        periods=forecast_length,
    )

    # Encode holidays for prediction dates
    holiday_dummy = encode_holidays(
        dates=prediction_dates, holiday_dates=holidays["ds"]
    )

    # Get forecast for those dates and extract summary dataframe
    model_forecast = model.get_forecast(forecast_length, exog=holiday_dummy)
    forecast = model_forecast.summary_frame(alpha=0.05)

    # Rearranging/relabelling forecast dataframe
    # statsmodels ARIMA labels these as confidence intervals, but they are
    # actually better described as approximate prediction intervals
    # See: https://github.com/statsmodels/statsmodels/issues/8230
    forecast = forecast.rename_axis("ds").drop("mean_se", axis=1)
    forecast = forecast.rename(
        columns={
            "mean": "arima_mean",
            "mean_ci_lower": "arima_pi_lower",
            "mean_ci_upper": "arima_pi_upper",
        }
    )
    return forecast
