"""Holt-Winters Exponential Smoothing.

* Exponential smoothing has additive errors and seasonality.
* ETS model has additive or multiplicative errors, trend and seasonality.
"""

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.statespace.exponential_smoothing import (
    ExponentialSmoothing,
)
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from .structures import CustomRepr


# ----------------------------------------------------------------------------
# Version with additive errors and seasonality
# ----------------------------------------------------------------------------

@dataclass(kw_only=True, repr=False)
class ExponentialSmoothingParams(CustomRepr):
    """Parameters for the exponential smoothing model.

    Parameters
    ----------
    trend : bool
        Whether to include a trend component.
    damped_trend : bool
        Whether the included trend component is damped.
    seasonal : int
        Number of periods in a seasonal cycle - e.g., 7 for daily data with
        a weekly cycle.
    interval_width : float
        Width of the prediction intervals - for example, 0.95 will produce
        95% prediction intervals.

    """

    trend: bool = True
    damped_trend: bool = True
    seasonal: int = 7
    interval_width: float = 0.95


def exponential_smoothing(train, params, test=None, horizon=None):
    """Fit exponential smoothing model and generate forecast.

    Parameters
    ----------
    train : pd.DataFrame
        Historic training data.
    params : ExponentialSmoothingParams
        Parameters controlling the model.
    test : pd.DataFrame | None
        Data containing the dates to forecast in a `ds` column. Typically
        a held-out test set or cross-validation fold.
    horizon : int | None
        Number of days to forecast, after the final date in `train`. To be
        used if there is no test set (e.g., if actually predicting into future
        with no data to compare against).

    Returns
    -------
    pd.DataFrame
        Forecast dataframe.

    """
    if (test is None) == (horizon is None):
        raise ValueError("Provide exactly one of 'test' or 'horizon'.")

    # Convert training data into required format
    train = train.sort_values("ds").set_index("ds")["y"]
    train = train.asfreq("D")

    # Fit model
    model = ExponentialSmoothing(
        endog=train,
        trend=params.trend,
        damped_trend=params.damped_trend,
        seasonal=params.seasonal,
    )
    fitted = model.fit()

    # Get dates for forecast
    if test is not None:
        forecast_dates = test["ds"].reset_index(drop=True)
    else:
        forecast_dates = pd.date_range(
            start=train.index.max() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

    # Get forecast for those dates and extract summary dataframe
    pred = fitted.get_forecast(len(forecast_dates))
    forecast = pred.summary_frame(alpha=1 - params.interval_width)

    # Rearranging/relabelling forecast dataframe
    # statsmodels labels these as confidence intervals, but they are
    # actually better described as approximate prediction intervals
    # See: https://github.com/statsmodels/statsmodels/issues/8230
    forecast = (
        forecast.rename_axis(columns=None)
        .reset_index(names="ds")
        .drop("mean_se", axis=1)
        .rename(
            columns={
                "mean": "forecast",
                "mean_ci_lower": "pi_lower",
                "mean_ci_upper": "pi_upper"
            }
        )
    )
    return forecast


# ----------------------------------------------------------------------------
# Version with additive or multiplicative errors, trend and seasonality
# ----------------------------------------------------------------------------

@dataclass(kw_only=True, repr=False)
class ETSParams(CustomRepr):
    """Parameters for the ETS (Error, Trend and Seasonality) model.

    Parameters
    ----------
    error : str
        Error model - can be None, "additive" or "multiplicative".
    trend : str
        Trend component - can be None, "additive" or "multiplicative".
    damped_trend : bool
        Whether the included trend component is damped.
    seasonal : str
        Seasonality model - can be None, "additive" or "multiplicative".
    seasonal_periods : int
        Number of periods in a seasonal cycle - e.g., 7 for daily data with
        a weekly cycle.
    interval_width : float
        Width of the prediction intervals - for example, 0.95 will produce
        95% prediction intervals.

    """

    error : str = "multiplicative"
    trend: str = "multiplicative"
    damped_trend: bool = True
    seasonal: str = "multiplicative"
    seasonal_periods: int = 7
    interval_width: float = 0.95


def ets(train, params, test=None, horizon=None):
    """Fit exponential smoothing model and generate forecast.

    Parameters
    ----------
    train : pd.DataFrame
        Historic training data.
    params : ETSParams
        Parameters controlling the model.
    test : pd.DataFrame | None
        Data containing the dates to forecast in a `ds` column. Typically
        a held-out test set or cross-validation fold.
    horizon : int | None
        Number of days to forecast, after the final date in `train`. To be
        used if there is no test set (e.g., if actually predicting into future
        with no data to compare against).

    Returns
    -------
    pd.DataFrame
        Forecast dataframe.

    """
    if (test is None) == (horizon is None):
        raise ValueError("Provide exactly one of 'test' or 'horizon'.")

    # Convert training data into required format
    train = train.sort_values("ds").set_index("ds")["y"]
    train = train.asfreq("D")

    # Fit model
    model = ETSModel(
        endog=train,
        error=params.error,
        trend=params.trend,
        damped_trend=params.damped_trend,
        seasonal=params.seasonal,
        seasonal_periods=params.seasonal_periods
    )
    fitted = model.fit()

    # Get dates for forecast
    if test is not None:
        forecast_dates = test["ds"].reset_index(drop=True)
    else:
        forecast_dates = pd.date_range(
            start=train.index.max() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

    # Get forecast for those dates and extract summary dataframe
    horizon = len(forecast_dates)
    pred = fitted.get_prediction(
        start=len(train),
        end=len(train) + horizon - 1
    )
    forecast = pred.summary_frame(alpha=1 - params.interval_width)

    forecast = (
        forecast.reset_index(names="ds")
        .drop("mean_numerical", axis=1)
        .rename(columns={"mean": "forecast"})
    )
    return forecast
