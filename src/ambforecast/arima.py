"""Forecasts using ARIMA."""

import warnings
from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .structures import CustomRepr


@dataclass(kw_only=True, repr=False)
class ARIMAParams(CustomRepr):
    """Parameters for the ARIMA forecast model.

    Parameters
    ----------
    holidays : pd.DataFrame | None
        Holiday dataframe. If None, no holiday effects are fitted.
    order : tuple
        The (p, d, q) order of the model.
    seasonal_order : tuple
        The (P, D, Q, s) order of the seasonal component of the model.
    enforce_stationarity : bool
        Whether or not to require the autoregressive parameters to correspond
        to a stationarity process.
    max_iter : int
        The maximum number of iterations. Using ARIMA default (50), we did
        observe a warning that "Maximum Likelihood optimisation failed to
        converge". This warning can be resolved by increasing the maximum.
    interval_width : float
        Width of the prediction intervals - for example, 0.95 will produce
        95% prediction intervals.

    """

    holidays: pd.DataFrame | None = None

    # ARIMA default
    order: tuple = (0, 0, 0)
    # ARIMA default
    seasonal_order: tuple = (0, 0, 0, 0)
    # ARIMA default
    enforce_stationarity: bool = True
    # ARIMA default
    max_iter: int = 50

    interval_width: float = 0.95


def encode_holidays(dates, holiday_dates):
    """Create dataframe with encoded holidays for ARIMA.

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
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {"holiday": dates.isin(holiday_dates)}, index=dates
    ).astype(int)


def arima(train, params, test=None, horizon=None):
    """Fit ARIMA model and generate forecast.

    Parameters
    ----------
    train : pd.DataFrame
        Historic training data.
    params : ARIMAParams
        Parameters controlling the ARIMA model.
    test : pd.DataFrame | None
        Data containing the dates to forecast in a `ds` column. Typically
        a held-out test set or cross-validation fold.
    horizon : int | None
        Number of days to forecast, after the final date in `train`. To be
        used if there is no test set (e.g., if actually predicting into future
        with no data to compare against).

    Returns
    -------
    forecast : pd.DataFrame
        Forecast dataframe.

    """
    if (test is None) == (horizon is None):
        raise ValueError("Provide exactly one of 'test' or 'horizon'.")

    # ARIMA requires an array - so set date as index and just extract y values
    arima_train = train.set_index("ds")["y"]
    arima_train.index.freq = "D"

    # Create dataframe where index is each date from the training data and
    # column is "holiday" which is 1 when the date is listed as a holiday and
    # 0 otherwise. This just uses the date - it doesn't use lower_window and
    # upper_window
    if params.holidays is not None:
        arima_holidays = encode_holidays(
            dates=arima_train.index, holiday_dates=params.holidays["ds"]
        )
    else:
        arima_holidays = None

    # Fit ARIMA model
    model = sm.tsa.arima.ARIMA(
        endog=arima_train,
        exog=arima_holidays,
        order=params.order,
        seasonal_order=params.seasonal_order,
        enforce_stationarity=params.enforce_stationarity,
        freq="D",
    )

    # threadpool_limits is required to ensure consistency on Linux when
    # running sequentially v.s., in parallel. This is because statsmodels
    # relies on NumPy/SciPy, which do their maths using BLAS - a library
    # that can split calculations across multiple threads. Running in
    # parallel changes how those threads get shared out, which can nudge
    # the numbers slightly and lead to a different result. Forcing BLAS
    # to use just one thread keeps things running the same way every time.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = model.fit(method_kwargs={"maxiter": params.max_iter})

    # Create index of dates to make prediction for
    if test is not None:
        forecast_dates = test["ds"].reset_index(drop=True)
    else:
        forecast_dates = pd.date_range(
            start=arima_train.index.max() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

    # Encode holidays for prediction dates
    if params.holidays is not None:
        forecast_holidays = encode_holidays(
            dates=forecast_dates, holiday_dates=params.holidays["ds"]
        )
    else:
        forecast_holidays = None

    # Get forecast for those dates and extract summary dataframe
    model_forecast = model.get_forecast(
        steps=len(forecast_dates), exog=forecast_holidays
    )
    forecast = model_forecast.summary_frame(alpha=1 - params.interval_width)

    # Rearranging/relabelling forecast dataframe
    # statsmodels ARIMA labels these as confidence intervals, but they are
    # actually better described as approximate prediction intervals
    # See: https://github.com/statsmodels/statsmodels/issues/8230
    forecast = (
        forecast.rename_axis("ds")
        .reset_index()
        .drop("mean_se", axis=1)
        .rename(
            columns={
                "mean": "forecast",
                "mean_ci_lower": "pi_lower",
                "mean_ci_upper": "pi_upper",
            }
        )
    )
    return forecast
