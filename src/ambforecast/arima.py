"""ARIMA forecast."""

import warnings
from datetime import timedelta

import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def _encode_holidays(dates, holiday_dates):
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


def predict_arima(
    # Data and settings for the forecast
    historic,
    holidays,
    horizon,
    interval_width=0.95,
    # ARIMA parameters
    order=(0, 0, 0),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=True,
    max_iter=50,
):
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
    horizon : int
        Number of days into future that the data is predicted.
    interval_width : float
        Width of the prediction intervals - for example, 0.95 will produce
        95% prediction intervals.
    order : tuple
        The (p, d, q) order of the model. If none specified, will use the
        ARIMA default (0, 0, 0).
    seasonal_order : tuple
        The (P, D, Q, s) order of the seasonal component of the model. If
        none specified, will use the ARIMA default (0, 0, 0, 0).
    enforce_stationarity : bool
        Whether or not to require the autoregressive parameters to correspond
        to a stationarity process. If none specified, will use the ARIMA
        default (True).
    max_iter: int
        The maximum number of iterations. If none specified, will use the
        ARIMA default (50) - although we did find this produced a warning that
        "Maximum Likelihood optimisation failed to converge". This warning can
        be resolved by increasing the maximum.

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
    holiday_dummy = _encode_holidays(
        dates=arima_historic.index, holiday_dates=holidays["ds"]
    )

    # Fit ARIMA model
    model = sm.tsa.arima.ARIMA(
        endog=arima_historic,
        exog=holiday_dummy,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
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
        model = model.fit(method_kwargs={"maxiter": max_iter})

    # Create index of dates to make prediction for
    prediction_dates = pd.date_range(
        start=arima_historic.index[-1] + timedelta(days=1),
        periods=horizon,
    )

    # Encode holidays for prediction dates
    holiday_dummy = _encode_holidays(
        dates=prediction_dates, holiday_dates=holidays["ds"]
    )

    # Get forecast for those dates and extract summary dataframe
    model_forecast = model.get_forecast(horizon, exog=holiday_dummy)
    forecast = model_forecast.summary_frame(alpha=1 - interval_width)

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
