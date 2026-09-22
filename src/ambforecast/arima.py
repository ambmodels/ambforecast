"""Forecasts using ARIMA."""

import warnings
from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .helpers import CustomRepr, merge_regressor


@dataclass(kw_only=True, repr=False)
class ARIMARegressor(CustomRepr):
    """Configuration and data for a single ARIMA exogenous regressor.

    Parameters
    ----------
    name : str
        Name of the regressor column in `data`.
    data : pd.DataFrame
        Regressor data with `ds` and column named by `name`. It must cover
        every training and forecast date required by the model.

    """

    name: str
    data: pd.DataFrame


@dataclass(kw_only=True, repr=False)
class ARIMAParams(CustomRepr):
    """Parameters for the ARIMA model.

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
    regressors : tuple[ARIMARegressor, ...]
        Additional regressors to add before fitting.
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

    regressors: tuple[ARIMARegressor, ...] = ()
    interval_width: float = 0.95


def encode_holidays(dates, holidays):
    """Create dataframe with encoded holidays for ARIMA.

    Parameters
    ----------
    dates : pd.Series
        Dates in the data that is being fit and predicted.
    holidays : pd.DataFrame
        Holiday dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with `ds` , `area` and binary `holiday` column.

    """
    # Get a list of all areas
    areas = holidays["area"].unique()

    # Build one row for every date x area combination
    date_area_grid = pd.MultiIndex.from_product(
        [dates, areas],
        names=["ds", "area"]
    ).to_frame(index=False)

    # Convert holiday data to a binary indicator of date x area with holiday
    holiday_flag = (
        holidays
        .assign(holiday=1)[["ds", "holiday", "area"]]
        .drop_duplicates()
    )

    # Add flags to every date x area row - dates absent from holiday become 0
    return date_area_grid.merge(
        holiday_flag,
        on=["ds", "area"],
        how="left"
    ).fillna({"holiday": 0}).astype({"holiday": int})


def arima(train, params, test=None, horizon=None):
    """Fit ARIMA model and generate forecast.

    Parameters
    ----------
    train : pd.DataFrame
        Historic training data.
    params : ARIMAParams
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
    forecast : pd.DataFrame
        Forecast dataframe.

    """
    if (test is None) == (horizon is None):
        raise ValueError("Provide exactly one of 'test' or 'horizon'.")

    # ARIMA requires an array - so set date as index and just extract y values
    arima_train = train.set_index("ds")["y"]
    arima_train.index.freq = "D"

    # Create index of dates to make prediction for
    if test is not None:
        future = test[["ds"]].copy()
    else:
        forecast_dates = pd.date_range(
            start=arima_train.index.max() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        future = pd.DataFrame({"ds": forecast_dates})

    # Create dataframe where index is each date from the training data and
    # column is "holiday" which is 1 when the date is listed as a holiday and
    # 0 otherwise. This just uses the date - it doesn't use lower_window and
    # upper_window. Then add this dataframe to the list of regressors.
    all_dates = pd.concat([train["ds"], future["ds"]], ignore_index=True)
    if params.holidays is None:
        regressors = params.regressors
    else:
        holiday = ARIMARegressor(
            name="holiday",
            data=encode_holidays(
                dates=all_dates, holidays=params.holidays
            ),
        )
        regressors = (*params.regressors, holiday)

    # Add regressor data to the training data
    # Will only run loop if regressors are provided
    for regressor in regressors:
        train = merge_regressor(data=train, regressor=regressor)

    # Construct dataframe of exogenous regressors
    regressor_names = [regressor.name for regressor in regressors]
    if regressor_names:
        arima_exog = train.set_index("ds")[regressor_names]
        arima_exog.index.freq = "D"
    else:
        arima_exog = None

    # Fit ARIMA model
    model = sm.tsa.arima.ARIMA(
        endog=arima_train,
        exog=arima_exog,
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

    # Add regressor data to test data
    for regressor in regressors:
        future = merge_regressor(data=future, regressor=regressor)

    # Construct dataframe of exogneous regressors
    if regressor_names:
        forecast_exog = future.set_index("ds")[regressor_names]
    else:
        forecast_exog = None

    # Get forecast for those dates and extract summary dataframe
    model_forecast = model.get_forecast(steps=len(future), exog=forecast_exog)
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
