"""Forecasts using Meta's Prophet package."""

import logging
from dataclasses import dataclass
from typing import Literal

import cmdstanpy
import pandas as pd
from prophet import Prophet

from .structures import CustomRepr


@dataclass(kw_only=True, repr=False)
class ProphetRegressor(CustomRepr):
    """Configuration and data for a single Prophet regressor.

    For full documentation on Prophet regressors see:
    https://facebook.github.io/prophet/api/prophet.html#Prophet.add_regressor

    Parameters
    ----------
    name : str
        Name of the regressor column in `data`.
    data : pd.DataFrame
        Regressor data with `ds` and column named by `name`. It must cover
        every training and forecast date required by the model.
    prior_scale : float | None
        Strength of regularisation. If None, Prophet uses
        `holidays_prior_scale`.
    standardize : bool | {"auto"}
        Whether Prophet standardises the regressor. `"auto"` standardises
        non-binary regressors.
    mode : {"additive", "multiplicative"} | None
        How the regressor affects the forecast. If None, uses the model's
        `seasonality_mode`.

    """

    name: str
    data: pd.DataFrame
    prior_scale: float | None = None
    standardize: bool | Literal["auto"] = "auto"
    mode: Literal["additive", "multiplicative"] | None = None


@dataclass(kw_only=True, repr=False)
class ProphetParams(CustomRepr):
    """Parameters for the Prophet model.

    For full range of parameters see:
    https://facebook.github.io/prophet/api/prophet.html

    Parameters
    ----------
    holidays : pd.DataFrame | None
        Holiday dataframe. If None, no holiday effects are fitted.
    yearly_seasonality : bool | int | {"auto"}
        Whether to fit yearly seasonality, which is a repeating pattern across
        a calendar year (e.g., peaks in certain seasons).
    weekly_seasonality : bool | int | {"auto"}
        Whether to fit weekly seasonality, which is a repeating pattern within
        a single week (e.g., peaks on certain days of the week).
    daily_seasonality : bool | int | {"auto"}
        Whether to fit daily seasonality, which is a repeating pattern within
        a day (e.g., peaks at certain hours of day).
    seasonality_mode : {"additive", "multiplicative"}
        Either "additive" (seasonal effect is add to the trend, so size of
        effect is constant) or "multiplicative" (seasonality effect grows
        with the trend). May wish to switch to multiplicative if a seasonal
        effect is too large near the start of a series but too small by the
        end.
    changepoint_range : float
        Proportion of history in which trend changepoints will be estimated.
    changepoint_prior_scale : float
        Parameter modulating the flexibility of the automatic changepoint
        selection. Large values will allow many changepoints, small values
        will allow few changepoints.
    seasonality_prior_scale : float
        Parameter modulating strength of seasonality model. Larger values
        allow the model to fit larger seasonal fluctuations, smaller values
        dampen the seasonality.
    holidays_prior_scale : float
        Parameter modulating the strength of the holiday components model.
    regressors : tuple[ProphetRegressor, ...]
        Additional regressors to add before fitting.
    interval_width : float
        Width of the prediction intervals - for example, 0.95 will produce
        95% prediction intervals.
    plot_components : bool
        If True, will display the Prophet plot_components figure.

    """

    # Prophet default
    holidays: pd.DataFrame | None = None

    yearly_seasonality: bool | int | Literal["auto"] = True
    weekly_seasonality: bool | int | Literal["auto"] = True
    daily_seasonality: bool | int | Literal["auto"] = False

    # Prophet default
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"
    # Prophet default
    changepoint_range: float = 0.8
    # Prophet default
    changepoint_prior_scale: float = 0.05
    # Prophet default
    seasonality_prior_scale: float = 10
    # Prophet default
    holidays_prior_scale: float = 10

    regressors: tuple[ProphetRegressor, ...] = ()
    interval_width: float = 0.95
    plot_components: bool = False


def merge_regressor(data, regressor):
    """Merge a regressor and check it covers required dates and area.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing `ds` and `area` columns.
    regressor : ProphetRegressor
        Regressor configuration and data.

    Returns
    -------
    data : pd.DataFrame
        Data with the regressor column added.

    """
    data = pd.merge(
        data,
        regressor.data[["ds", "area", regressor.name]],
        on=["ds", "area"],
        how="left",
        validate="one_to_one",
    )

    missing = data.loc[
        data[regressor.name].isna(),
        ["ds", "area"],
    ]

    if not missing.empty:
        raise ValueError(
            f"Regressor {regressor.name!r} has missing values for:\n"
            f"{missing.to_string(index=False)}"
        )

    return data


def prophet(train, params, test=None, horizon=None):
    """Fit Prophet model and generate forecast.

    Parameters
    ----------
    train : pd.DataFrame
        Historic training data.
    params : ProphetParams
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

    # Disable "start/done processing" from prophet
    # Have to do within predict_prophet() rather than in notebook so that
    # workers to keep these settings when running in parallel
    cmdstanpy.disable_logging()
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    # Set up Prophet model parameters
    model = Prophet(
        holidays=params.holidays,
        yearly_seasonality=params.yearly_seasonality,
        weekly_seasonality=params.weekly_seasonality,
        daily_seasonality=params.daily_seasonality,
        seasonality_mode=params.seasonality_mode,
        changepoint_range=params.changepoint_range,
        changepoint_prior_scale=params.changepoint_prior_scale,
        seasonality_prior_scale=params.seasonality_prior_scale,
        holidays_prior_scale=params.holidays_prior_scale,
        interval_width=params.interval_width,
    )

    # Add regressor data to the training data and add them to the model
    # Will only run loop if regressors are provided
    for regressor in params.regressors:
        train = merge_regressor(data=train, regressor=regressor)
        model.add_regressor(
            name=regressor.name,
            prior_scale=regressor.prior_scale,
            standardize=regressor.standardize,
            mode=regressor.mode,
        )

    # Fit Prophet model using the training data
    model.fit(train)

    # Construct future dataframe based on horizon or on dates in test
    # We set-up with "area" as any regressors will merge based on it.
    if test is None:
        future = model.make_future_dataframe(
            freq="D", periods=horizon, include_history=False
        )
        future["area"] = train["area"].iloc[0]
    else:
        future = test[["ds", "area"]].copy()

    # Add regressor data to the future dataframe
    # Will only run loop if regressors are provided
    for regressor in params.regressors:
        future = merge_regressor(data=future, regressor=regressor)

    # Generate forecast
    forecast = model.predict(future)

    # Plot components
    if params.plot_components:
        model.plot_components(forecast)

    # The lower and upper boundaries from prophet are prediction intervals
    # See: https://facebook.github.io/prophet/docs/diagnostics.html
    forecast = forecast.rename(
        columns={
            "yhat": "forecast",
            "yhat_lower": "pi_lower",
            "yhat_upper": "pi_upper",
        }
    )

    return forecast[["ds", "forecast", "pi_lower", "pi_upper"]]
