"""Prophet forecast."""

import logging

import cmdstanpy
from prophet import Prophet


def predict_prophet(
    # Data and settings for the forecast
    historic,
    holidays,
    horizon,
    interval_width=0.95,
    # Prophet parameters
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_range=0.8,
    changepoint_prior_scale=0.05,
    # Additional regressors: data and parameters
    regressors=None,
    regressors_data=None,
    # Whether to show the Prophet plot_components plot
    plot_components=False
):
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
    horizon : int
        Number of days into future that the data is predicted.
    interval_width : float
        Width of the prediction intervals - for example, 0.95 will produce
        95% prediction intervals.
    weekly_seasonality : bool
        Whether to fit weekly seasonality, which is a repeating pattern within
        a single week (e.g., peaks on certain days of the week).
    yearly_seasonality : bool
        Whether to fit yearly seasonality, which is a repeating pattern across
        a calendar year (e.g., peaks in certain seasons).
    changepoint_range : float
        Proportion of history in which trend changepoints will be estimated.
        Uses Prophet default (0.8) if not specified.
    changepoint_prior_scale : float
        Parameter modulating the flexibility of the automatic changepoint
        selection. Large values will allow many changepoints, small values
        will allow few changepoints. Uses Prophet default (0.05) if not
        specified.
    regressors : dict of dict
        Mapping of regressor names to `Prophet.add_regressor` keyword
        arguments. For example:
            {
                "temp": {"prior_scale": 0.5, "mode": "multiplicative"},
                "rain": {"prior_scale": 0.5, "mode": "multiplicative"},
            }
    regressors_data : pd.DataFrame
        Dataframe containing `ds` plus a column for every configured regressor,
        which one row per date.
    plot_components : bool
        If True, will display the Prophet plot_components figure.

    Returns
    -------
    forecast : pd.DataFrame
        Forecast dataframe.

    """
    # Disable "start/done processing" from prophet
    # Have to do within predict_prophet() rather than in notebook so that
    # workers to keep these settings when running in parallel
    cmdstanpy.disable_logging()
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    prophet = Prophet(
        holidays=holidays,
        interval_width=interval_width,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_range=changepoint_range,
        changepoint_prior_scale=changepoint_prior_scale,
    )

    if regressors_data is not None:
        for name, options in regressors.items():
            prophet.add_regressor(name, **options)
        historic = historic.merge(
            regressors_data,
            on="ds",
            how="left",
            validate="one_to_one"
        )

    prophet.fit(historic)

    future = prophet.make_future_dataframe(
        freq="D", periods=horizon, include_history=False
    )

    if regressors_data is not None:
        future = future.merge(
            regressors_data,
            on="ds",
            how="left",
            validate="one_to_one",
        )

    forecast = prophet.predict(future)

    if plot_components:
        prophet.plot_components(forecast)

    # Choose which columns to keep
    forecast = forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper",
        ]
    ]

    # The lower and upper boundaries from prophet are prediction intervals
    # See: https://facebook.github.io/prophet/docs/diagnostics.html
    forecast = forecast.rename(
        columns={
            "yhat": "forecast",
            "yhat_lower": "pi_lower",
            "yhat_upper": "pi_upper",
        }
    )

    return forecast
