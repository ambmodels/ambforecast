"""Functions for calculating forecast errors."""

import pandas as pd
from forecast_tools.metrics import (
    coverage,
    mean_absolute_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)


def forecast_errors(forecast, horizon=None):
    """Calculate forecast accuracy measures for a single metric/area/fold.

    Measures included:

    - RMSE (root mean squared error)
    - MASE (mean absolute scaled error)
    - sMAPE (symmetric mean absolute percentage error)
    - Coverage (prediction interval coverage)

    Parameters
    ----------
    forecast : pd.DataFrame
        Forecast results.
    horizon : int | None
        Number of forecast days to include. If None, uses the full forecast.

    Returns
    -------
    dict
        Dictionary of forecast accuracy measures.

    """
    forecast = forecast.sort_values("ds").reset_index(drop=True)

    if horizon is not None:
        forecast = forecast.iloc[:horizon]

    return {
        "rmse": root_mean_squared_error(
            y_true=forecast["actual"], y_pred=forecast["forecast"]
        ),
        # Assumes every row in forecast_group shares the same mae_insample
        "mase": (
            mean_absolute_error(forecast["actual"], forecast["forecast"])
            / forecast["mae_insample"].iloc[0]
        ),
        "smape": symmetric_mean_absolute_percentage_error(
            y_true=forecast["actual"], y_pred=forecast["forecast"]
        ),
        "coverage": coverage(
            y_true=forecast["actual"],
            pred_intervals=forecast[["pi_lower", "pi_upper"]].values.tolist(),
        ),
    }


def calculate_errors(forecast, error_horizons=(7, 14, 21, 28, 35, 42)):
    """Calculate forecast accuracy measures for every metric/area/fold.

    Works for ordinary forecasts, cross-validation forecasts, and ensembles,
    as long as `forecast` contains `actual` and `mae_insample`.

    Parameters
    ----------
    forecast : pd.DataFrame
        Forecast results.
    error_horizons: list
        Horizons to calculate error at.

    Returns
    -------
    dict
        Dictionary of forecast accuracy measures.

    """
    # Group by metric and area, and optionally fold if included
    group_keys = ["metric", "area"]
    if "fold" in forecast.columns:
        group_keys = ["fold"] + group_keys

    errors = []

    for key, forecast_group in forecast.groupby(group_keys, sort=False):
        for horizon in error_horizons:
            error = forecast_errors(forecast_group, horizon=horizon)
            errors.append(
                {
                    # This line reconstructs the columns used to identify
                    # the group and adds them to the error row
                    **dict(zip(group_keys, key, strict=True)),
                    "forecast_start_date": forecast_group["ds"].min(),
                    "horizon": horizon,
                    **error,
                }
            )
    return pd.DataFrame(errors)
