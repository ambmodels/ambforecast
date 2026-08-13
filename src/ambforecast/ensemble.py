"""Ensemble forecasts."""

import pandas as pd


def ensemble(forecasts):
    """Calculate the mean of multiple forecasts.

    This function will work with the outputs of run_single_forecast() and
    run_forecasts().

    Parameters
    ----------
    forecasts : list[pd.DataFrame]
        Forecast dataframes to combine.

    Returns
    -------
    ensemble : pd.DataFrame
        Ensemble forecast.

    """
    keys = ["ds", "metric", "area"]

    # Checking for issues before combining
    reference = forecasts[0][keys].sort_values(keys).reset_index(drop=True)
    for forecast in forecasts:
        if forecast.duplicated(keys).any():
            raise ValueError(
                "Each forecast must have one row per date, metric, and area."
            )
        forecast_keys = forecast[keys].sort_values(keys).reset_index(drop=True)
        if not forecast_keys.equals(reference):
            raise ValueError(
                "All forecasts must cover the same dates, metrics, and areas."
            )

    return (
        pd.concat(forecasts, ignore_index=True)
        .groupby(keys, as_index=False)
        .agg(
            {
                "forecast": "mean",
                "pi_lower": "mean",
                "pi_upper": "mean",
            }
        )
        .sort_values(["ds", "metric", "area"])
    )
