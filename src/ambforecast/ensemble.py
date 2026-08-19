"""Ensemble forecasts."""

import pandas as pd


def ensemble(forecasts):
    """Calculate the mean of multiple forecasts.

    Works with ordinary forecasts and cross-validation forecasts. Any shared
    identifier columns (`fold` and `forecast_start_date`) are retained and
    used to align forecasts before averaging.

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

    # Retain optional identifiers only when every forecast contains them.
    keys = ["ds", "metric", "area"] + [
        column
        for column in ["fold", "forecast_start_date"]
        if all(column in forecast.columns for forecast in forecasts)
    ]

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
        .reset_index(drop=True)
    )
