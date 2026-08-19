"""Ensemble forecasts."""

import pandas as pd


def ensemble(forecasts):
    """Calculate the mean of multiple forecasts.

    Works with ordinary forecasts and cross-validation forecasts. Shared
    identifier columns are retained.

    Parameters
    ----------
    forecasts : list[pd.DataFrame]
        Forecast dataframes to combine.

    Returns
    -------
    ensemble : pd.DataFrame
        Ensemble forecast.

    """
    # Retain optional identifiers only when every forecast contains them.
    keys = ["ds", "metric", "area"] + [
        column
        for column in ["fold", "forecast_start_date", "actual", "mae_insample"]
        if all(column in forecast.columns for forecast in forecasts)
    ]

    # Checking for issues before combining
    reference = forecasts[0][keys].sort_values(keys).reset_index(drop=True)
    for forecast in forecasts:
        if forecast.duplicated(keys).any():
            raise ValueError(f"Each forecast must have one row per: {keys}.")
        forecast_keys = forecast[keys].sort_values(keys).reset_index(drop=True)
        if not forecast_keys.equals(reference):
            raise ValueError(
                f"All forecasts must have identical values for: {keys}"
            )

    # Calculate ensemble by finding mean of forecast and prediction intervals
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
