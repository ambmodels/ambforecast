"""Ensemble forecasts."""

import pandas as pd


def ensemble(forecasts):
    """Calculate the mean of multiple forecasts.

    Parameters
    ----------
    forecasts : list[pd.DataFrame]
        Forecast dataframes to combine.

    Returns
    -------
    pd.DataFrame
        Ensemble forecast.
    """
    # Each dataframe in the list should be the result for a single model,
    # metric and area
    for forecast in forecasts:
        if forecast["ds"].duplicated().any():
            raise ValueError("Each forecast must have one row per date.")

    # The provided forecasts must all cover the same dates
    dates = forecasts[0]["ds"]
    for forecast in forecasts[1:]:
        if not forecast["ds"].equals(dates):
            raise ValueError("All forecasts must cover the same dates.")

    # Combine into single dataframe
    return (
        pd.concat(forecasts)
        .groupby("ds")
        .agg({"forecast": "mean", "pi_lower": "mean", "pi_upper": "mean"})
        .reset_index()
    )
