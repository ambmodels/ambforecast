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
    ensemble : pd.DataFrame
        Ensemble forecast.
    """
    metric = forecasts[0]["metric"].iloc[0]
    area = forecasts[0]["area"].iloc[0]
    dates = forecasts[0]["ds"]

    for forecast in forecasts:
        if forecast["ds"].duplicated().any():
            raise ValueError("Each forecast must have one row per date.")

        if not (forecast["metric"] == metric).all():
            raise ValueError("All forecasts must have the same metric.")

        if not (forecast["area"] == area).all():
            raise ValueError("All forecasts must have the same area.")

        if not forecast["ds"].equals(dates):
            raise ValueError("All forecasts must cover the same dates.")

    # Combine into single dataframe
    ensemble = (
        pd.concat(forecasts)
        .groupby("ds")
        .agg({"forecast": "mean", "pi_lower": "mean", "pi_upper": "mean"})
        .reset_index()
    )
    ensemble.insert(1, "metric", metric)
    ensemble.insert(2, "area", area)

    return ensemble
