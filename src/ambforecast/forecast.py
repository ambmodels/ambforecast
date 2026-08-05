"""Run forecast."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .arima import predict_arima
from .prophet import predict_prophet


def run_forecast(df_historic, df_holidays, metrics, method, forecast_length):
    """Run forecast.

    Parameters
    ----------
    df_historic : pd.DataFrame
        Historic data. Should have columns "ds" (date), "currency" (metric),
        "ora" (county/trust) and "y" (value). It should not include dates that
        will be forecast, it should only contain metrics you want to forecast,
        and it shouldn't contain any NA.
    df_holidays : pd.DataFrame
        Holidays. Should have columns "ds" (date), "holiday" (name of
        holiday), "lower_window", "upper_window" and "county".
    metrics : list[str]
        Metric names (values of "currency") to forecast.
    method : str
        Name of forecasting method to run. Either "prophet" or "arima".
    forecast_length : int
        Number of days to generate a forecast for.

    Returns
    -------
    pd.DataFrame
        Forecast results.

    """
    # Filter to requested metrics and drop any incomplete rows
    df_historic = df_historic[df_historic["currency"].isin(metrics)].dropna()

    # Filter to unique pairs of counties and metrics
    unique_pairs = (
        df_historic[["ora", "currency"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Loop through each combination of county and metric
    forecast_list = []
    seed_counter = 0
    for county, metric in tqdm(
        unique_pairs.itertuples(index=False), total=len(unique_pairs)
    ):
        print(f"Running {method} forecast for: {metric} - {county}...")

        # Filter historic data to specified county and metric, then just keep
        # "ds" and "y" col
        historic = df_historic.loc[
            (df_historic["ora"] == county)
            & (df_historic["currency"] == metric),
            ["ds", "y"],
        ]

        # Filter holidays to specified county
        holidays = df_holidays[df_holidays["county"] == county]

        # Generate forecasts
        if method == "prophet":
            seed_counter += 1
            np.random.seed(seed_counter)
            forecast = predict_prophet(
                historic=historic,
                holidays=holidays,
                forecast_length=forecast_length,
                metric=metric,
            )
        if method == "arima":
            forecast = predict_arima(
                historic=historic,
                holidays=holidays,
                forecast_length=forecast_length,
            )

        # Add columns with county and metric name, then save result to list
        forecast.insert(1, "county", county)
        forecast.insert(2, "currency", metric)
        forecast_list.append(forecast)

    return pd.concat(forecast_list)
