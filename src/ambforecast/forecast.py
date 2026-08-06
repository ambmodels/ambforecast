"""Run forecast."""

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from .arima import predict_arima
from .prophet import predict_prophet


class Forecaster:
    """Run forecasts.

    Attributes
    ----------
    df_holidays : pd.DataFrame
        Holidays.
    metrics : list[str]
        Metrics to forecast.
    forecast_length : int
        Number of days to generate a forecast for.
    cores : int
        Number of CPU cores to use for parallel execution.
    df_historic : pd.DataFrame
        Historic data.
    unique_pairs : pd.DataFrame
        Each row is a unique combination of county and metric from
        df_historic.
    results_list : list[pd.DataFrame]
        List to store forecast results dataframes.

    """

    def __init__(
        self, df_historic, df_holidays, metrics, forecast_length, cores=1
    ):
        """Initialise with data and settings to use across all forecasts.

        Parameters
        ----------
        df_historic : pd.DataFrame
            Historic data. Should have columns "ds" (date), "currency"
            (metric), "ora" (county/trust) and "y" (value). It should not
            include dates that will be forecast, it should only contain
            metrics you want to forecast, and it shouldn't contain any NA.
        df_holidays : pd.DataFrame
            Holidays. Should have columns "ds" (date), "holiday" (name of
            holiday), "lower_window", "upper_window" and "county".
        metrics : list[str]
            Metric names (values of "currency") to forecast.
        forecast_length : int
            Number of days to generate a forecast for.
        cores : int
            Number of CPU cores to use for parallel execution. For all
            available cores, set to -1. For sequential execution, set to 1.

        """
        self.df_holidays = df_holidays
        self.metrics = metrics
        self.forecast_length = forecast_length
        self.cores = cores

        # Filter to requested metrics and drop any incomplete rows
        self.df_historic = df_historic[
            df_historic["currency"].isin(self.metrics)
        ].dropna()

        # Find unique pairs of counties and metrics
        self.unique_pairs = (
            self.df_historic[["ora", "currency"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Empty list to store forecast results
        self.results_list = []

        # Print the forecast start date
        forecast_start_date = df_historic["ds"].max() + pd.Timedelta(days=1)
        print(
            "Forecast start date: "
            + f"{forecast_start_date.strftime('%A %d %B %Y')}"
        )

    def generate_forecast(self, county, metric, method, seed=None):
        """Generate single forecast.

        Parameters
        ----------
        county : str
            Name of county/area to generate forecast for.
        metric : str
            Name of metric to generate forecast for.
        method : str
            Either "arima" or "prophet".
        seed : int
            Random seed

        Returns
        -------
        forecast : pd.DataFrame
            Forecast results.

        """
        if self.cores == 1:
            print(f"Running {method} forecast for: {metric} - {county}...")

        # Filter historic data to specified county and metric, then just keep
        # "ds" and "y" col
        historic = self.df_historic.loc[
            (self.df_historic["ora"] == county)
            & (self.df_historic["currency"] == metric),
            ["ds", "y"],
        ]

        # Filter holidays to specified county
        holidays = self.df_holidays[self.df_holidays["county"] == county]

        # Generate forecasts
        if method == "prophet":
            if seed:
                np.random.seed(seed)
            forecast = predict_prophet(
                historic=historic,
                holidays=holidays,
                forecast_length=self.forecast_length,
                metric=metric,
            )
        if method == "arima":
            forecast = predict_arima(
                historic=historic,
                holidays=holidays,
                forecast_length=self.forecast_length,
            )

        # Add columns with county and metric name, then save result to list
        forecast.insert(1, "county", county)
        forecast.insert(2, "currency", metric)
        return forecast

    def run(self, method):
        """Generate forecasts for all metrics using specified method.

        Parameters
        ----------
        method : str
            Either "prophet" or "arima".

        """
        pairs = list(self.unique_pairs.itertuples(index=False, name=None))
        seeds = [i for i in range(len(pairs))]

        # Run sequentially
        if self.cores == 1:
            forecast_list = [
                self.generate_forecast(
                    county=county, metric=metric, method=method, seed=seed
                )
                for (county, metric), seed in zip(pairs, seeds, strict=True)
            ]
        # Run in parallel
        else:
            # Check number of cores is valid
            valid_cores = [-1] + list(range(1, cpu_count()))
            if self.cores not in valid_cores:
                raise ValueError(
                    f"Invalid cores: {self.cores}. Must be one of: "
                    + f"{valid_cores}."
                )
            forecast_list = Parallel(n_jobs=self.cores)(
                delayed(self.generate_forecast)(county, metric, method, seed)
                for (county, metric), seed in zip(pairs, seeds, strict=True)
            )
        self.results_list.append(pd.concat(forecast_list))

    @property
    def results(self):
        return pd.concat(self.results_list)
