"""Run forecast."""

import datetime as dt
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from .arima import predict_arima
from .ensemble import calculate_ensemble
from .prophet import predict_prophet


class Forecaster:
    """Run forecasts.

    Attributes
    ----------
    df_holidays : pd.DataFrame
        Holidays.
    metrics : list[str]
        Metrics to forecast.
    horizon : int
        Number of days into future that the data is predicted.
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

    def __init__(self, df_historic, df_holidays, metrics, horizon, cores=1):
        """Initialise with data and settings to use across all forecasts.

        Parameters
        ----------
        df_historic : pd.DataFrame
            Historic data. Should have columns "ds" (date), "currency"
            (metric), "ora" (county/trust) and "y" (value).
        df_holidays : pd.DataFrame
            Holidays. Should have columns "ds" (date), "holiday" (name of
            holiday), "lower_window", "upper_window" and "county".
        metrics : list[str]
            Metric names (values of "currency") to forecast.
        horizon : int
            Number of days into future that the data is predicted.
        cores : int
            Number of CPU cores to use for parallel execution. For all
            available cores, set to -1. For sequential execution, set to 1.

        """
        self.df_holidays = df_holidays
        self.metrics = metrics
        self.horizon = horizon
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

        # Empty dict to store forecast results
        self.results_dict = {}

        # Print the forecast start date
        forecast_start_date = df_historic["ds"].max() + pd.Timedelta(days=1)
        print(
            "Forecast start date: "
            + f"{forecast_start_date.strftime('%A %d %B %Y')}"
        )

    def generate_forecast(
        self, county, metric, method, name, params=None, seed=None
    ):
        """Generate single forecast.

        Parameters
        ----------
        county : str
            Name of county/area to generate forecast for.
        metric : str
            Name of metric to generate forecast for.
        method : str
            Either "arima" or "prophet".
        name : str
            Name for the forecast.
        params : dict
            Extra keyword arguments for the forecasting method.
        seed : int
            Random seed.

        Returns
        -------
        forecast : pd.DataFrame
            Forecast results.

        """
        # Set-up params object - including fetching relevant params for that
        # metric, if provided as dict with different params for each metric
        params = params or {}
        params = resolve_params(params, metric)

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
            if seed is not None:
                np.random.seed(seed)
            forecast = predict_prophet(
                historic=historic,
                holidays=holidays,
                horizon=self.horizon,
                **params,
            )
        elif method == "arima":
            forecast = predict_arima(
                historic=historic,
                holidays=holidays,
                horizon=self.horizon,
                **params,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Add columns with name, county and metric, then save result to list
        forecast.insert(0, "name", name)
        forecast.insert(1, "county", county)
        forecast.insert(2, "currency", metric)
        return forecast

    def run(self, scenarios, base_seed=0):
        """Generate forecasts for all metrics and counties.

        Parameters
        ----------
        scenarios : list[dict]
            List of dictionaries, where each dictionary has the keys "name"
            (label), "method" (one of the supported methods), and "params"
            (dictionary of named parameters for that method). For example,
            {"name": "arima_baseline", "method": "arima",
            "params": {"order": (1, 1, 1), ...} }.
        base_seed : int
            Base seed - when creating seeds, they are add to the base.

        """
        # Return an error if any names are duplicate or already in results
        names = [scenario["name"] for scenario in scenarios]
        duplicates = [i for i, count in Counter(names).items() if count > 1]
        existing = set(names).intersection(self.results_dict.keys())
        if duplicates or existing:
            raise ValueError(
                "The provided scenario names are either duplicates or "
                "already exist in the results_dict. "
                f"Duplicates: {duplicates}. Pre-existing {existing}."
            )

        pairs = list(self.unique_pairs.itertuples(index=False, name=None))
        seeds = [base_seed + i for i in range(len(pairs))]

        # Loop over scenarios
        for scenario in scenarios:
            print(f"Running scenario: {scenario}")

            # Ensemble if just a simple groupby operation so don't need to
            # use generate_forecast, which runs all paris in a loop
            if scenario["method"] == "ensemble":
                full_forecast = calculate_ensemble(
                    forecast_results=self.results,
                    **scenario.get("params", {}),
                )
                full_forecast.insert(0, "name", scenario["name"])

            else:
                # Run sequentially
                if self.cores == 1:
                    forecast_list = [
                        self.generate_forecast(
                            county=county,
                            metric=metric,
                            method=scenario["method"],
                            name=scenario["name"],
                            params=scenario.get("params", {}),
                            seed=seed,
                        )
                        for (county, metric), seed in zip(
                            pairs, seeds, strict=True
                        )
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
                        delayed(self.generate_forecast)(
                            county=county,
                            metric=metric,
                            method=scenario["method"],
                            name=scenario["name"],
                            params=scenario.get("params", {}),
                            seed=seed,
                        )
                        for (county, metric), seed in zip(
                            pairs, seeds, strict=True
                        )
                    )
                full_forecast = pd.concat(forecast_list)

            # Store results for this scenario by name
            self.results_dict[scenario["name"]] = full_forecast

    @property
    def results(self):
        """Return results_list as a single dataframe.

        Returns
        -------
        pd.DataFrame
            Single dataframe with all forecast results.

        """
        return pd.concat(list(self.results_dict.values()), ignore_index=True)


def resolve_params(params, metric):
    """Resolve params for a specific metric.

    Any parameter value can be given as a plain value (used for all
    metrics) or as a dict keyed by metric name.

    Parameters
    ----------
    params : dict
        Parameters, where each value is either a plain value or a
        dict mapping metric name to value.
    metric : str
        Metric to resolve params for.

    Returns
    -------
    dict
        Params with any per-metric dicts resolved to a single value.

    """
    resolved = {}
    for key, value in params.items():
        if isinstance(value, dict) and metric in value:
            resolved[key] = value[metric]
        else:
            resolved[key] = value
    return resolved


def create_rolling_samples(
    df_historic, min_train=365 * 2, horizon=42, step=42
):
    """Create samples for cross-validation with rolling forecast origin.

    Cross-validation with rolling forecast origin creates overlapping sets of
    training and test data. Each sample trains on data available up to a
    forecast origin and validates forecasts for the following `horizon` days.

    Samples are generated from the most recent data backwards. `step` specifies
    how many days earlier the forecast origin moves between samples. For
    example, `step=7` creates samples with forecast origins 7 days apart.

    Parameters
    ----------
    df_historic : pd.DataFrame
        Historic data.
    min_train : int
        Minimum number of days to include in training sample. By default, set
        to 2 years as that allows detection of yearly seasonality (e.g.,
        Prophet minimum is 2 years).
    horizon : int
        Number of days into future that the data is predicted.
    step : int
        How many days to move by before creating a new sample. Warning:
        using a step of 365 will produce test samples all at approximately the
        same time of year.

    Returns
    -------
    train, val : tuple[list[pd.DataFrame],list[pd.DataFrame]]
        Two lists containing training and validation datasets.

    """
    if df_historic["ds"].nunique() < min_train + horizon:
        raise ValueError(
            "Insufficient data for requested cross-validation. The provided "
            f"historic data covers {len(df_historic['ds'].unique())}, but "
            f"the minimum training size is {min_train} and horizon is "
            f"{horizon}."
        )

    train = []
    test = []
    sample = df_historic.copy()

    while sample["ds"].nunique() >= min_train + horizon:

        # Find start date of testing sample
        test_start = sample["ds"].max() - dt.timedelta(days=horizon - 1)

        # Split into train and test dataset
        train.append(sample[sample["ds"] < test_start])
        test.append(sample[sample["ds"] >= test_start])

        # Remove days from the end of sample (number removed = step)
        sample_end = sample["ds"].max() - dt.timedelta(days=step - 1)
        sample = sample[sample["ds"] < sample_end].copy()

    return train, test
