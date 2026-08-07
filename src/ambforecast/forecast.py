"""Run forecast."""

from collections import Counter
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

        # Empty dict to store forecast results
        self.results_dict = {}

        # Print the forecast start date
        forecast_start_date = df_historic["ds"].max() + pd.Timedelta(days=1)
        print(
            "Forecast start date: "
            + f"{forecast_start_date.strftime('%A %d %B %Y')}"
        )

    @staticmethod
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
        params = self.resolve_params(params, metric)

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
                forecast_length=self.forecast_length,
                **params,
            )
        elif method == "arima":
            forecast = predict_arima(
                historic=historic,
                holidays=holidays,
                forecast_length=self.forecast_length,
                **params,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Add columns with name, county and metric, then save result to list
        forecast.insert(0, "method", name)
        forecast.insert(1, "county", county)
        forecast.insert(2, "currency", metric)
        return forecast

    def run(self, scenarios=None, *, name=None, method=None, params=None):
        """Generate forecasts for all metrics and counties.

        Can run as single scenario:
        >> forecaster.run(name="arima_current_version",
                            method="arima",
                            params={"order": (1, 1, 1), ...})

        Or can run a list of multiple scenarios:
        >> forecaster.run(scenarios=[{...}, {...}])

        Parameters
        ----------
        scenarios : list[dict]
            List of dictionaries, where each dictionary has the keys "name",
            "method" and "params".
        name : str
            Name for the scenario.
        method : str
            Method to use - either "arima" or "prophet".
        params : dict
            Parameters for that method.

        """
        # If scenarios is provided, shouldn't also provide name/method/params
        if scenarios is not None:
            if name or method or params:
                raise ValueError(
                    "Pass either 'scenarios' OR 'name/method/params', ",
                    "not both."
                )
            scenario_list = scenarios
        # If scenarios is not provided, make sure have name + method
        else:
            if name is None or method is None:
                raise ValueError(
                    "When 'scenarios' is not provided, 'name' and 'method' ",
                    "are required."
                )
            scenario_list = [
                {
                    "name": name,
                    "method": method,
                    "params": params,
                }
            ]
        # Return an error if any names are duplicate or already in results
        names = [scenario["name"] for scenario in scenario_list]
        duplicates = [i for i, count in Counter(names).items() if count > 1]
        existing = set(names).intersection(self.results_dict.keys())
        if duplicates or existing:
            raise ValueError(
                "The provided scenario names are either duplicates or "
                "already exist in the results_dict. "
                f"Duplicates: {duplicates}. Pre-existing {existing}."
            )

        pairs = list(self.unique_pairs.itertuples(index=False, name=None))
        seeds = [i for i in range(len(pairs))]

        # Loop over scenarios
        for scenario in scenario_list:
            print(f"Running scenario: {scenario}")

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
            # Store results for this scenario by name
            self.results_dict[scenario["name"]] = pd.concat(forecast_list)

    @property
    def results(self):
        """Return results_list as a single dataframe.

        Returns
        -------
        pd.DataFrame
            Single dataframe with all forecast results.

        """
        return pd.concat(list(self.results_dict.values()), ignore_index=True)
