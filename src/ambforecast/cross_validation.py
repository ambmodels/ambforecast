"""Cross-validation with rolling forecast origin."""

import datetime as dt

import pandas as pd
from forecast_tools.metrics import (
    coverage,
    mean_absolute_scaled_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)

from .forecast import Forecaster


class CrossValidator:
    """Run cross-validation with rolling forecast origin.

    This will create overlapping sets of training and test data. Each sample
    trains on data available up to a forecast origin and generates forecasts
    for the horizon after that.
    """

    def __init__(
        self,
        df_historic,
        metrics,
        df_holidays=None,
        horizon=42,
        step=42,
        min_train=365 * 2,
        cores=-1,
    ):
        """Initialise with data and settings for cross-validation.

        Parameters
        ----------
        df_historic : pd.DataFrame
            Historic data.
        metrics : list[str]
            Metrics to forecast.
        df_holidays : pd.DataFrame
            Holidays data.
        horizon : int
            Number of days into future that the data is predicted.
        step : int
            How many days to move by before creating a new sample. Warning:
            using a step of 365 will produce test samples all at approximately
            the same time of year.
        min_train : int
            Minimum number of days to include in training sample. By default,
            set to 2 years as that allows detection of yearly seasonality
            (e.g., Prophet minimum is 2 years).
        cores : int
            Number of CPU cores to use for parallel execution. For all
            available cores, set to -1. For sequential execution, set to 1.

        """
        self.df_historic = df_historic
        self.df_holidays = df_holidays
        self.metrics = metrics
        self.horizon = horizon
        self.step = step
        self.min_train = min_train
        self.cores = cores

    def create_rolling_samples(self):
        """Create samples for cross-validation with rolling forecast origin.

        Samples are generated from the most recent data backwards. `step`
        specifies how many days earlier the forecast origin moves between
        samples. For example, `step=7` creates samples with forecast origins
        7 days apart.

        Returns
        -------
        train, val : tuple[list[pd.DataFrame],list[pd.DataFrame]]
            Two lists containing training and validation datasets.

        """
        if self.df_historic["ds"].nunique() < self.min_train + self.horizon:
            raise ValueError(
                "Insufficient data for requested cross-validation. The "
                f"provided historic data covers "
                f"{len(self.df_historic['ds'].unique())}, but the minimum "
                f"training size is {self.min_train} and horizon is "
                f"{self.horizon}."
            )

        train = []
        test = []
        sample = self.df_historic.copy()

        while sample["ds"].nunique() >= self.min_train + self.horizon:
            # Find start date of testing sample
            test_start = sample["ds"].max() - dt.timedelta(
                days=self.horizon - 1
            )

            # Split into train and test dataset
            train.append(sample[sample["ds"] < test_start])
            test.append(sample[sample["ds"] >= test_start])

            # Remove days from the end of sample (number removed = step)
            sample_end = sample["ds"].max() - dt.timedelta(days=self.step - 1)
            sample = sample[sample["ds"] < sample_end].copy()

        return train, test

    def run(self, scenario):
        """Perform cross-validation for the chosen model.

        Parameters
        ----------
        scenario : dict
            Single scenario dictionary in suitable format for Forecaster.run().

        Returns
        -------
        forecasts, errors : tuple[pd.DataFrame, pd.DataFrame]
            First dataframe contains the forecast results, and second
            contains the errors for each county, metric and cross-validation
            fold.

        """
        name = scenario["name"]

        # Create samples for cross-validation with rolling forecast origin
        train, test = self.create_rolling_samples()

        cv_forecasts = []
        for i in range(len(train)):
            # Fit model using training sample
            forecaster = Forecaster(
                df_historic=train[i],
                df_holidays=self.df_holidays,
                metrics=self.metrics,
                horizon=self.horizon,
                cores=self.cores,
            )
            # Set name for cross-validation scenario and run forecaster
            scenario["name"] = f"cross_validation_{i}"
            forecaster.run([scenario], base_seed=i * 100000)
            # Save results dataframe to cv_forecasts
            cv_forecasts.append(forecaster.results)

        errors_list = []

        # Loop through all the counties and metrics
        pairs = list(
            forecaster.unique_pairs.itertuples(index=False, name=None)
        )
        for county, metric in pairs:
            # Loop through all the cross-validation samples
            for fold, (train_fold, test_fold, cv_fold) in enumerate(
                zip(train, test, cv_forecasts, strict=True)
            ):
                # Filter the datasets to the relevant county and metric
                train_subset = train_fold[
                    (train_fold["ora"] == county)
                    & (train_fold["currency"] == metric)
                ]
                test_subset = test_fold[
                    (test_fold["ora"] == county)
                    & (test_fold["currency"] == metric)
                ]
                cv_subset = cv_fold[
                    (cv_fold["county"] == county)
                    & (cv_fold["currency"] == metric)
                ]

                errors_list.append(
                    {
                        "county": county,
                        "metric": metric,
                        "fold": fold,
                        "rmse": root_mean_squared_error(
                            y_true=test_subset["y"],
                            y_pred=cv_subset["forecast"],
                        ),
                        "mase": mean_absolute_scaled_error(
                            y_true=test_subset["y"],
                            y_pred=cv_subset["forecast"],
                            y_train=train_subset["y"],
                        ),
                        "smape": symmetric_mean_absolute_percentage_error(
                            y_true=test_subset["y"],
                            y_pred=cv_subset["forecast"],
                        ),
                        "coverage": coverage(
                            y_true=test_subset["y"],
                            pred_intervals=cv_subset[
                                ["pi_lower", "pi_upper"]
                            ].values.tolist(),
                        ),
                    }
                )

        forecasts = pd.concat(cv_forecasts, ignore_index=True)
        errors = pd.DataFrame(errors_list)
        errors.insert(0, "name", name)
        return forecasts, errors
