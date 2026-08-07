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


def cross_validation(df_historic, df_holidays, metrics, scenario, horizon, step, cores):
    """Perform cross-validation for the chosen model.

    Parameters
    ----------
    df_historic : pd.DataFrame
        Historic data.
    df_holidays : pd.DataFrame
        Holidays data.
    metrics : list
        Metrics.
    scenario : dict
        Single scenario dictionary in suitable format for Forecaster.run().
    horizon : int
        Number of days into future that the data is predicted.
    step : int
        How many days to move by before creating a new sample. Warning:
        using a step of 365 will produce test samples all at approximately the
        same time of year.
    cores : int
        Cores.

    Returns
    -------
    pd.DataFrame
        Dataframe with errors for each county, metric and cross-validation
        fold.

    """
    # Create samples for cross-validation with rolling forecast origin
    train, test = create_rolling_samples(df_historic, step=step)

    cv_forecasts = []
    for i in range(len(train)):
        # Fit model using training sample
        forecaster = Forecaster(
            df_historic=train[i],
            df_holidays=df_holidays,
            metrics=metrics,
            horizon=horizon,
            cores=cores,
        )
        # Set name for cross-validation scenario and run forecaster
        scenario["name"] = f"cross_validation_{i}"
        forecaster.run([scenario], base_seed=i * 100000)
        # Save results dataframe to cv_forecasts
        cv_forecasts.append(forecaster.results)

    errors_list = []

    # Loop through all the counties and metrics
    pairs = list(forecaster.unique_pairs.itertuples(index=False, name=None))
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
                (cv_fold["county"] == county) & (cv_fold["currency"] == metric)
            ]

            errors_list.append(
                {
                    "county": county,
                    "metric": metric,
                    "fold": fold,
                    "rmse": root_mean_squared_error(
                        y_true=test_subset["y"], y_pred=cv_subset["forecast"]
                    ),
                    "mase": mean_absolute_scaled_error(
                        y_true=test_subset["y"],
                        y_pred=cv_subset["forecast"],
                        y_train=train_subset["y"],
                    ),
                    "smape": symmetric_mean_absolute_percentage_error(
                        y_true=test_subset["y"], y_pred=cv_subset["forecast"]
                    ),
                    "coverage": coverage(
                        y_true=test_subset["y"],
                        pred_intervals=cv_subset[
                            ["pi_lower", "pi_upper"]
                        ].values.tolist(),
                    ),
                }
            )

    return pd.DataFrame(errors_list)
