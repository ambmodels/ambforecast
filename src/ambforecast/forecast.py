"""Functions used to run forecasts."""

import hashlib
from time import perf_counter

import pandas as pd
from forecast_tools.metrics import (
    coverage,
    mean_absolute_scaled_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)
from joblib import Parallel, delayed, effective_n_jobs
from tqdm.auto import tqdm

from .prophet import prophet
from .splits import rolling_forecast_origin


def unique_pairs(data):
    """Every (metric, area) combination present in the data.

    Parameters
    ----------
    data : pd.DataFrame
        Historic data containing `metric` and `area` columns.

    Returns
    -------
    list[tuple]
        List of tuples with metric and area e.g., [('Responses', 'Cornwall')].

    """
    return list(
        data[["metric", "area"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .itertuples(index=False, name=None)
    )


def make_seed(*parts):
    """Create a reproducible integer seed from one or more identifiers.

    Parameters
    ----------
    *parts
        Values that uniquely identify a model fit, such as `metric`, `area`,
        and, for cross-validation, `fold`.

    Returns
    -------
    int
        A deterministic integer seed suitable for Prophet.

    """
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def run_single_forecast(
    forecast_function,
    train,
    params,
    metric,
    area,
    test=None,
    horizon=None,
    seed_parts=None,
):
    """Run a forecast for one metric and area.

    Filters the full training and optional test datasets to one metric/area
    series and runs the selected forecasting function.

    Parameters
    ----------
    forecast_function : prophet | arima | snaive
        Forecasting function to run.
    train : pd.DataFrame
        Historic data used to train the model.
    params : ProphetParams | ARIMAParams | SNaiveParams | dict
        Parameters for the selected forecasting model. Can be one parameter
        object used for all metrics, or a dictionary mapping metric names to
        parameter objects.
    metric : str
        Name of metric to forecast.
    area : str
        Name of area to forecast.
    test : pd.DataFrame
        Held-out data to forecast and later compare against. If None, provide
        `horizon` instead,
    horizon : int
        Number of days to forecast after the final training date. If None,
        provide `test` instead.
    seed_parts : tuple | None
        Values used to create a deterministic random seed for Prophet. Ignored
        for forecasting functions other than Prophet.

    Returns
    -------
    forecast : pd.DataFrame
        Forecast results.

    """
    if isinstance(params, dict):
        if metric not in params:
            raise ValueError(f"No parameters provided for metric: {metric!r}")
        params = params[metric]

    train_subset = train[(train["metric"] == metric) & (train["area"] == area)]

    if test is not None:
        test_subset = test[(test["metric"] == metric) & (test["area"] == area)]
    else:
        test_subset = None

    forecast_kwargs = {
        "train": train_subset,
        "params": params,
        "test": test_subset,
        "horizon": horizon,
    }
    # Uses metric and area if seed_parts = None
    if forecast_function is prophet:
        seed_parts = seed_parts or (metric, area)
        forecast_kwargs["seed"] = make_seed(*seed_parts)

    forecast = forecast_function(**forecast_kwargs)

    forecast.insert(1, "metric", metric)
    forecast.insert(2, "area", area)
    return forecast


def run_forecasts(
    forecast_function,
    train,
    params,
    test=None,
    horizon=None,
    cores=1,
):
    """Run forecasts for every metric and area, sequentially or in parallel.

    Parameters
    ----------
    forecast_function : prophet | arima
        Forecasting function to run.
    train : pd.DataFrame
        Historic data used to train the model.
    params : ProphetParams | ARIMAParams
        Parameters for the selected forecasting model.
    test : pd.DataFrame
        Held-out data to forecast and later compare against. If None, provide
        `horizon` instead,
    horizon : int
        Number of days to forecast after the final training date. If None,
        provide `test` instead.
    cores : int
        Number of CPU cores to use. Set to 1 for sequential processing or -1
        to use all available cores.

    Returns
    -------
    pd.DataFrame
        Forecasts for every metric and area.

    """
    pairs = unique_pairs(train)

    if cores == 1:
        forecasts = [
            run_single_forecast(
                forecast_function=forecast_function,
                train=train,
                params=params,
                metric=metric,
                area=area,
                test=test,
                horizon=horizon,
                seed_parts=(metric, area),  # Only used by Prophet
            )
            for (metric, area) in pairs
        ]
    else:
        forecasts = Parallel(n_jobs=cores)(
            delayed(run_single_forecast)(
                forecast_function=forecast_function,
                train=train,
                params=params,
                metric=metric,
                area=area,
                test=test,
                horizon=horizon,
                seed_parts=(metric, area),  # Only used by Prophet
            )
            for (metric, area) in pairs
        )
    return pd.concat(forecasts, ignore_index=True)


def run_cross_validation(
    forecast_function,
    historic,
    params,
    horizon,
    step,
    min_train=365 * 2,
    cores=1,
):
    """Run rolling forecast origin cross-validation.

    This can take some time because it fits a separate forecasting model for
    every combination of fold, metric, and area. Progress is displayed with
    `tqdm`, including the number of completed forecasts. The function also
    prints the number of worker processes used and the total run time.

    Parameters
    ----------
    forecast_function : prophet | arima
        Forecasting function to run.
    historic : pd.DataFrame
        Historic data used to create rolling training and test samples.
    params : ProphetParams | ARIMAParams
        Parameters for the selected forecasting model.
    horizon : int
        Number of daily observations in each test set.
    step : int
        How many days to move by before creating a new sample. Warning:
        using a step of 365 will produce test samples all at approximately
        the same time of year.
    min_train : int
        Minimum number of days to include in training sample. By default,
        set to 2 years as that allows detection of yearly seasonality.
    cores : int
        Number of CPU cores to use. Set to 1 for sequential processing or -1
        to use all available cores.

    Returns
    -------
    forecasts, errors : tuple[pd.DataFrame, pd.DataFrame]
        Forecasts contains results for every fold, metric, and area.
        Errors contains accuracy measures for every fold, metric, and area.

    """
    start_time = perf_counter()

    # Create several sets of training and test data
    train_folds, test_folds = rolling_forecast_origin(
        data=historic, horizon=horizon, step=step, min_train=min_train
    )

    # Find unique combinations of metric and area
    pairs = unique_pairs(historic)

    # Find all combinations of metric, area and fold to loop through
    jobs = [
        (fold, metric, area)
        for fold in range(len(train_folds))
        for metric, area in pairs
    ]

    workers = effective_n_jobs(cores)
    print(
        f"Running {len(pairs)} series across {len(train_folds)} folds "
        f"({len(jobs)} forecasts) using {workers} worker(s)."
    )

    # Fit models and generate forecasts, sequentially or in parallel
    if cores == 1:
        forecasts = [
            run_single_forecast(
                forecast_function=forecast_function,
                train=train_folds[fold],
                test=test_folds[fold],
                params=params,
                metric=metric,
                area=area,
                seed_parts=(fold, metric, area),  # Only used by Prophet
            )
            for fold, metric, area in tqdm(
                jobs,
                desc="Running forecasts",
                unit="forecast",
            )
        ]
    else:
        forecast_generator = Parallel(
            n_jobs=cores,
            return_as="generator",
        )(
            delayed(run_single_forecast)(
                forecast_function=forecast_function,
                train=train_folds[fold],
                test=test_folds[fold],
                params=params,
                metric=metric,
                area=area,
                seed_parts=(fold, metric, area),  # Only used by Prophet
            )
            for fold, metric, area in jobs
        )
        forecasts = list(
            tqdm(
                forecast_generator,
                total=len(jobs),
                desc="Running forecasts",
                unit="forecast",
            )
        )

    # Calculate forecast errors
    errors = []

    for forecast, (fold, metric, area) in zip(forecasts, jobs, strict=True):
        forecast.insert(0, "fold", fold)

        error = forecast_errors(
            train=train_folds[fold],
            test=test_folds[fold],
            forecast=forecast,
        )

        errors.append(
            {
                "fold": fold,
                "forecast_start_date": forecast["ds"].min(),
                "metric": metric,
                "area": area,
                **error,
            }
        )

    forecasts = pd.concat(forecasts, ignore_index=True)
    errors = pd.DataFrame(errors)

    elapsed_seconds = round(perf_counter() - start_time)
    minutes, seconds = divmod(elapsed_seconds, 60)
    print(
        "Cross-validation completed in "
        f"{minutes} minute(s) and {seconds} second(s)."
    )

    return forecasts, errors


def forecast_errors(train, test, forecast):
    """Calculate forecast accuracy measures.

    Measures included:
    - RMSE (root mean squared error)
    - MASE (mean absolute scaled error)
    - sMAPE (symmetric mean absolute percentage error)
    - Coverage (prediction interval coverage)

    Parameters
    ----------
    train : pd.DataFrame
        Historic data used to train the model.
    test : pd.DataFrame
        Test data.
    forecast : pd.DataFrame
        Forecast results.

    Returns
    -------
    dict
        Dictionary of forecast accuracy measures.

    """
    # Filter training and test data to the metric and area in the forecast
    metric = forecast["metric"].iloc[0]
    area = forecast["area"].iloc[0]
    train_subset = train[(train["metric"] == metric) & (train["area"] == area)]
    test_subset = test[(test["metric"] == metric) & (test["area"] == area)]

    # Check that forecast and training data have same dates
    test_subset = test_subset.sort_values("ds").reset_index(drop=True)
    forecast = forecast.sort_values("ds").reset_index(drop=True)
    if not test_subset["ds"].equals(forecast["ds"]):
        raise ValueError("Test and forecast do not contain the same dates.")

    # Calculate forecast accuracy measures
    return {
        "rmse": root_mean_squared_error(
            y_true=test_subset["y"], y_pred=forecast["forecast"]
        ),
        "mase": mean_absolute_scaled_error(
            y_true=test_subset["y"],
            y_pred=forecast["forecast"],
            y_train=train_subset["y"],
        ),
        "smape": symmetric_mean_absolute_percentage_error(
            y_true=test_subset["y"], y_pred=forecast["forecast"]
        ),
        "coverage": coverage(
            y_true=test_subset["y"],
            pred_intervals=forecast[["pi_lower", "pi_upper"]].values.tolist(),
        ),
    }
