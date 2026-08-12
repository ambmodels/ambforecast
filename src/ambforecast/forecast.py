"""Functions used to run forecasts."""

import pandas as pd
from joblib import Parallel, delayed
from forecast_tools.metrics import (
    coverage,
    mean_absolute_scaled_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)


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
        data[["metric", "area"]].drop_duplicates()
        .reset_index(drop=True)
        .itertuples(index=False, name=None)
    )


def run_single_forecast(
    forecast_function,
    train,
    params,
    metric,
    area,
    test=None,
    horizon=None,
):
    """Run a forecast for one metric and area.

    Filters the full training and optional test datasets to one metric/area
    series and runs the selected forecasting function.

    Parameters
    ----------
    forecast_function : prophet | arima
        Forecasting function to run.
    train : pd.DataFrame
        Historic data used to train the model.
    params : ProphetParams | ARIMAParams
        Parameters for the selected forecasting model.
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

    Returns
    -------
    forecast : pd.DataFrame
        Forecast results.

    """
    train_subset = train[(train["metric"] == metric) & (train["area"] == area)]

    if test is not None:
        test_subset = test[(test["metric"] == metric) & (test["area"] == area)]
    else:
        test_subset = None

    forecast = forecast_function(
        train=train_subset,
        params=params,
        test=test_subset,
        horizon=horizon
    )

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
                horizon=horizon
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
                horizon=horizon
            )
            for (metric, area) in pairs
        )
    return pd.concat(forecasts, ignore_index=True)


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
    train_subset = train[
        (train["metric"] == metric)
        & (train["area"] == area)
    ]
    test_subset = test[
        (test["metric"] == metric)
        & (test["area"] == area)
    ]

    # Check that forecast and training data have same dates
    test_subset = test_subset.sort_values("ds").reset_index(drop=True)
    forecast = forecast.sort_values("ds").reset_index(drop=True)
    if not test_subset["ds"].equals(forecast["ds"]):
        raise ValueError("Test and forecast do not contain the same dates.")

    # Calculate forecast accuracy measures
    return {
        "rmse": root_mean_squared_error(
            y_true=test_subset["y"],
            y_pred=forecast["forecast"]
        ),
        "mase": mean_absolute_scaled_error(
            y_true=test_subset["y"],
            y_pred=forecast["forecast"],
            y_train=train_subset["y"]
        ),
        "smape": symmetric_mean_absolute_percentage_error(
            y_true=test_subset["y"],
            y_pred=forecast["forecast"]
        ),
        "coverage": coverage(
            y_true=test_subset["y"],
            pred_intervals=forecast[["pi_lower", "pi_upper"]].values.tolist()
        )
    }
