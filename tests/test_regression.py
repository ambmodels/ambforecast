"""Regression tests - results are consistent over time."""

from pathlib import Path

import numpy as np
import pandas as pd

from ambforecast.arima import predict_arima
from ambforecast.prophet import predict_prophet
from ambforecast.utility import default_ensemble, pre_process_daily_data

FILES = Path(__file__).parent.joinpath("regression_results")

DF_HISTORIC = FILES / "fake_historic.csv"
DF_HOLIDAYS = FILES / "fake_holidays.csv"
FORECAST = FILES / "forecast.csv"


def test_forecast_consistent():
    """Check that forecast results are consistent with previous run."""
    # Import the fake data
    df_historic = pd.read_csv(DF_HISTORIC)
    df_holidays = pd.read_csv(DF_HOLIDAYS)

    # Process data
    df_in = df_historic.drop(["currency", "forecast_type"], axis=1)
    clean = pre_process_daily_data(df_in, observation_col="y", index_col="ds")

    # Run ensemble forecast
    np.random.seed(42)
    model = default_ensemble(county="FakeCounty", df_holiday=df_holidays)
    model.fit(clean)
    forecast = model.predict(horizon=30, return_all_models=True)

    # Import expected forecast results and compare
    exp_forecast = pd.read_csv(FORECAST, index_col="ds")
    exp_forecast.index = pd.to_datetime(exp_forecast.index)
    exp_forecast.index.freq = "D"
    pd.testing.assert_frame_equal(forecast, exp_forecast)


def test_new_forecast_consistent():
    """Check that new forecast results are consistent with previous run."""
    # Import the fake data
    df_historic = pd.read_csv(DF_HISTORIC, parse_dates=["ds"])
    df_holidays = pd.read_csv(DF_HOLIDAYS, parse_dates=["ds"])

    # Process data
    historic = df_historic[["ds", "y"]]
    holidays = df_holidays.copy()

    # Run prophet prediction
    np.random.seed(42)
    prophet_forecast = predict_prophet(
        historic=historic, holidays=holidays, forecast_length=30, metric="Responses",
    )

    # Run ARIMA prediction
    np.random.seed(42)
    arima_forecast = predict_arima(
        historic=historic, holidays=holidays, forecast_length=30
    )

    # Import expected forecast results
    exp_forecast = pd.read_csv(FORECAST, index_col="ds")

    def check_columns(actual_df, pairs):
        for actual_col, expected_col in pairs:
            try:
                pd.testing.assert_series_equal(
                    actual_df[actual_col].reset_index(drop=True),
                    exp_forecast[expected_col].reset_index(drop=True),
                    check_dtype=False,
                    check_names=False,
                )
            except AssertionError as e:
                raise AssertionError(
                    f"Mismatch for actual column '{actual_col}' vs "
                    f"expected column '{expected_col}':\n{e}"
                ) from e

    check_columns(
        arima_forecast,
        [
            ("arima_mean", "arima_mean"),
            ("arima_lower", "arima_lower_95"),
            ("arima_upper", "arima_upper_95"),
        ],
    )

    check_columns(
        prophet_forecast,
        [
            ("yhat", "prophet_mean"),
            ("yhat_lower", "prophet_lower_95"),
            ("yhat_upper", "prophet_upper_95"),
        ],
    )