"""Regression tests - results are consistent over time."""

from pathlib import Path

import numpy as np
import pandas as pd

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
