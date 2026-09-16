"""Regression tests - results are consistent over time."""

from pathlib import Path

import pandas as pd
import pytest

from ambforecast.arima import ARIMAParams, arima
from ambforecast.naive import SNaiveParams, snaive
from ambforecast.prophet import ProphetParams, prophet

INPUT = Path(__file__).parent.joinpath("data")
AREA = "Trust"
HORIZON = 42
SEED = 123


@pytest.fixture
def historic():
    """Import historic data."""
    historic = pd.read_csv(
        INPUT / "responses_2013_to_2019.csv", parse_dates=["ds"]
    )
    return historic[historic["area"] == AREA]


@pytest.fixture
def holidays():
    """Import holiday data."""
    holidays = pd.read_csv(INPUT / "holidays.csv", parse_dates=["ds"])
    return holidays[holidays["area"] == AREA]


def test_arima_consistent(historic, holidays):
    """Check simple ARIMA forecast is consistent with previous run."""
    params = ARIMAParams(holidays=holidays)
    forecast = arima(train=historic, params=params, horizon=HORIZON)
    exp_forecast = pd.read_csv(INPUT / "exp_arima.csv", parse_dates=["ds"])
    pd.testing.assert_frame_equal(forecast, exp_forecast, check_names=False)


def test_prophet_consistent(historic, holidays):
    """Check simple Prophet forecast is consistent with previous run."""
    params = ProphetParams(holidays=holidays)
    forecast = prophet(
        train=historic, params=params, horizon=HORIZON, seed=SEED
    )
    exp_forecast = pd.read_csv(INPUT / "exp_prophet.csv", parse_dates=["ds"])
    pd.testing.assert_frame_equal(forecast, exp_forecast, check_names=False)


def test_snaive_consistent(historic):
    """Check simple seasonal naive forecast is consistent with previous run."""
    params = SNaiveParams(period=7)
    forecast = snaive(train=historic, params=params, horizon=HORIZON)
    exp_forecast = pd.read_csv(INPUT / "exp_snaive.csv", parse_dates=["ds"])
    pd.testing.assert_frame_equal(forecast, exp_forecast, check_names=False)
