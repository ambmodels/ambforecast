"""Generate results for regression tests."""

from pathlib import Path

import pandas as pd

from ambforecast.arima import ARIMAParams, arima
from ambforecast.naive import SNaiveParams, snaive
from ambforecast.prophet import ProphetParams, prophet

DATA = Path(__file__).parent
AREA = "Trust"
HORIZON = 42
SEED = 123

historic = pd.read_csv(DATA / "responses_2013_to_2019.csv", parse_dates=["ds"])
historic = historic[historic["area"] == AREA]

holidays = pd.read_csv(DATA / "holidays.csv", parse_dates=["ds"])
holidays = holidays[holidays["area"] == AREA]

params = ARIMAParams(holidays=holidays)
forecast = arima(train=historic, params=params, horizon=HORIZON)
forecast.to_csv(DATA / "exp_arima.csv", index=False)

params = ProphetParams(holidays=holidays)
forecast = prophet(train=historic, params=params, horizon=HORIZON, seed=SEED)
forecast.to_csv(DATA / "exp_prophet.csv", index=False)

params = SNaiveParams(period=7)
forecast = snaive(train=historic, params=params, horizon=HORIZON)
forecast.to_csv(DATA / "exp_snaive.csv", index=False)
