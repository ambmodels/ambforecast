"""Naive benchmark model."""

from dataclasses import dataclass

import pandas as pd
from forecast_tools.baseline import SNaive

from .structures import CustomRepr


@dataclass(kw_only=True, repr=False)
class SNaiveParams(CustomRepr):
    """Parameters for the SNaive model.

    Parameters
    ----------
    period : float
        Seasonal period of the data. For example, day of week = 7,
        monthly = 12, and quarterly = 4.
    interval_width : float
        Width of the prediction intervals - for example, 0.95 will produce
        95% prediction intervals.

    """

    period: int = 7
    interval_width: float = 0.95


def snaive(train, params, test=None, horizon=None):
    """Fit Seasonal Naive model and generate forecast.

    Parameters
    ----------
    train : pd.DataFrame
        Historic training data.
    params : SNaiveParams
        Parameters controlling the SNaive model.
    test : pd.DataFrame | None
        Data containing the dates to forecast in a `ds` column. Typically
        a held-out test set or cross-validation fold.
    horizon : int | None
        Number of days to forecast, after the final date in `train`. To be
        used if there is no test set (e.g., if actually predicting into future
        with no data to compare against).

    Returns
    -------
    pd.DataFrame
        Forecast dataframe.

    """
    if (test is None) == (horizon is None):
        raise ValueError("Provide exactly one of 'test' or 'horizon'.")

    # Convert training data into required format
    train = train.sort_values("ds").set_index("ds")["y"]

    # Fit model
    model = SNaive(period=params.period)
    model.fit(train=train)

    # Get dates for forecast
    if test is not None:
        forecast_dates = test["ds"].reset_index(drop=True)
    else:
        forecast_dates = pd.date_range(
            start=train.index.max() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

    pred, pred_int = model.predict(
        horizon=len(forecast_dates),
        return_predict_int=True,
        alpha=[1 - params.interval_width],
    )
    return pd.DataFrame(
        {
            "ds": forecast_dates,
            "forecast": pred,
            "pi_lower": pred_int[0][:, 0],
            "pi_upper": pred_int[0][:, 1],
        }
    )
