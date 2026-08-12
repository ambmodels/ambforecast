"""Plotting functions."""

import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def plot_forecast(
    train,
    forecast,
    historic_length=42,
    forecast_colour="tab:blue",
    forecast_name="Forecast"
):
    """Plot historic data and forecast with 95% prediction intervals.

    Parameters
    ----------
    train : pd.DataFrame
        Historic data used to train the model.
    forecast : pd.DataFrame
        Forecast.
    historic_length : int
        Number of historic days to display before the forecast.
    forecast_colour : str
        Colour for the forecast line and prediction interval.
    forecast_name : str
        Label to use for forecast in title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Forecast figure.

    """
    metric = forecast["metric"].iloc[0]
    area = forecast["area"].iloc[0]

    train_plot = train[
        (train["metric"] == metric)
        & (train["area"] == area)
    ].copy()

    # Get the final X days specified of the historic data
    date_min = forecast["ds"].min() - dt.timedelta(days=historic_length)
    train_plot = train_plot[
        train_plot["ds"] >= date_min
    ].sort_values("ds")

    forecast = forecast.sort_values("ds")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(train_plot["ds"], train_plot["y"], color="black", label="Observed")
    ax.plot(forecast["ds"], forecast["forecast"], color=forecast_colour, label="Forecast")
    ax.fill_between(
        forecast["ds"].to_numpy(),
        forecast["pi_lower"],
        forecast["pi_upper"],
        color=forecast_colour,
        alpha=0.2,
        label="95% Prediction Interval",
    )

    # Vertical line marking start of the forecast
    boundary = forecast["ds"].min()
    ax.axvline(boundary, color="red")
    ax.annotate(
        "Forecast begins",
        xy=(boundary, ax.get_ylim()[1]),
        xytext=(5, -5),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=9,
    )

    # X axis ticks for dates every 7 days from first date
    ticks = pd.date_range(
        start=train_plot["ds"].min(),
        end=forecast["ds"].max(),
        freq="7D"
    )
    ax.xaxis.set_major_locator(mticker.FixedLocator(mdates.date2num(ticks)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b %y"))
    fig.autofmt_xdate(rotation=45, ha="right")

    ax.set_ylim(ymin=0)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.set_title(f"{area} {metric} {forecast_name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
