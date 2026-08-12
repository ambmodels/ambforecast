"""Plotting functions."""

import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def plot_forecast(
    train,
    forecast,
    test=None,
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
    test : pd.DataFrame | None
        Optional held-out observed data to display alongside the forecast.
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
    # Identify the relevant metric and area
    metric = forecast["metric"].iloc[0]
    area = forecast["area"].iloc[0]

    # Filter training data to relevant metric and area
    observed_plot = train[
        (train["metric"] == metric)
        & (train["area"] == area)
    ].copy()

    # Get the final X days specified of the training data
    date_min = forecast["ds"].min() - dt.timedelta(days=historic_length)
    observed_plot = observed_plot[
        observed_plot["ds"] >= date_min
    ].sort_values("ds")

    # If provided, also filter test data to relevant metric and area and
    # combine into the training dataframe
    if test is not None:
        test_plot = test[
            (test["metric"] == metric)
            & (test["area"] == area)
        ].sort_values("ds")
        observed_plot = (
            pd.concat([observed_plot, test_plot], ignore_index=True)
            .sort_values("ds")
        )

    # Ensure forecast is sorted by date
    forecast = forecast.sort_values("ds")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Observed data: training plus held-out test data, if provided
    ax.plot(
        observed_plot["ds"],
        observed_plot["y"],
        color="black",
        label="Observed",
        zorder=2
    )

    # Forecast
    ax.plot(
        forecast["ds"],
        forecast["forecast"],
        color=forecast_colour,
        label="Forecast",
        zorder=3
    )
    ax.fill_between(
        forecast["ds"].to_numpy(),
        forecast["pi_lower"],
        forecast["pi_upper"],
        color=forecast_colour,
        alpha=0.2,
        label="95% Prediction Interval",
        zorder=1
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
        start=observed_plot["ds"].min(),
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
