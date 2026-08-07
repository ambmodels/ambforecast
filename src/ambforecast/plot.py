"""Plotting functions."""

import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def plot_forecast(
    df_historic,
    forecast,
    metric,
    county,
    name,
    historic_length,
    forecast_colour,
):
    """Plot actual and forecast values with 95% prediction intervals.

    Parameters
    ----------
    df_historic : pd.DataFrame
        Historic data.
    forecast : pd.DataFrame
        Forecast.
    metric : str
        Name of metric.
    county : str
        Name of county.
    name : str
        Name of the forecast to plot - corresponds to "name" column in the
        forecast results dataframe.
    historic_length : int
        Number of days of data to include from the historic data. For example,
        to show 6 weeks of actual data before the forecast, set to 6*7.
    forecast_colour : str
        Forecast colour.

    """
    # Get the final X days specified of the historic data
    date_min = df_historic["ds"].max() - dt.timedelta(days=historic_length - 1)
    historic_finaldays = df_historic[df_historic["ds"] >= date_min]
    historic_plot = historic_finaldays.loc[
        (historic_finaldays["currency"] == metric)
        & (historic_finaldays["ora"] == county),
        ["ds", "y"],
    ]

    # Filter to the relevant forecast
    forecast_plot = forecast.loc[
        (forecast["currency"] == metric)
        & (forecast["county"] == county)
        & (forecast["name"] == name),
        ["ds", "forecast", "pi_lower", "pi_upper"],
    ]

    df = pd.concat([historic_plot, forecast_plot]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["ds"], df["y"], color="black", label="Actual")
    ax.plot(df["ds"], df["forecast"], color=forecast_colour, label="Forecast")
    ax.fill_between(
        df["ds"],
        df["pi_lower"],
        df["pi_upper"],
        color=forecast_colour,
        alpha=0.2,
        label="95% Prediction Interval",
    )

    # Vertical line marking start of the forecast
    boundary = forecast_plot["ds"].min()
    ax.axvline(boundary, color="red")
    y_top = ax.get_ylim()[1]
    ax.annotate(
        "Forecast begins",
        xy=(boundary, y_top),
        xytext=(5, -5),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=9,
    )

    # X axis ticks for dates every 7 days from first date
    ticks = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq="7D")
    ax.xaxis.set_major_locator(mticker.FixedLocator(mdates.date2num(ticks)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b %y"))
    fig.autofmt_xdate(rotation=45, ha="right")

    ax.set_ylim(ymin=0)

    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.set_title(f"{county} {metric} {name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
