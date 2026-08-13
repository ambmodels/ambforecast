"""Plotting functions."""

import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

DEFAULT_COLOURS = {
    "Calls": "tab:blue",
    "Incidents": "tab:orange",
    "Responses": "tab:green",
}


def plot_forecast(
    train,
    forecast,
    test=None,
    metric=None,
    area=None,
    historic_length=42,
    forecast_colour=None,
    forecast_name="Forecast",
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
    metric : str | None
        Metric to plot. Provide with `area` when `forecast` contains more
        than one metric/area series. If None, the metric is taken from
        `forecast`.
    area : str | None
        Area to plot. Provide with `metric` when `forecast` contains more
        than one metric/area series. If None, the area is taken from
        `forecast`.
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
    if (metric is None) != (area is None):
        raise ValueError("Provide both 'metric' and 'area', or neither.")

    # If metric and area are not provided, forecast must contain one series.
    if metric is None:
        pairs = forecast[["metric", "area"]].drop_duplicates()

        if len(pairs) != 1:
            raise ValueError(
                "Forecast contains multiple metric/area series. "
                "Provide 'metric' and 'area'."
            )

        metric = pairs["metric"].iloc[0]
        area = pairs["area"].iloc[0]

    # Filter forecast in case it contains multiple metric/area series.
    forecast_plot = forecast[
        (forecast["metric"] == metric) & (forecast["area"] == area)
    ].sort_values("ds")

    if forecast_plot.empty:
        raise ValueError(
            f"No forecast found for metric={metric!r}, area={area!r}."
        )

    # Filter training data to relevant metric and area
    observed_plot = train[
        (train["metric"] == metric) & (train["area"] == area)
    ].copy()

    # Get the final X days specified of the training data
    date_min = forecast_plot["ds"].min() - dt.timedelta(days=historic_length)
    observed_plot = observed_plot[observed_plot["ds"] >= date_min].sort_values(
        "ds"
    )

    # If provided, also filter test data to relevant metric and area and
    # combine into the training dataframe
    if test is not None:
        test_plot = test[
            (test["metric"] == metric) & (test["area"] == area)
        ].sort_values("ds")
        observed_plot = pd.concat(
            [observed_plot, test_plot], ignore_index=True
        ).sort_values("ds")

    # Ensure forecast_plot is sorted by date
    forecast_plot = forecast_plot.sort_values("ds")

    if forecast_colour is None:
        forecast_colour = DEFAULT_COLOURS.get(metric, "tab:red")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Observed data: training plus held-out test data, if provided
    ax.plot(
        observed_plot["ds"],
        observed_plot["y"],
        color="black",
        label="Observed",
        zorder=2,
    )

    # Forecast
    ax.plot(
        forecast_plot["ds"],
        forecast_plot["forecast"],
        color=forecast_colour,
        label="Forecast",
        zorder=3,
    )
    ax.fill_between(
        forecast_plot["ds"].to_numpy(),
        forecast_plot["pi_lower"],
        forecast_plot["pi_upper"],
        color=forecast_colour,
        alpha=0.2,
        label="95% Prediction Interval",
        zorder=1,
    )

    # Vertical line marking start of the forecast
    boundary = forecast_plot["ds"].min()
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
        end=forecast_plot["ds"].max(),
        freq="7D",
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


def plot_cross_validation(
    historic,
    forecast,
    metric,
    area,
    start_date=None,
    end_date=None,
    forecast_colour=None,
    forecast_name="Forecast",
):
    """Plot observed and forecast values from all cross-validation folds.

    Parameters
    ----------
    historic : pd.DataFrame
        Historic data.
    forecast : pd.DataFrame
        Forecast results.
    metric : str
        Name of metric.
    area : str
        Name of area.
    start_date : pd.Timestamp
        Date to show data from.
    end_date : pd.Timestamp
        Date to show data up to.
    forecast_colour : str
        Colour for the forecast line and prediction interval.
    forecast_name : str
        Label to use for forecast in title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Forecast figure.

    """
    historic_plot = historic[
        (historic["metric"] == metric) & (historic["area"] == area)
    ].sort_values("ds")
    forecast_plot = forecast[
        (forecast["metric"] == metric) & (forecast["area"] == area)
    ].sort_values("ds")

    if start_date is not None:
        historic_plot = historic_plot[historic_plot["ds"] >= start_date]
        forecast_plot = forecast_plot[forecast_plot["ds"] >= start_date]
    if end_date is not None:
        historic_plot = historic_plot[historic_plot["ds"] <= end_date]
        forecast_plot = forecast_plot[forecast_plot["ds"] <= end_date]

    if forecast_colour is None:
        forecast_colour = DEFAULT_COLOURS.get(metric, "tab:red")

    fig, ax = plt.subplots(figsize=(16, 6))

    # Observed values
    ax.plot(
        historic_plot["ds"],
        historic_plot["y"],
        color="black",
        label="Observed",
        zorder=2,
    )

    # Forecast
    ax.plot(
        forecast_plot["ds"],
        forecast_plot["forecast"],
        color=forecast_colour,
        label="Forecast",
        zorder=3,
    )
    ax.fill_between(
        forecast_plot["ds"].to_numpy(),
        forecast_plot["pi_lower"],
        forecast_plot["pi_upper"],
        color=forecast_colour,
        alpha=0.2,
        label="95% Prediction Interval",
        zorder=1,
    )

    # Start and end dates of cross-validation folds
    fold_dates = (
        forecast_plot.groupby("fold")["ds"]
        .agg(["min", "max"])
        .stack()
        .drop_duplicates()
        .sort_values()
    )
    for i, date in enumerate(fold_dates):
        ax.axvline(
            date,
            color="lightgrey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
            zorder=0,
            label="Fold boundary" if i == 0 else "_nolegend_",
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate()

    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.set_title(f"{area} {metric} {forecast_name} Cross-Validation")
    ax.legend()

    fig.tight_layout()
