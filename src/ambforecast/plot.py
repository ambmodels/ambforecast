"""Plotting functions."""

import datetime as dt
import warnings

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
    # Check for any gaps in dates
    dates = forecast["ds"].sort_values().unique()
    gaps = pd.date_range(start=dates.min(), end=dates.max()).difference(dates)
    if len(gaps) > 0:
        warnings.warn(
            "Uses continuous line but there are gaps in forecast.",
            category=UserWarning,
            stacklevel=1,
        )

    # Filter to relevant metric and area
    historic_plot = historic[
        (historic["metric"] == metric) & (historic["area"] == area)
    ].sort_values("ds")
    forecast_plot = forecast[
        (forecast["metric"] == metric) & (forecast["area"] == area)
    ].sort_values("ds")

    # Filter to specified start date and/or end date
    if start_date is not None:
        historic_plot = historic_plot[historic_plot["ds"] >= start_date]
        forecast_plot = forecast_plot[forecast_plot["ds"] >= start_date]
    if end_date is not None:
        historic_plot = historic_plot[historic_plot["ds"] <= end_date]
        forecast_plot = forecast_plot[forecast_plot["ds"] <= end_date]

    # If no colour provided, use default based on metric
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
    fig.autofmt_xdate()

    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.set_title(f"{area} {metric} {forecast_name} Cross-Validation")
    ax.legend()

    fig.tight_layout()


def plot_observed_against_forecast(
    historic, forecast, metric, area, colour=None, ax=None
):
    """Scatter plot of observed values against forecast values.

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
    colour : str
        Colour for the scatter plot.
    ax: matplotlib.axes.Axes | None
        Axes to create plot on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the scatter plot.

    """
    # Filter to relevant metric and area
    historic_plot = historic[
        (historic["metric"] == metric) & (historic["area"] == area)
    ]
    forecast_plot = forecast[
        (forecast["metric"] == metric) & (forecast["area"] == area)
    ]

    # Merge into single dataframe matched on date
    data = pd.merge(
        forecast_plot[["ds", "forecast"]],
        historic_plot[["ds", "y"]],
        how="left",
        on="ds",
        validate="one_to_one",
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # If no colour provided, use default based on metric
    if colour is None:
        colour = DEFAULT_COLOURS.get(metric, "tab:red")

    axis_min = data[["y", "forecast"]].min().min()
    axis_max = data[["y", "forecast"]].max().max()

    # Scatter plot
    ax.scatter(data["y"], data["forecast"], color=colour, alpha=0.5, s=25)

    # Diagonal line
    ax.axline((axis_min, axis_min), slope=1, color=colour, ls="--")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Forecast")
    ax.set_title(f"{metric} - {area}")

    # Square equal axises and dimensions
    padding = 0.05 * (axis_max - axis_min)
    ax.set_xlim(axis_min - padding, axis_max + padding)
    ax.set_ylim(axis_min - padding, axis_max + padding)
    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_holiday_coverage(data, start=None, historic_end=None):
    """Visualise when each holiday occurs over the whole time series.

    Create a plot where the X-axis is the date, and then there is a row for
    each holiday in the dataset, marking each date the holiday is on. This can
    be used to understand which holidays are included, whether they cover the
    full range of dates in the dataset, and whether they project beyond the
    current end date of the historic data.

    Parameters
    ----------
    data : pd.DataFrame
        Holidays data with one row for each holiday date (as returned by
        expand_holiday_windows()).
    start : pd.Timestamp | None
        Start date for the visualisation.
    historic_end : pd.Timestamp | None
        Final date in the historic data, marked as a vertical line on the plot.

    """
    plot_df = data.copy()

    if start is not None:
        plot_df = plot_df.loc[plot_df["ds"] >= pd.Timestamp(start)]

    # Sort by each holiday's first active date: earliest at the top
    holidays = (
        plot_df.groupby("holiday")["ds"].min().sort_values().index.tolist()
    )
    holiday_to_y = {holiday: i for i, holiday in enumerate(holidays)}

    _, ax = plt.subplots(figsize=(16, max(6, 0.36 * len(holidays) + 1.5)))

    ax.scatter(
        plot_df["ds"],
        plot_df["holiday"].map(holiday_to_y),
        marker="s",
        s=50,
        alpha=0.85,
        linewidths=0,
    )

    ax.axvline(historic_end, color="red")

    ax.set_yticks(range(len(holidays)))
    ax.set_yticklabels(holidays)
    ax.set_xlabel("Date")
    ax.set_ylabel("Holiday / event")
    ax.set_title("Holiday coverage")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.set_ylim(-0.75, len(holidays) - 0.25)
    ax.invert_yaxis()
    ax.margins(x=0.01)
    plt.tight_layout()
    plt.show()


def plot_error_over_time(error_df, error_name, metric, area, error_horizons):
    """Plot error over time.

    Parameters
    ----------
    error_df : pd.DataFrame
        Dataframe with errors for each fold, metric and area (as returned by
        forecast_errors()).
    error_name : str
        Name of error column in `error_df` to plot.
    metric : str
        Name of ambulance metric to plot.
    area : str
        Name of area to plot.
    error_horizons : int | list[int] | tuple[int, ...]
        One or more error horizons to plot, for example `42` or `[7, 42]`.

    """
    if isinstance(error_horizons, int):
        error_horizons = [error_horizons]
    error_horizons = sorted(set(error_horizons))

    # Filter to relevant metric and area
    error_plot = error_df[
        (error_df["metric"] == metric) & (error_df["area"] == area)
    ].sort_values("forecast_start_date")

    fig, ax = plt.subplots(figsize=(12, 5))

    for horizon in error_horizons:
        horizon_errors = error_plot[error_plot["horizon"] == horizon]
        ax.plot(
            horizon_errors["forecast_start_date"],
            horizon_errors[error_name],
            marker="o",
            label=f"{horizon}-day horizon",
        )

    ax.set_xlabel("Start date of cross-validation fold")
    ax.set_ylabel(error_name)
    ax.set_title(f"{error_name} {metric} {area}")
    ax.legend(title="Forecast horizon")
    ax.grid()


def plot_error_boxplot(error_df, error_name, metric, area, ax=None, title=None):
    """Plot cross-validation error distributions by forecast horizon.

    Parameters
    ----------
    error_df : pd.DataFrame
        Dataframe with errors for each fold, metric, area, and horizon.
    error_name : str
        Name of error column in `error_df` to plot.
    metric : str
        Name of ambulance metric to plot.
    area : str
        Name of area to plot.
    ax : matplotlib.axes.Axes | None
        Axis to plot on.
    title : str | None
        Title for the plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the boxplot.

    """
    error_plot = error_df.loc[
        (error_df["metric"] == metric) & (error_df["area"] == area)
    ]

    horizons = sorted(error_plot["horizon"].unique())

    box_data = [
        error_plot.loc[
            error_plot["horizon"] == horizon,
            error_name,
        ].dropna()
        for horizon in horizons
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if error_name == "coverage":
        ax.axhline(
            y=0.95,
            color="red",
            label="95% target coverage",
        )
        ax.legend()

    ax.boxplot(
        box_data,
        tick_labels=[f"{horizon}-day" for horizon in horizons],
    )

    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(error_name)
    ax.grid(axis="y")

    if title is None:
        ax.set_title(f"{error_name} {metric} {area}")
    else:
        ax.set_title(title)

    return ax
