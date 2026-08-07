"""Ensemble forecast."""

def calculate_ensemble(forecast_results, methods):
    """Find the average forecast and intervals.

    Using the provided table of forecasts, it will group and find the mean
    forecast and pi_intervals for each county, currency (metric) and date.

    Parameters
    ----------
    forecast_results : pd.DataFrame
        Dataframe of forecast results returned by Forecaster.results.
    methods : list[str]
        List of methods to include in ensemble - these are the names in the
        "method" column of the Forecaster.results dataframe.
    """
    df = forecast_results[forecast_results["method"].isin(methods)]
    ensemble = (
        df.groupby(["county", "currency", "ds"])
        .agg({"forecast": "mean", "pi_lower": "mean", "pi_upper": "mean"})
        .reset_index()
    )
    return ensemble
