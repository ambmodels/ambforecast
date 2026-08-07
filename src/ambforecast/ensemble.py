"""Ensemble forecast."""

def calculate_ensemble(forecast_results, names):
    """Find the average forecast and intervals.

    Using the provided table of forecasts, it will group and find the mean
    forecast and pi_intervals for each county, currency (metric) and date.

    Parameters
    ----------
    forecast_results : pd.DataFrame
        Dataframe of forecast results returned by Forecaster.results.
    names : list[str]
        List of forecasts to include in ensemble. These are the names in the
        "name" column of the Forecaster.results dataframe.
    """
    df = forecast_results[forecast_results["name"].isin(names)]
    ensemble = (
        df.groupby(["county", "currency", "ds"])
        .agg({"forecast": "mean", "pi_lower": "mean", "pi_upper": "mean"})
        .reset_index()
    )
    return ensemble
