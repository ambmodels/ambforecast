"""Functions to support handling COVID-19 period."""

import pandas as pd


def create_dummy(df_historic, start_date, end_date, dummy_name, horizon=0):
    """Create a dummy variable for historic and optional future dates.

    Parameters
    ----------
    df_historic : pd.DataFrame
        Historic data.
    start_date : pd.Timestamp
        First date to replace, inclusive.
    end_date : pd.Timestamp
        Last date to replace, inclusive.
    dummy_name : str
        Name for the dummy variable.
    horizon : int
        Planned forecast horizon, as will require dummy for those dates too.

    Returns
    -------
    dummy : pd.DataFrame
        Dataframe with each date (+ horizon) and area from `df_historic`, plus
        a dummy indicator marked as 1 if between dates and otherwise 0.

    """
    # Get unique dates and areas
    dummy = df_historic[["ds", "area"]].drop_duplicates()

    # Extend date and area combinations into horizon
    if horizon:
        future_rows = []

        for area, group in dummy.groupby("area"):
            dates = pd.date_range(
                start=group["ds"].max() + pd.Timedelta(days=1),
                periods=horizon,
                freq="D",
            )
            future_rows.append(
                pd.DataFrame({"ds": dates, "area": area})
            )

        dummy = pd.concat([dummy, *future_rows], ignore_index=True)

    # Generate dummy variable
    dummy[dummy_name] = dummy["ds"].between(start_date, end_date).astype(int)
    return dummy


def naivefill_prior_year(df_historic, start_date, end_date):
    """Replace values between two dates based on same date in prior year.

    For 29th Feburary, uses 28th Feburary from the prior year.

    Parameters
    ----------
    df_historic : pd.DataFrame
        Historic data.
    start_date : pd.Timestamp
        First date to replace, inclusive.
    end_date : pd.Timestamp
        Last date to replace, inclusive.

    Returns
    -------
    result : pd.DataFrame
        Copy of `df_historic` with values in requested date range replaced.

    """
    result = df_historic.copy()

    # Create look-up where each key is a tuple of metric, area and ate
    values = result.set_index(["metric", "area", "ds"])["y"].to_dict()

    # Loop through rows requiring replacement
    rows_to_replace = result["ds"].between(start_date, end_date)
    for idx in result.index[rows_to_replace]:
        # Get date
        date = result.at[idx, "ds"]

        # Find date to fetch replacement values from
        # pd.Timestamp.replace would fail for, e.g.,
        # 2024-02-29 -> 2023-02-29, so use 28 February.
        if date.month == 2 and date.day == 29:
            prior_year_date = date.replace(year=date.year - 1, day=28)
        else:
            prior_year_date = date.replace(year=date.year - 1)

        # Check if key is available
        key = (
            result.at[idx, "metric"],
            result.at[idx, "area"],
            prior_year_date,
        )
        if key not in values:
            raise ValueError(f"No prior value available for: {key}")

        # Extract value from look-up, replace in dataframe and add to look-up
        replacement = values[key]
        result.at[idx, "y"] = replacement
        values[
            (
                result.at[idx, "metric"],
                result.at[idx, "area"],
                date,
            )
        ] = replacement

    return result
