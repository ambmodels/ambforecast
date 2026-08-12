"""Prepare and validate datasets used in forecasting."""

import pandas as pd

# ----------------------------------------------------------------------------
# Historic data
# ----------------------------------------------------------------------------


def prepare_historic(data, metrics=("Calls", "Responses", "Incidents")):
    """Prepare historic data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw historic data.
    metrics : list[str] | tuple[str, ...]
        Metrics to keep.

    Returns
    -------
    data : pd.DataFrame
        Cleaned historic data.

    """
    # Rename ora and currency columns to instead use area and metric
    data = data.rename(columns={"ora": "area", "currency": "metric"})

    # Filter to metrics
    data = data[data["metric"].isin(metrics)]

    # Remove any missing data
    data = data.dropna()

    # Select which columns to keep
    data = data[["ds", "metric", "area", "y"]]

    # Convert ds column to datetime
    data["ds"] = pd.to_datetime(data["ds"])

    return data.sort_values(by=["ds", "area", "metric"])


def validate_historic(data):
    """Validate historic data.

    Parameters
    ----------
    data : pd.DataFrame
        Historic data.

    Raises
    ------
    ValueError
        If any issues are identified in the data.

    """
    # Check for missing values
    find_missing_values(data)

    # Check data types of each column
    if not pd.api.types.is_datetime64_any_dtype(data["ds"]):
        raise ValueError("'ds' must be datetime.")
    for col in ["metric", "area"]:
        if not pd.api.types.is_string_dtype(data[col]):
            raise ValueError(f"'{col}' must be str.")
    if not pd.api.types.is_integer_dtype(data["y"]):
        raise ValueError("'y' must be integer.")

    # Check for missing and duplicate dates
    find_missing_dates(data, groups=["metric", "area"])
    find_duplicate_dates(data, groups=["metric", "area"])

    print("✅ No problems identified in historic data.")


# ----------------------------------------------------------------------------
# Holiday data
# ----------------------------------------------------------------------------


def prepare_holidays(data):
    """Prepare holiday data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw holiday data.

    Returns
    -------
    data : pd.DataFrame
        Cleaned holiday data.

    """
    # Rename county column for consistency
    data = data.rename(columns={"county": "area"})

    # Convert ds column to datetime
    data["ds"] = pd.to_datetime(data["ds"])

    return data


def validate_holidays(data):
    """Validate holiday data.

    Parameters
    ----------
    data : pd.DataFrame
        Holiday data.

    Raises
    ------
    ValueError
        If any issues are identified in the data.

    """
    # Check for missing values
    find_missing_values(data)

    # Check data types of each column
    if not pd.api.types.is_datetime64_any_dtype(data["ds"]):
        raise ValueError("'ds' must be datetime.")
    for col in ["holiday", "area"]:
        if not pd.api.types.is_string_dtype(data[col]):
            raise ValueError(f"'{col}' must be str.")
    for col in ["lower_window", "upper_window"]:
        if not pd.api.types.is_integer_dtype(data[col]):
            raise ValueError(f"'{col}' must be integer.")

    print("✅ No problems identified in holiday data.")


# ----------------------------------------------------------------------------
# Temperature data
# ----------------------------------------------------------------------------


def prepare_temp(data):
    """Prepare midas air temperature data.

    Extract the minimum and maximum temperature in each area per day.

    Parameters
    ----------
    data : pd.DataFrame
        Raw temperature data.

    Returns
    -------
    pd.DataFrame
        Cleaned temperature data.

    """
    # Convert full datetime into just the date
    data["OB_END_TIME"] = pd.to_datetime(data["OB_END_TIME"])
    data["ds"] = data["OB_END_TIME"].dt.normalize()

    # Rename ICB column for consistency
    data = data.rename(columns={"ICB": "area"})

    # Find minimum and maximum per ICB per day
    icb_min = (
        data.groupby(["ds", "area"])["MIN_AIR_TEMP"].agg("min").reset_index()
    )
    icb_max = (
        data.groupby(["ds", "area"])["MAX_AIR_TEMP"].agg("max").reset_index()
    )
    icb_temp = pd.merge(icb_min, icb_max, on=["ds", "area"])

    # Find minimum and maximum across whole trust per day
    trust_min = data.groupby("ds")["MIN_AIR_TEMP"].min().reset_index()
    trust_max = data.groupby("ds")["MAX_AIR_TEMP"].max().reset_index()
    trust_temp = pd.merge(trust_min, trust_max, on="ds")
    trust_temp["area"] = "Trust"

    return pd.concat([icb_temp, trust_temp]).sort_values(by=["ds", "area"])


def interpolate_temp(data):
    """Interpolate missing air temperature data.

    Parameters
    ----------
    data : pd.DataFrame
        Temperature data.

    Returns
    -------
    pd.DataFrame
        Temperature data with missing rows completed via linear interpolation.

    """
    filled = []

    for area, df in data.groupby("area"):
        df = df.sort_values("ds").set_index("ds")

        # Add new rows for missing dates which will be all NA, so set area
        df = df.asfreq("D")
        df["area"] = area

        # Then interpolate temperatures for the NA in the new rows
        df[["MIN_AIR_TEMP", "MAX_AIR_TEMP"]] = df[
            ["MIN_AIR_TEMP", "MAX_AIR_TEMP"]
        ].interpolate(method="linear")

        filled.append(df.reset_index())

    return (
        pd.concat(filled, ignore_index=True)
        .sort_values(by=["ds", "area"])
        .reset_index(drop=True)
    )


def validate_temp(data):
    """Validate midas air temperature data.

    Parameters
    ----------
    data : pd.DataFrame
        Air temperature data.

    Raises
    ------
    ValueError
        If any issues are identified in the data.

    """
    # Check for missing values
    find_missing_values(data)

    # Check data types of each column
    if not pd.api.types.is_datetime64_any_dtype(data["ds"]):
        raise ValueError("'ds' must be datetime.")
    if not pd.api.types.is_string_dtype(data["area"]):
        raise ValueError("'county' must be str.")
    for col in ["MIN_AIR_TEMP", "MAX_AIR_TEMP"]:
        if not pd.api.types.is_float_dtype(data[col]):
            raise ValueError(f"'{col}' must be float.")

    # Check for missing and duplicate dates
    find_missing_dates(data, groups="area")
    find_duplicate_dates(data, groups="area")

    print("✅ No problems identified in air temperature data.")


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def find_missing_values(data):
    """Find any rows with missing values in a dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Data to check.

    """
    missing_values = data.loc[data.isna().any(axis=1)]
    if not missing_values.empty:
        raise ValueError(
            f"The following rows contain missing values: {missing_values}"
        )


def find_missing_dates(data, groups):
    """Identify missing dates.

    For example, if data runs from 1 January to 31 December, but one of the
    groups doesn't have a row for a particular date.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing a datetime `ds` column.
    groups : list[str]
        List of column names to group by when checking.

    Raises
    ------
    ValueError
        If there are missing dates.

    """
    missing = {}

    for group, series in data.groupby(groups):
        dates = series["ds"].sort_values()
        expected_dates = pd.date_range(dates.min(), dates.max(), freq="D")
        missing_dates = expected_dates.difference(dates)

        if not missing_dates.empty:
            missing[group] = missing_dates

    if missing:
        raise ValueError(
            f"There are missing rows for the following dates: {missing}"
        )


def find_duplicate_dates(data, groups):
    """Extract duplicate dates within groups.

    For example, if a group has multiple entries for a single date.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing a datetime `ds` column.
    groups : list[str]
        List of column names to group by when checking.

    Raises
    ------
    ValueError
        If there are missing dates.

    """
    duplicates = {}

    for group, series in data.groupby(groups):
        duplicate_dates = series.loc[
            series["ds"].duplicated(keep=False),
            "ds",
        ].unique()

        if len(duplicate_dates) > 0:
            duplicates[group] = pd.DatetimeIndex(duplicate_dates).sort_values()

    if duplicates:
        raise ValueError(
            f"There are duplicate rows for the following dates: {duplicates}"
        )
