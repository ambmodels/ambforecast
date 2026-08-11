"""Functions for splitting data into training and test datasets."""

import datetime as dt


def train_test_split(data, horizon, min_train=365*2, test_end=None):
    """Create a single train/test split.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing a `ds` column.
    horizon : int
        Number of daily observations in the test set.
    min_train : int
        Minimum number of days to include in training sample. By default,
        set to 2 years as that allows detection of yearly seasonality.
    test_end : pd.Timestamp | None
        Final date in the test set. If None, uses the latest date in the data.

    Returns
    -------
    train, test : tuple[pd.DataFrame, pd.DataFrame]
        Training and test data.

    """
    if test_end is None:
        test_end = data["ds"].max()

    # Find start date of test sample
    test_start = test_end - dt.timedelta(days=horizon - 1)

    # Split into train and test dataset
    train = data[data["ds"] < test_start]
    test = data[data["ds"] >= test_start]

    if train["ds"].nunique() < min_train:
        raise ValueError(
            f"Insufficient data. Training sample requires at least "
            f"{min_train} days but only has {train['ds'].nunique()}."
        )

    return train, test


def rolling_forecast_origin(data, horizon, step, min_train=365*2):
    """Create rolling forecast origin train/test samples.

    Samples are generated from the most recent data backwards. The test period
    moves back by `step` days for each fold.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing a `ds` column.
    horizon : int
        Number of daily observations in the test set.
    step : int
        How many days to move by before creating a new sample. Warning:
        using a step of 365 will produce test samples all at approximately
        the same time of year.
    min_train : int
        Minimum number of days to include in training sample. By default,
        set to 2 years as that allows detection of yearly seasonality.

    Returns
    -------
    train, test : tuple[list[pd.DataFrame],list[pd.DataFrame]]
        Training and test dataframes for each fold, ordered from most
        recent to oldest.

    """
    if data["ds"].nunique() < min_train + horizon:
        raise ValueError(
            "Insufficient data to create at least one fold. Training "
            f"requires {min_train} days and testing requires {horizon} "
            f"days, but data only has {data['ds'].nunique()} days."
        )

    train_samples = []
    test_samples = []
    test_end = data["ds"].max()

    while True:
        try:
            train, test = train_test_split(
                data=data,
                horizon=horizon,
                min_train=min_train,
                test_end=test_end,
            )
        except ValueError:
            break

        train_samples.append(train)
        test_samples.append(test)

        test_end -= dt.timedelta(days=step)

    return train_samples, test_samples
