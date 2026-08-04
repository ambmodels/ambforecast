import pandas as pd


def next_monday():
    """Calculate the date of the next Monday after today.

    Returns
    -------
    pd.Timestamp
        Date of the next Monday after today's date.
    """
    today = pd.Timestamp.today()
    mondays = (
        pd.date_range(today, freq="W-MON", periods=2).normalize().to_series()
    )
    return mondays.loc[mondays > today].iloc[0]


def next_sunday():
    """Calculate the date of the next Sunday after today.

    Returns
    -------
    pd.Timestamp
        Date of the next Sunday after today's date.
    """
    today = pd.Timestamp.today()
    sundays = (
        pd.date_range(today, freq="W-SUN", periods=2).normalize().to_series()
    )
    return sundays.loc[sundays > today].iloc[0]
