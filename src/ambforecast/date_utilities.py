import pandas as pd


def next_weekday(weekday):
    """Calculate the date of the next given weekday after today.

    Parameters
    ----------
    weekday : str
        Three-letter pandas offset alias for the day of week,
        e.g. "MON" for Monday, "SUN" for Sunday.

    Returns
    -------
    pd.Timestamp
        Date of the next occurrence of the given weekday after today.

    """
    today = pd.Timestamp.today()
    days = (
        pd.date_range(today, freq=f"W-{weekday}", periods=2)
        .normalize()
        .to_series()
    )
    return days.loc[days > today].iloc[0]
