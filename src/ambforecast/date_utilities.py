import pandas as pd


def date_ordinal(d):
    """Returns the ordinal date suffix for a given integer.

    Args:
        d (int): The integer for which the ordinal suffix is required.

    Returns:
        str: The ordinal suffix ('st', 'nd', 'rd', or 'th') based on the integer value.

    Example:
        >>> date_ordinal(1)
        'st'
        >>> date_ordinal(2)
        'nd'
        >>> date_ordinal(3)
        'rd'
        >>> date_ordinal(4)
        'th'

    """
    return (
        "th"
        if 4 <= d % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    )


def next_monday():
    """Calculate the date of the next Monday after today.

    Returns:
        pd.Timestamp: A pandas Timestamp object representing the date
        of the next Monday after today's date.

    Example:
        >>> next_monday()
        Timestamp('YYYY-MM-DD 00:00:00')

    """
    today = pd.Timestamp.today()
    mondays = (
        pd.date_range(today, freq="W-MON", periods=2).normalize().to_series()
    )
    return mondays.loc[mondays > today].iloc[0]


def next_sunday():
    """Calculate the date of the next Sunday after today.

    Returns:
        pd.Timestamp: A pandas Timestamp object representing the date
        of the next Sunday after today's date.

    Example:
        >>> next_sunday()
        Timestamp('YYYY-MM-DD 00:00:00')

    """
    today = pd.Timestamp.today()
    sundays = (
        pd.date_range(today, freq="W-SUN", periods=2).normalize().to_series()
    )
    return sundays.loc[sundays > today].iloc[0]
