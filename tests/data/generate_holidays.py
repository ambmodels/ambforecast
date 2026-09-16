"""Generate simple holiday dataframe with some major holidays."""

import pandas as pd

data = pd.read_csv("responses_2013_to_2019.csv", parse_dates=["ds"])

# Define day and month of holidays to include
holiday_dates = {
    "New Year's Day": (1, 1),
    "Christmas Day": (12, 25),
    "Boxing Day": (12, 26),
}

# Identify years of data that we require holidays for
years = range(
    data["ds"].min().year,
    (data["ds"].max() + pd.Timedelta(weeks=52)).year + 1,
)

# Create holiday dataframe in Prophet format
holidays = (
    pd.DataFrame(
        [
            {
                "ds": pd.Timestamp(year, month, day),
                "holiday": holiday,
            }
            for year in years
            for holiday, (month, day) in holiday_dates.items()
        ]
    )
    .assign(lower_window=0, upper_window=0)
    .merge(
        data[["area"]].dropna().drop_duplicates(),
        how="cross",
    )
    .assign(ds=lambda x: x["ds"].dt.to_period("D"))
    .sort_values(["ds", "holiday", "area"])
    .reset_index(drop=True)
)

holidays.to_csv("holidays.csv", index=False)
