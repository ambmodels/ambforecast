"""Generate fake data, run the forecast, and save results to CSV.

To run: python tests/regression_results/generate_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ambforecast.utility import default_ensemble, pre_process_daily_data

OUTPUT = Path(__file__).parent


def make_fake_data():
    """Create small fake dataframe matching real data schema."""
    # Set-up over two years of dates
    dates = pd.date_range(start="2010-01-01", periods=800, freq="D")

    # Sine wave cycling once per 7 days between -5 and + 5, to mimic variation
    # by day of week
    weekly = 5 * np.sin(2 * np.pi * dates.dayofweek / 7)

    # Linear increase from 0 to 10 over full date range, to mimic trend
    trend = np.linspace(0, 10, len(dates))

    # Noise (mean 0, std 3) per day
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 3, len(dates))

    # Combine baseline (50) + weekly + trend + noise, and clip negatives to 0
    y = np.clip(np.round(50 + weekly + trend + noise).astype(int), 0, None)

    # Construct dataframes
    df_historic = pd.DataFrame(
        {
            "ds": dates,
            "currency": "Responses",
            "ora": "FakeCounty",
            "y": y,
            "forecast_type": "daily",
        }
    )
    df_holidays = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-12-25",
                    "2011-01-01",
                    "2011-12-25",
                    "2012-01-01",
                ]
            ),
            "holiday": [
                "New Year",
                "Christmas",
                "New Year",
                "Christmas",
                "New Year",
            ],
            "lower_window": [0, -2, 0, -2, 0],
            "upper_window": [0, 2, 0, 2, 0],
            "county": [
                "FakeCounty",
                "FakeCounty",
                "FakeCounty",
                "FakeCounty",
                "FakeCounty",
            ],
        }
    )
    return df_historic, df_holidays


if __name__ == "__main__":
    # Generate fake data
    df_historic, df_holidays = make_fake_data()
    df_historic.to_csv(OUTPUT / "fake_historic.csv", index=False)
    df_holidays.to_csv(OUTPUT / "fake_holidays.csv", index=False)

    # Process data
    df_in = df_historic.drop(["currency", "forecast_type"], axis=1)
    clean = pre_process_daily_data(df_in, observation_col="y", index_col="ds")

    # Run ensemble forecast
    np.random.seed(42)
    model = default_ensemble(county="FakeCounty", df_holiday=df_holidays)
    model.fit(clean)
    forecast = model.predict(horizon=30, return_all_models=True)
    forecast.to_csv(OUTPUT / "forecast.csv")
