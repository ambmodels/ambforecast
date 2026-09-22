"""Helpers."""

from dataclasses import fields

import pandas as pd


class CustomRepr:
    """Provide a compact representation for dataclasses containing DataFrames.

    DataFrames can be large, so their full contents are not included when an
    instance is displayed. Instead, DataFrames are shown with their shape and
    column names.
    """

    @staticmethod
    def _format_value(value):
        """Replace DataFrames with a short summary."""
        if isinstance(value, pd.DataFrame):
            return (
                f"DataFrame(shape={value.shape}, "
                f"columns={value.columns.tolist()!r})"
            )
        return value

    def __repr__(self):
        """Return a concise representation that summarises DataFrame fields."""
        values = []
        for field in fields(self):
            value = getattr(self, field.name)
            values.append(f"{field.name}={self._format_value(value)}")
        return f"{type(self).__name__}({', '.join(values)})"

    def __rich_repr__(self):
        """Yield dataclass fields for Rich to display."""
        for field in fields(self):
            value = getattr(self, field.name)
            yield field.name, self._format_value(value)


def merge_regressor(data, regressor):
    """Merge a regressor and check it covers required dates and area.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing `ds` and `area` columns.
    regressor : ProphetRegressor
        Regressor configuration and data.

    Returns
    -------
    data : pd.DataFrame
        Data with the regressor column added.

    """
    data = pd.merge(
        data,
        regressor.data[["ds", "area", regressor.name]],
        on=["ds", "area"],
        how="left",
        validate="one_to_one",
    )

    missing = data.loc[
        data[regressor.name].isna(),
        ["ds", "area"],
    ]

    if not missing.empty:
        raise ValueError(
            f"Regressor {regressor.name!r} has missing values for:\n"
            f"{missing.to_string(index=False)}"
        )

    return data