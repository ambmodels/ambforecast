"""Structures used in code."""

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
