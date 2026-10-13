import pandas as pd


def _require_arrival_columns(data: pd.DataFrame) -> None:
    """Raise a clear error when the source is missing a required field."""
    required = {"Date", "Total_Arrivals"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")