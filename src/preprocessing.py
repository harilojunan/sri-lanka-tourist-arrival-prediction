import pandas as pd


def _require_arrival_columns(data: pd.DataFrame) -> None:
    """Raise a clear error when the source is missing a required field."""
    required = {"Date", "Total_Arrivals"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

def _validate_arrival_values(data: pd.DataFrame) -> pd.DataFrame:
    """Convert source values to expected types and validate each observation."""
    clean = data.copy()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean["Total_Arrivals"] = pd.to_numeric(
        clean["Total_Arrivals"], errors="coerce"
    )
    if clean["Date"].isna().any():
        raise ValueError("Arrival data contains invalid dates.")
    if clean["Total_Arrivals"].isna().any():
        raise ValueError("Arrival data contains missing or non-numeric values.")
    if (clean["Total_Arrivals"] < 0).any():
        raise ValueError("Arrival data contains negative values.")
    if clean["Date"].duplicated().any():
        raise ValueError("Arrival data contains duplicate dates.")
    return clean