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

def _validate_month_start_dates(data: pd.DataFrame) -> None:
    """Reject dates that do not refer to the first day of a month."""
    month_starts = data["Date"].dt.to_period("M").dt.to_timestamp()
    if not data["Date"].eq(month_starts).all():
        raise ValueError("Dates must be month-start dates.")
    
def _validate_complete_coverage(
    data: pd.DataFrame, start: str, end: str | None
) -> None:
    """Require every month-start date in the requested period, exactly once."""
    if end is None:
        end = data["Date"].max().strftime("%Y-%m-%d")
    expected = pd.date_range(start, end, freq="MS")
    actual = pd.DatetimeIndex(data["Date"])
    if not actual.equals(expected):
        raise ValueError(
            f"Expected exactly {len(expected)} consecutive month-start "
            f"observations from {start} through {end}."
        )

def validate_monthly_data(
    data: pd.DataFrame,
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Validate a continuous monthly series without silently repairing it."""
    _require_arrival_columns(data)
    clean = _validate_arrival_values(data)
    clean = clean.sort_values("Date").reset_index(drop=True)
    _validate_month_start_dates(clean)
    _validate_complete_coverage(clean, start, end)
    return clean