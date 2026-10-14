from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import validate_monthly_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ECONOMIC_INDICATORS_PATH = (PROJECT_ROOT / "data" / "raw" / "cbsl_sources" / "cbsl_monthly.csv")
RAW_ARRIVALS = PROJECT_ROOT / "data/raw/sltlda_monthly_arrivals_2015_2026.csv"
POST_PERIOD_SOURCE = PROJECT_ROOT / "data/raw/sltlda_monthly_arrivals_2015_2026.csv"
PROCESSED_DATA = (PROJECT_ROOT / "data" / "processed" / "tourism_model_data.csv")
ECONOMIC_INDICATORS_COLUMNS = ["USD_LKR", "Inflation"]

def load_arrivals(path: Path = RAW_ARRIVALS) -> pd.DataFrame: 
    if not path.is_file():
        raise FileNotFoundError(
            f"SLTDA arrivals data not found: {path}. "
            "Restore the monthly source CSV before running the pipeline."
        )
    data = pd.read_csv(path)
    return validate_monthly_data(data)

def validate_economic_indicators(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", *ECONOMIC_INDICATORS_COLUMNS}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Economic indicators file has missing columns: {sorted(missing_columns)}")

    clean = data.loc[:, ["Date", *ECONOMIC_INDICATORS_COLUMNS]].copy()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    for column in ECONOMIC_INDICATORS_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean.isna().any().any():
        raise ValueError(f"Economic indicator data contains missing or invalid values.")
    if not np.isfinite(clean[ECONOMIC_INDICATORS_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Economic indicator data contains infinite values.")
    if clean["Date"].duplicated().any():
        raise ValueError("Economic indicator data contains duplicate months.")

    clean = clean.sort_values("Date").reset_index(drop=True)
    month_starts = clean["Date"].dt.to_period("M").dt.to_timestamp()
    if not clean["Date"].equals(month_starts):
        raise ValueError("Economic indicator data contains non-month-start dates.")
    return clean


def load_model_dataset() -> pd.dataFrame:
    arrivals = load_arrivals()
    if not RAW_ECONOMIC_INDICATORS_PATH.exists():
        raise FileNotFoundError(f"Raw economic indicators file not found: {RAW_ECONOMIC_INDICATORS_PATH}")
    economic = validate_economic_indicators(
        pd.read_csv(RAW_ECONOMIC_INDICATORS_PATH, parse_dates=["Date"])
    )

    expected_dates = pd.DatetimeIndex(arrivals["Date"])
    if not pd.DatetimeIndex(economic["Date"]).equals(expected_dates):
        raise ValueError(
            "Arrival and economic indicator files must cover the same consecutive "
            "months exactly. Update both source files through the same final month."
        )

    return arrivals.merge(
        economic,
        on="Date",
        how="inner",
        validate="one_to_one",
    )

def load_post_period_arrivals() -> pd.DataFrame: 
    if not POST_PERIOD_SOURCE.is_file():
        return pd.DataFrame(columns=["Date", "Total_Arrivals"])
    data = pd.read_csv(POST_PERIOD_SOURCE)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Total_Arrivals"] = pd.to_numeric(data["Total_Arrivals"], errors="coerce")
    if data[["Date", "Total_Arrivals"]].isna().any().any():
        raise ValueError(f"Post-period data contains invalid values: {POST_PERIOD_SOURCE}")
    if data["Date"].duplicated().any():
        raise ValueError(f"Post-period data contains duplicate dates: {POST_PERIOD_SOURCE}")
    post_period = data.loc[data["Date"] > "2025-12-01", ["Date", "Total_Arrivals"]]
    return post_period.sort_values("Date").reset_index(drop=True)

def build_model_dataset() -> pd.DataFrame:
    data = load_model_dataset()
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA, index=False)
    print(
        f"Validated {len(data)} monthly rows with tourist arrivals, and economic indicators  "
        f"from {data['Date'].min().strftime('%Y-%m')} to {data['Date'].max().strftime('%Y-%m')}.  "
    )
    return data

