import numpy as np
import pandas as pd

from src.preprocessing import validate_monthly_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ECONOMIC_INDICATORS_PATH = (PROJECT_ROOT / "data" / "raw" / "cbsl_sources" / "cbsl_monthy.csv")
# dev01 code
PROCESSED_DATA = (PROJECT_ROOT / "data" / "processed" / "tourism_model_data.csv")
ECONOMIC_INDICATORS_COLUMNS = ["USD_LKR", "Inflation"]

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
    if not np.infinite(clean[ECONOMIC_INDICATORS_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Economic indicator data contains infinite values.")
    if clean["Date"].duplicated().any():
        raise ValueError("Economic indicator data contains duplicate months.")

    clean = clean.sort_values("Date").reset_index(drop=True)
    month_starts = clean["Date"].dt.to_period("M").dt.to_timestamp()
    if not clean["Date"].equals(month_starts).all():
        raise ValueError("Economic indicator data contains non-month-start dates.")
    return clean


def load_model_dataset() -> pd.dataFrame:
    # dev01 code
    if not RAW_ECONOMIC_INDICATORS_PATH.exists():
        raise FileNotFoundError(f"Raw economic indicators file not found: {RAW_ECONOMIC_INDICATORS_PATH}")
    economic = validate_economic_indicators(
        pd.read_csv(RAW_ECONOMIC_INDICATORS_PATH, parse_dates=["Date"])
    )

    # dev01 code


# dev01 code

def build_model_dataset() -> pd.DataFrame:
    data = load_model_dataset()
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA, index=False)
    print(
        f"Validated {len(data)} monthly rows with tourist arrivals, and economic indicators  "
        f"from {data['Date'].min().strftime('%Y-%m')} to {data['Date'].max().strftime('%Y-%m')}.  "
    )
    return data



