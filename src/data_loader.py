import numpy as np
import pandas as pd

from src.preprocessing import validate_monthly_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ECONOMIC_INDICATORS_PATH = (PROJECT_ROOT / "data" / "raw" / "cbsl_sources" / "cbsl_monthy.csv")
# dev01 code
PROCESSED_DATA = (PROJECT_ROOT / "data" / "processed" / "tourism_model_data.csv")
ECONOMIC_INDICATORS_COLUMNS = ["USD_LKR", "Inflation"]

def _validate_economic_indicators(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", *ECONOMIC_INDICATORS_COLUMNS}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Economic indicators file has missing columns: {sorted(missing_columns)}")

    clean = data.loc[:, ["Date", *ECONOMIC_INDICATORS_COLUMNS]].copy()
    



