import re
import time
from io import StringIO
from pathlib import Path
import warnings

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "data/raw/sltlda_monthly_arrivals_2015_2026.csv"
MAX_DOWNLOAD_ATTEMPTS = 3
BASE_URLS = {
    2015: "https://www.sltda.gov.lk/en/monthly-statistical-bulletins-2015",
    2016: "https://www.sltda.gov.lk/en/monthly-statistical-bulletins-2016",
    2017: "https://www.sltda.gov.lk/en/monthly-statistical-bulletins-2017",
    **{
        year: f"https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-{year}"
        for year in range(2018, 2027)
    },
}
MONTHS = {
    month.lower(): number
    for number, month in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}

def clean_number(value) -> float:
    text = str(value).replace(",", "").replace("\xa0", " ").strip()
    if text in {"", "-", "---", "nan", "NaN"}:
        return float("nan")
    text = re.sub(r"[^0-9.-]", "", text)
    return float(text) if text else float("nan")

def _download_year_page(year: int) -> str:
    url = BASE_URLS[year]
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            response = requests.get(
                url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            return response.text
        except (requests.ConnectionError, requests.Timeout) as error:
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"Could not connect to the SLTDA page for {year}: {url}. "
                    "Check your internet connection and DNS settings, then retry. "
                    "The existing local CSV is not changed when a download fails."
                ) from error
            time.sleep(attempt)
            
def _find_monthly_table(page: str, year: int) -> pd.DataFrame:
    tables = pd.read_html(StringIO(page))
    table = next(
        (
            candidate
            for candidate in tables
            if {"January", "December"}.issubset(
                set(candidate.astype(str).to_numpy().ravel())
            )
        ),
        None,
    )
    if table is None:
        raise RuntimeError(f"No January-December arrivals table found for {year}.")
    return table

def _empty_year(year: int, reason: str, end_month: int = 12) -> pd.DataFrame:
    warnings.warn(f"{year}: {reason} Assuming zero arrivals.", stacklevel=2)
    return pd.DataFrame(
        {
            "Date": pd.date_range(f"{year}-01-01", periods=end_month, freq="MS"),
            "Total_Arrivals": 0.0,
        }
    )

def _find_month_and_value_columns(
    table: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, str, str | None]:
    table = table.copy()
    table.columns = [str(column) for column in table.columns]
    if not any("month" in column.lower() for column in table.columns):
        first_row = table.iloc[0].astype(str).str.strip()
        if first_row.str.lower().eq("month").any():
            table.columns = first_row.tolist()
            table.drop(index=table.index[0], inplace=True)

    month_column = next(
        (column for column in table.columns if "month" in column.lower()),
        table.columns[0],
    )
    value_column = next(
        (column for column in table.columns if str(year) in column), None
    )
    return table, month_column, value_column

def _extract_monthly_rows(
    table: pd.DataFrame,
    year: int,
    month_column: str,
    value_column: str,
    end_month: int = 12,
) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        month = str(row[month_column]).strip().lower()
        if month in MONTHS:
            value = (
                clean_number(row[value_column])
                if value_column is not None
                else float("nan")
            )
            rows.append(
                (pd.Timestamp(year, MONTHS[month], 1), value)
            )
    expected_dates = pd.date_range(f"{year}-01-01", periods=end_month, freq="MS")
    result = (
        pd.DataFrame(rows, columns=["Date", "Total_Arrivals"])
        .drop_duplicates(subset="Date", keep="last")
        .set_index("Date")
        .reindex(expected_dates)
        .rename_axis("Date")
        .reset_index()
    )
    missing_count = int(result["Total_Arrivals"].isna().sum())
    if missing_count:
        result["Total_Arrivals"] = result["Total_Arrivals"].fillna(0)
        warnings.warn(
            f"{year}: {missing_count} month(s) have blank, null, or unavailable "
            "arrivals; assuming zero.",
            stacklevel=2,
        )
    return result