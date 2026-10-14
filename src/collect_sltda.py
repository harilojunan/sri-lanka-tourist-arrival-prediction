import re
import time
from pathlib import Path

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