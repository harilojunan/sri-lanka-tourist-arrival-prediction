from pathlib import Path
import re


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