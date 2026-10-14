import pandas as pd

FEATURE_COLUMNS = [
    "Year",
    "Month",
    "Lag_1", # Arrivals in the previous month 
    "Lag_3", # Arrivals three months before the target
    "Lag_12", # Arrivals in the same calendar month one year earlier
    "Rolling_3", # Average arrivals in the three months immediately before the target
    "USD_LKR_Lag_1", # Previous months supplied exchange-rate value
    "Inflation_Lag_1", # Previous months supplied inflation value
    "COVID_Flag", # Indicator for target months from March 2020 through December 2021
    "Economic_Crisis_Flag", # Indicator for target months during 2022
]
    
def event_features(target_date: pd.Timestamp) -> dict[str, int]:
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2021-12-01")
    crisis_start = pd.Timestamp("2022-01-01")
    crisis_end = pd.Timestamp("2022-12-01")
    return {
        "COVID_Flag": int(covid_start <= target_date <= covid_end),
        "Economic_Crisis_Flag": int(crisis_start <= target_date <= crisis_end),
    }

def build_supervised_features(monthly_data: pd.DataFrame) -> pd.DataFrame:
    arrivals = monthly_data.set_index("Date")["Total_Arrivals"].sort_index()
    economic = monthly_data.set_index("Date")[["USD_LKR", "Inflation"]].sort_index()
    frame = pd.DataFrame(index=arrivals.index)
    frame.index.name = "Target_Date"
    frame["Year"] = frame.index.year
    frame["Month"] = frame.index.month
    frame["Lag_1"] = arrivals.shift(1)
    frame["Lag_3"] = arrivals.shift(3)
    frame["Lag_12"] = arrivals.shift(12)
    frame["Rolling_3"] = arrivals.shift(1).rolling(window=3, min_periods=3).mean()
    frame["USD_LKR_Lag_1"] = economic["USD_LKR"].shift(1)
    frame["Inflation_Lag_1"] = economic["Inflation"].shift(1)
    event_flags = frame.index.to_series().map(event_features)
    frame["COVID_Flag"] = event_flags.map(lambda flags: flags["COVID_Flag"])
    frame["Economic_Crisis_Flag"] = event_flags.map(
        lambda flags: flags["Economic_Crisis_Flag"]
    )
    frame["Target_Arrivals"] = arrivals
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["Target_Arrivals"])
    return frame.reset_index()

def make_next_month_features(
        monthly: pd.DataFrame, target_date: pd.Timestamp
) -> pd.DataFrame:
    if target_date not in monthly["Date"].values:
        raise ValueError(f"Target date {target_date} not found in monthly data.")
    target_date = pd.Timestamp(target_date).to_period("M").to_timestamp()
    arrivals = monthly.set_index("Date")["Total_Arrivals"].sort_index()
    economic = monthly.set_index("Date")[["USD_LKR", "Inflation"]].sort_index()
    prior_months = get_required_history_dates(target_date)
    validate_forecast_history(arrivals.index, prior_months, target_date)
    validate_economic_history(economic.index, prior_months[-1])

    values = {
        "Year": target_date.year,
        "Month": target_date.month,
        "Lag_1": arrivals.loc[prior_months[-1]],
        "Lag_3": arrivals.loc[prior_months[-3]],
        "Lag_12": arrivals.loc[prior_months[0]],
        "Rolling_3": arrivals.loc[prior_months[-3:]].mean(),
        "USD_LKR_Lag_1": economic.loc[prior_months[-1], "USD_LKR"],
        "Inflation_Lag_1": economic.loc[prior_months[-1], "Inflation"],
        **event_features(target_date),
    }
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)

def get_required_history_dates(target_date: pd.Timestamp) -> pd.DatetimeIndex:
    last_prior_month = target_date - pd.offsets.MonthBegin(1)
    return pd.date_range(
        end=last_prior_month, periods=12, freq="MS"
    )

def validate_forecast_history(
        available_dates: pd.DatetimeIndex, 
        required_dates: pd.DatetimeIndex, 
        target_date: pd.Timestamp,
) -> None:
    if not required_dates.isin(available_dates).all():
        raise ValueError(
            "Twelve consecutive historical months are required to forecast.")
    if available_dates.max() >= target_date:
        raise ValueError(
            "Forecast features must not include target-month observations.")


def validate_economic_history(
        available_dates: pd.DatetimeIndex, 
        latest_required_month: pd.Timestamp
) -> None:
    if latest_required_month not in available_dates:
        raise ValueError(
            "Economic indicator data must include the month immediately before the target month."
        )
