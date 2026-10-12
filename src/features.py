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
    
