# Monthly Tourist Arrival Forecasting in Sri Lanka

**Assignment report draft - replace the clearly marked team/submission placeholders before submission.**

## 1. Introduction and problem definition

Tourism is a major part of Sri Lanka's economy. This project uses supervised regression to predict the total international tourist arrivals for the next month. Each prediction uses only the calendar and historical arrival information available before the target month. The task is classical machine learning; no deep-learning methods are used.

## 2. Dataset and sources

The primary dataset is the Sri Lanka Tourism Development Authority (SLTDA) monthly arrivals series. The model period is 2015-01 through 2026-09: 141 monthly observations, one row per month. The response is `Total_Arrivals`, the reported monthly count of international tourist arrivals. The joined modelling data also contain the supplied `USD_LKR` and `Inflation` indicators.

Official sources:
- SLTDA monthly reports: https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports
- SLTDA all-country reports (optional extension): https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries
- CBSL monthly indicators (candidate source): https://www.cbsl.gov.lk/statistics/economic-indicators/monthly-indicators

The economic file is supplied as a CBSL candidate and its month rows are joined to arrivals. Its exact official table/series names, exchange-rate definition, inflation measure, units and publication timing are not recorded with the file and have not been independently verified. The model uses only the preceding month's values, not target-month macro values; this reduces direct look-ahead but does not by itself prove publication-time availability.

## 3. Exploratory data analysis

The validated series has a visible structural interruption and recovery around the COVID period, as well as differences in annual totals and seasonal month averages. Annual totals peak in 2025 (2,362,521) and are lowest in 2021 (194,495); the highest average month is December and the lowest is May. The notebook produces the time-series, annual-total, and month-of-year plots so that these patterns can be checked visually. COVID-era observations are retained, not treated as erroneous outliers.

## 4. Data preprocessing and feature engineering

The pipeline requires one consecutive non-negative numeric arrival value for every month from 2015-01 through 2026-09. It rejects invalid dates, duplicate months, missing/non-numeric arrivals, negative arrivals, non-month-start dates, and incomplete coverage. The collector replaces blank, unavailable, or omitted monthly values with assumed zeroes and emits a warning. The 2020 SLTDA report marks April-November as `---`, not numeric zeroes, so those replacements are assumptions and must be verified before interpreting model results. Economic rows must cover the same months, with valid numeric values and no duplicates.

The target is the arrivals count in the target month. For each target month, the predictors are:

| Predictor | Definition |
|---|---|
| Year, Month | Calendar year and month of the target month, known in advance. |
| Lag_1 | Arrivals in the immediately preceding month. |
| Lag_3 | Arrivals three months before the target month. |
| Lag_12 | Arrivals in the same calendar month one year earlier. |
| Rolling_3 | Mean arrivals in the three months immediately before the target month. |
| USD_LKR_Lag_1 | Supplied exchange-rate column from the preceding month. |
| Inflation_Lag_1 | Supplied inflation column from the preceding month. |
| COVID_Flag | Derived indicator for targets from 2020-03 through 2021-12. |
| Economic_Crisis_Flag | Derived indicator for targets in 2022; a documented modelling window, not a raw source field. |

The first 12 target rows lack a complete annual lag; the current validated supervised frame therefore has 129 rows. The last 18 targets are used for the chronological test set; earlier rows are used for fitting and time-series validation. Calendar and disruption features are aligned to the target month, while arrivals and economic indicators end before the target month. The current-month actual and same-month economic values are never included for the month being predicted.

## 5. Algorithms and rationale

**Linear Regression** is a transparent linear baseline. It estimates a weighted additive relationship between the engineered predictors and the next-month arrival count. Its simplicity makes it a useful reference, but it may underfit nonlinear patterns and interactions.

**Random Forest Regressor** combines decision trees trained on resampled data and averages their predictions. It can represent nonlinear relationships and feature interactions without requiring a linear response surface. It is less directly interpretable and may overfit a short monthly history.

Both models use the same chronological training/test rows and the same preprocessing pipeline. Month is one-hot encoded; numeric predictors are imputed/scaled within the model pipeline, so preprocessing is fitted on training data only.

## 6. Experimental design and metrics

The test set is the final 18 supervised target months in chronological order. No random shuffle is used. This is a rolling one-month-ahead evaluation: each later test target may use earlier observed test-period months as lagged inputs, but its target value is not used to fit the model. It is not a single fixed-origin 18-month forecast. Random Forest hyperparameters are selected by `TimeSeriesSplit` using training data only; the test set is used once for model comparison. The search varies tree count (200 or 500), maximum depth (8 or 12), and minimum leaf size (1 or 2).

- **MAE** is the average absolute error in arrivals and is easy to interpret in the target's units.
- **RMSE** is in arrivals and penalizes large errors more strongly.
- **R2** compares residual variance against a constant-mean prediction; it can be negative.
- **MAPE** is the average absolute percentage error. It is reported cautiously because it is undefined at zero actuals (the pipeline excludes zero denominators).

## 7. Results

The table below is generated by the latest pipeline run and compares all models on the same held-out months (2025-04 to 2026-09).

| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| Linear Regression | 17,169.70 | 23,489.21 | 0.7432 | 9.15% |
| Random Forest | 20,385.21 | 24,682.06 | 0.7164 | 11.25% |
| Tuned Random Forest | 22,805.95 | 27,063.30 | 0.6590 | 12.53% |

The lowest-MAE model on this holdout is Linear Regression; this is a descriptive comparison, not a claim that the test set was used to tune hyperparameters. Tuned Random Forest MAE improvement over the fixed-parameter forest (positive means lower error): -2,420.74 arrivals.

### Error analysis

The three largest absolute errors for the lowest-MAE model are:

| Target month | Actual | Predicted | Absolute error |
|---|---:|---:|---:|
| 2026-03 | 183,979 | 237,932 | 53,953 |
| 2026-01 | 277,327 | 230,030 | 47,297 |
| 2026-07 | 196,845 | 161,153 | 35,692 |

The report folder includes actual-versus-predicted and residual-over-time plots; the results folder includes per-model errors to support further discussion.

## 8. Next-month forecast

The forecast for October 2026 is made using arrival and economic data available through September 2026. A Random Forest is refitted on the full available supervised history, with hyperparameters selected by time-series cross-validation. The forecast is separate from the holdout comparison. No actual for this forecast month is present in the supplied arrival history, so forecast error is not reported.

## 9. Critical discussion, limitations, and future work

The linear baseline provides an interpretable reference; the Random Forest can model nonlinear patterns. The table should be interpreted using all four metrics and the error plots, not a single score. Any large errors during COVID recovery or other structural changes should be discussed rather than removed as outliers. A short monthly series limits generalization and makes tuned forests vulnerable to overfitting.

The supplied macro series are used as previous-month candidate features, but their exact official definitions and publication timing remain unverified. Before relying on them for operational forecasts, document the official CBSL table/series IDs, units, monthly aggregation and release timestamps, and verify each feature was actually available at the forecast cutoff. Other future work includes longer verified history, source-market features with consistent coverage and availability, prediction intervals, and additional classical models if allowed by the lecturer.

## 10. Individual contributions

Complete these with the group's true details; do not invent member identities or work history.

| Member | ID / name | Verified contribution | GitHub commits / presentation section |
|---|---|---|---|
| Member 1 | [Fill in] | [Fill in actual work] | [Fill in] |
| Member 2 | [Fill in] | [Fill in actual work] | [Fill in] |

## 11. Conclusion

This project establishes a reproducible, leakage-aware classical-ML model for next-month tourist arrivals using official SLTDA arrivals and supplied prior-month economic indicators. The economic file's exact series definitions and publication timing remain unverified and should be confirmed before operational use.

## References

1. Sri Lanka Tourism Development Authority, Monthly Tourist Arrivals Reports, https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports
2. Sri Lanka Tourism Development Authority, Tourist Arrivals from All Countries, https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries
3. Central Bank of Sri Lanka, Monthly Economic Indicators, https://www.cbsl.gov.lk/statistics/economic-indicators/monthly-indicators (candidate source; supplied series definitions and data-release timing require confirmation)
4. Fundamentals of Machine Learning - Programming Assignment and Assignment Marking Rubric, supplied course documents.

The complete Python implementation and executable notebook code are available in the project source files and notebooks.
