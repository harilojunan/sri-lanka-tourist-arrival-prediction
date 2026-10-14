from pathlib import Path

import pandas as pd

from src.data_loader import build_model_dataset, load_post_period_arrivals
from src.evaluation import evaluate_models, save_results
from src.features import build_supervised_features, make_next_month_features
from src.models import tune_random_forest
from src.preprocessing import chronological_split, validate_monthly_data
from src.reporting import create_report

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

def evaluate_holdout(
    supervised: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train, test = chronological_split(supervised, test_months=18)
    comparison, predictions, _ = evaluate_models(train, test)
    return comparison, predictions


def forecast_next_month(
    monthly: pd.DataFrame, supervised: pd.DataFrame
) -> tuple[pd.Timestamp, float]:
    
    final_model = tune_random_forest(train=supervised)
    history = validate_monthly_data(monthly)
    next_date = history["Date"].max() + pd.offsets.MonthBegin(1)
    forecast_row = make_next_month_features(history, next_date)
    forecast = float(final_model.predict(forecast_row)[0])
    return next_date, forecast


def get_post_period_actual(
    forecast_date: pd.Timestamp,
) -> float | None:
    
    post_period = load_post_period_arrivals()
    actuals = post_period.loc[
        post_period["Date"].eq(forecast_date), "Total_Arrivals"
    ]
    return float(actuals.iloc[0]) if not actuals.empty else None


def print_run_summary(
    comparison: pd.DataFrame,
    forecast_date: pd.Timestamp,
    forecast: float,
    actual: float | None,
) -> None:
    
    print("\nModel comparison (same chronological holdout):")
    print(comparison.round(3).to_string(index=False))
    print(f"\nNext-month forecast: {forecast_date:%Y-%m} = {forecast:,.0f} arrivals")
    if actual is not None:
        print(
            f"Post-period actual: {actual:,.0f}; "
            f"absolute error: {abs(actual - forecast):,.0f}"
        )
    print(f"Results and report saved under: {RESULTS_DIR}")


def run() -> None:
    
    monthly = build_model_dataset()
    supervised = build_supervised_features(monthly)
    comparison, predictions = evaluate_holdout(supervised)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(comparison, predictions, RESULTS_DIR)

    forecast_date, forecast = forecast_next_month(monthly, supervised)
    actual = get_post_period_actual(forecast_date)
    save_post_period_check(forecast_date, forecast, actual, RESULTS_DIR)
    create_report(
        comparison=comparison,
        predictions=predictions,
        forecast_date=forecast_date,
        forecast=forecast,
        post_period_actual=actual,
        project_root=PROJECT_ROOT,
    )
    print_run_summary(comparison, forecast_date, forecast, actual)


def save_post_period_check(
    forecast_date: pd.Timestamp,
    prediction: float,
    actual: float | None,
    output_dir: Path,
) -> None:
    
    row = {
        "Forecast_Date": forecast_date.strftime("%Y-%m-%d"),
        "Prediction": prediction,
        "Actual_Observed": actual,
        "Absolute_Error": abs(actual - prediction) if actual is not None else None,
        "Note": (
            "Forecast only; the target-month actual is not in the supplied history."
            if actual is None
            else "Actual is shown for comparison and is excluded from model fitting."
        ),
    }
    path = output_dir / "predictions" / "post_period_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    pd.DataFrame([row]).to_csv(
        output_dir / "predictions" / "next_month_prediction.csv", index=False
    )


if __name__ == "__main__":
    run()