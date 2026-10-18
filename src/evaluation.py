from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import FEATURE_COLUMNS
from src.models import make_baseline_models, tune_random_forest


def regression_metrics(actual, predicted) -> dict[str, float]:
    # Compute common regression metrics, excluding zero actuals from MAPE.
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    # Percentage error is undefined for zero actual values.
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE" : float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2" : float(r2_score(actual, predicted)),
        "MAPE": mape,
    }

def evaluate_one_model(
    name: str,
    estimator: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    already_fitted: bool = False,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    # Fit when needed, score a model on holdout rows, and retain its residuals.
    if not already_fitted:
        estimator.fit(train[FEATURE_COLUMNS], train["Target_Arrivals"])

    actual = test["Target_Arrivals"].to_numpy()
    predicted = estimator.predict(test[FEATURE_COLUMNS])
    metrics = {"Model": name, **regression_metrics(actual, predicted)}
    predictions = pd.DataFrame(
        {
            "Target_Date": test["Target_Date"].values,
            "Actual": actual,
            "Predicted": predicted,
            "Residual": actual - predicted,
            "Model": name,
        }
    )
    return metrics, predictions

def evaluate_models(
        train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    # Compare baseline and tuned pipelines on the same chronological holdout.
    models = make_baseline_models()
    # Hyperparameter selection sees training rows only, never the holdout.
    search = tune_random_forest(train)
    models["Tuned Random Forest"] = search.best_estimator_

    result_rows = []
    prediction_frames = []
    fitted_models: dict[str, Pipeline] = {}
    for name, estimator in models.items(): 
        metrics, model_predictions = evaluate_one_model(
            name,
            estimator, 
            train,
            test,
            already_fitted=name == "Tuned Random Forest"
        )
        result_rows.append(metrics)
        prediction_frames.append(model_predictions)
        fitted_models[name] = estimator

    comparison = pd.DataFrame(result_rows).sort_values("MAE").reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return comparison, predictions, fitted_models


def make_result_directories(output_dir: Path) -> tuple[Path, Path, Path]:
    # Create and return the three folders used for evaluation artifacts.
    tables = output_dir / "tables"
    prediction_dir = output_dir / "predictions"
    figures = output_dir / "figures"
    for folder in (tables, prediction_dir, figures):
        folder.mkdir(parents=True, exist_ok=True)
    return tables, prediction_dir, figures


def save_result_tables(
        comparison: pd.DataFrame,
        predictions: pd.DataFrame,
        tables_dir: Path,
        predictions_dir: Path
) -> None:
    # Write metric, holdout-prediction, and largest-error CSV tables.
    comparison.to_csv(
        tables_dir / "model_comparison.csv", index=False, float_format="%.3f"
    )
    predictions.to_csv(predictions_dir / "test_predictions.csv", index=False)
    errors  = (
        predictions.assign(Absolute_Error=lambda frame: frame["Residual"].abs())
        .sort_values(["Model", "Absolute_Error"], ascending=[True, False])
    )
    errors.to_csv(tables_dir / "error_analysis.csv", index=False)
    print(f"Saved model comparison table to {tables_dir / 'model_comparison.csv'}")


def best_model_predictions(
        comparison: pd.DataFrame, 
        predictions: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    # Select holdout predictions for the model with the lowest MAE.
    best_model = comparison.iloc[0]["Model"]
    best = predictions.loc[predictions["Model"].eq(best_model)].copy()
    best["Target_Date"] = pd.to_datetime(best["Target_Date"])
    return best_model, best


def save_actual_vs_predicted_plot(best_model: str, best: pd.DataFrame, path: Path) -> None:
    # Save actual and predicted arrivals over the holdout timeline.
    plt.figure(figsize=(10, 5))
    plt.plot(best["Target_Date"], best["Actual"], marker="o", label="Actual")
    plt.plot(best["Target_Date"], best["Predicted"], marker="o", label="Predicted")
    plt.title(f"Chronological holdout: {best_model}")
    plt.xlabel("Target month")
    plt.ylabel("Tourist arrivals")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def save_residual_plot(best_model: str, best: pd.DataFrame, path: Path) -> None:
    # Save holdout residuals against the zero-error reference line.
    plt.figure(figsize=(10,4))
    plt.axhline(0, color="black", linewidth=1)
    plt.scatter(best["Target_Date"], best["Residual"])
    plt.title(f"Holdout residuals: {best_model}")
    plt.xlabel("Target month")
    plt.ylabel("Actual - predicted arrivals")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def save_results(
        comparison: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path
) -> None:
    # Persist evaluation tables and diagnostic plots for the best model.
    tables_dir, prediction_dir, figures = make_result_directories(output_dir)
    save_result_tables(comparison, predictions, tables_dir, prediction_dir)

    best_model, best_predictions = best_model_predictions(comparison, predictions)
    save_actual_vs_predicted_plot(
        best_model, best_predictions, figures / "actual_vs_predicted.png"
    )
    save_residual_plot(best_model, best_predictions, figures / "residuals.png")
