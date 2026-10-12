from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
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

# dev01 code

def evaluate_models(
        train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    models = make_baseline_models()
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


