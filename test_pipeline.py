import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_pipeline
from src.evaluation import regression_metrics
from src.features import build_supervised_features, make_next_month_features
from src.preprocessing import chronological_split


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2015-01-01", periods=108, freq="MS")
        self.monthly_data = pd.DataFrame(
            {
                "Date": dates,
                "Total_Arrivals": np.arange(100, 100 + len(dates), dtype=float),
                "USD_LKR": np.arange(130, 130 + len(dates), dtype=float),
                "Inflation": np.arange(2, 2 + len(dates), dtype=float),
            }
        )

    def test_build_supervised_features_uses_only_prior_months(self) -> None:
        features = build_supervised_features(self.monthly_data)

        first = features.iloc[0]
        self.assertEqual(first["Target_Date"], pd.Timestamp("2016-01-01"))
        self.assertEqual(
            first["Target_Arrivals"], self.monthly_data.loc[12, "Total_Arrivals"]
        )
        self.assertEqual(first["Lag_1"], self.monthly_data.loc[11, "Total_Arrivals"])
        self.assertEqual(first["Lag_3"], self.monthly_data.loc[9, "Total_Arrivals"])
        self.assertEqual(first["Lag_12"], self.monthly_data.loc[0, "Total_Arrivals"])
        self.assertEqual(
            first["Rolling_3"],
            self.monthly_data.loc[9:11, "Total_Arrivals"].mean(),
        )
        self.assertEqual(first["USD_LKR_Lag_1"], self.monthly_data.loc[11, "USD_LKR"])

    def test_chronological_split_holds_out_latest_months(self) -> None:
        data = pd.DataFrame(
            {
                "Target_Date": pd.date_range("2020-01-01", periods=30, freq="MS"),
                "Target_Arrivals": range(30),
            }
        )

        train, test = chronological_split(data, test_months=6)

        self.assertEqual(len(train), 24)
        self.assertEqual(len(test), 6)
        self.assertLess(train["Target_Date"].max(), test["Target_Date"].min())
        self.assertEqual(test["Target_Arrivals"].tolist(), list(range(24, 30)))

    def test_make_next_month_features_rejects_nonconsecutive_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Target date must be the month immediately after"
        ):
            make_next_month_features(
                self.monthly_data, pd.Timestamp("2024-02-01")
            )

    def test_regression_metrics_excludes_zero_actuals_from_mape(self) -> None:
        metrics = regression_metrics(
            actual=np.array([10.0, 20.0, 0.0]),
            predicted=np.array([12.0, 18.0, 5.0]),
        )

        self.assertAlmostEqual(metrics["MAE"], 3.0)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(11.0))
        self.assertAlmostEqual(metrics["MAPE"], 15.0)

    def test_forecast_next_month_uses_latest_history(self) -> None:
        monthly_data = self.monthly_data
        seen_features: list[pd.DataFrame] = []

        class StubModel:
            def predict(self, features: pd.DataFrame) -> list[float]:
                seen_features.append(features.copy())
                return [1234.5]

        with patch.object(run_pipeline, "tune_random_forest", return_value=StubModel()):
            forecast_date, forecast = run_pipeline.forecast_next_month(
                monthly_data, pd.DataFrame()
            )

        self.assertEqual(forecast_date, pd.Timestamp("2024-01-01"))
        self.assertEqual(forecast, 1234.5)
        self.assertEqual(len(seen_features[0]), 1)
        self.assertEqual(
            seen_features[0].iloc[0]["Lag_1"],
            monthly_data.iloc[-1]["Total_Arrivals"],
        )
        self.assertEqual(seen_features[0].iloc[0]["Month"], 1)

    def test_save_post_period_check_writes_both_prediction_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "results"
            run_pipeline.save_post_period_check(
                pd.Timestamp("2024-01-01"),
                prediction=1200.0,
                actual=1250.0,
                output_dir=output_dir,
            )

            check = pd.read_csv(
                output_dir / "predictions" / "post_period_check.csv"
            )
            prediction = pd.read_csv(
                output_dir / "predictions" / "next_month_prediction.csv"
            )
            pd.testing.assert_frame_equal(check, prediction)
            self.assertEqual(check.loc[0, "Forecast_Date"], "2024-01-01")
            self.assertEqual(check.loc[0, "Absolute_Error"], 50.0)

    def test_run_wires_pipeline_outputs_to_report_and_summary(self) -> None:
        monthly = pd.DataFrame({"Date": [pd.Timestamp("2023-12-01")]})
        supervised = pd.DataFrame({"Target_Date": [pd.Timestamp("2023-12-01")]})
        comparison = pd.DataFrame({"Model": ["Linear Regression"]})
        predictions = pd.DataFrame({"Predicted": [1200.0]})
        forecast_date = pd.Timestamp("2024-01-01")
        report_calls: dict[str, object] = {}
        summary_calls: list[tuple[object, ...]] = []

        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            results_dir = project_root / "results"
            with (
                patch.object(run_pipeline, "PROJECT_ROOT", project_root),
                patch.object(run_pipeline, "RESULTS_DIR", results_dir),
                patch.object(run_pipeline, "build_model_dataset", return_value=monthly),
                patch.object(
                    run_pipeline,
                    "build_supervised_features",
                    return_value=supervised,
                ),
                patch.object(
                    run_pipeline,
                    "evaluate_holdout",
                    return_value=(comparison, predictions),
                ),
                patch.object(run_pipeline, "save_results"),
                patch.object(
                    run_pipeline,
                    "forecast_next_month",
                    return_value=(forecast_date, 1200.0),
                ),
                patch.object(
                    run_pipeline, "get_post_period_actual", return_value=1250.0
                ),
                patch.object(
                    run_pipeline,
                    "create_report",
                    side_effect=lambda **kwargs: report_calls.update(kwargs),
                ),
                patch.object(
                    run_pipeline,
                    "print_run_summary",
                    side_effect=lambda *args: summary_calls.append(args),
                ),
            ):
                run_pipeline.run()

            self.assertIs(report_calls["comparison"], comparison)
            self.assertIs(report_calls["predictions"], predictions)
            self.assertEqual(report_calls["forecast_date"], forecast_date)
            self.assertEqual(report_calls["forecast"], 1200.0)
            self.assertEqual(report_calls["post_period_actual"], 1250.0)
            self.assertEqual(report_calls["project_root"], project_root)
            self.assertEqual(len(summary_calls), 1)
            post_period_check = pd.read_csv(
                results_dir / "predictions" / "post_period_check.csv"
            )
            self.assertEqual(post_period_check.loc[0, "Absolute_Error"], 50.0)


if __name__ == "__main__":
    unittest.main()
