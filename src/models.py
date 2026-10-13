import pandas as pd
 
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FEATURE_COLUMNS

NUMERIC_FEATURES = [name for name in FEATURE_COLUMNS if name != "Month"]


def _make_numeric_preprocessor() -> Pipeline: 
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

def _make_preprocessor() -> ColumnTransformer: 
    return ColumnTransformer(
        [
            ("numeric", _make_numeric_preprocessor(), NUMERIC_FEATURES),
            ("month", OneHotEncoder(handle_unknown="ignore"), ["Month"]),
        ]
    )

def make_pipeline(estimator: BaseEstimator) -> Pipeline: 
    return Pipeline(
        [
            ("preprocess", _make_preprocessor()),
            ("model", estimator),
        ]
    )
    
def make_baseline_models() -> dict[str, Pipeline]: 
    return {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            )
        ),
    }