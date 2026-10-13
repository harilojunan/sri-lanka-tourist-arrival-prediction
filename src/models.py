import pandas as pd
 
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
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