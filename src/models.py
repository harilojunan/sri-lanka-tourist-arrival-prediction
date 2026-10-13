import pandas as pd
 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS

NUMERIC_FEATURES = [name for name in FEATURE_COLUMNS if name != "Month"]


def _make_numeric_preprocessor() -> Pipeline: 
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
