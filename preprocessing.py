# ==========================================================
#                PREPROCESSING MODULE
# ==========================================================

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# ==========================================================
#            RARE CATEGORY GROUPING TRANSFORMER
# ==========================================================

class RareCategoryGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.frequent_categories_ = {}

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.frequent_categories_[col] = freq[freq >= self.threshold].index.tolist()

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        for col in X.columns:

            frequent = self.frequent_categories_.get(col, [])

            X[col] = X[col].apply(
                lambda x: x if x in frequent else "Other"
            )

        return X

    # 🔧 FIX: allow sklearn to propagate feature names
    def get_feature_names_out(self, input_features=None):
        return input_features


# ==========================================================
#              NUMERIC OUTLIER HANDLING TRANSFORMER
# ==========================================================

class OutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        for col in X.columns:

            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR

            self.lower_bounds_[col] = lower
            self.upper_bounds_[col] = upper

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        for col in X.columns:

            lower = self.lower_bounds_.get(col)
            upper = self.upper_bounds_.get(col)

            X[col] = np.clip(X[col], lower, upper)

        return X

    # 🔧 FIX: allow sklearn to propagate feature names
    def get_feature_names_out(self, input_features=None):
        return input_features


# ==========================================================
#           COLUMN TYPE DETECTION FUNCTION
# ==========================================================

def detect_columns(df, target_col):

    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return num_cols, cat_cols


# ==========================================================
#             BUILD PREPROCESSING PIPELINE
# ==========================================================

def build_pipeline(num_cols, cat_cols):

    # ---------------- Numeric Pipeline ----------------
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_cap", OutlierCapper()),
        ("scaler", StandardScaler())
    ])

    # ---------------- Categorical Pipeline ----------------
    cat_pipeline = Pipeline([

        ("rare_category", RareCategoryGrouper(threshold=0.01)),

        ("imputer", SimpleImputer(strategy="most_frequent")),

        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ---------------- Combine Pipelines ----------------
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return full_pipeline