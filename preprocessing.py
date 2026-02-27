# ==========================================================
#                PREPROCESSING MODULE
# ==========================================================

# Import required preprocessing tools
from sklearn.pipeline import Pipeline                 # Allows chaining preprocessing steps
from sklearn.compose import ColumnTransformer         # Applies different transformations to different columns
from sklearn.impute import SimpleImputer              # Handles missing values
from sklearn.preprocessing import StandardScaler      # Scales numeric data
from sklearn.preprocessing import OneHotEncoder       # Encodes categorical data


def detect_columns(df, target_col):
    """
    Detect numeric and categorical feature columns automatically.
    """

    # Remove target column to isolate feature columns
    X = df.drop(columns=[target_col])

    # Detect numeric columns (int and float types)
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Detect categorical columns (object or category type)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Return both lists
    return num_cols, cat_cols


def build_pipeline(num_cols, cat_cols):
    """
    Build preprocessing pipeline for numeric and categorical features.
    """

    # ---------------- Numeric Pipeline ----------------
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # Replace missing numeric values with median
        ("scaler", StandardScaler())                    # Scale numeric values
    ])

    # ---------------- Categorical Pipeline ----------------
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Replace missing categorical values
        ("onehot", OneHotEncoder(handle_unknown="ignore"))     # Convert categories to numeric columns
    ])

    # Combine numeric and categorical pipelines
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols),  # Apply numeric pipeline
        ("cat", cat_pipeline, cat_cols)   # Apply categorical pipeline
    ])

    # Return complete preprocessing pipeline
    return full_pipeline