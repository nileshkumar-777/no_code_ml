# ==========================================================
#                MODEL SELECTION MODULE
# ==========================================================

# ---------------- CLASSIFICATION MODELS ----------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------- REGRESSION MODELS ----------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# ---------------- XGBOOST MODELS ----------------

from xgboost import XGBClassifier
from xgboost import XGBRegressor


# ==========================================================
#                   MODEL SELECTION FUNCTION
# ==========================================================

def get_models(task_type):
    """
    Returns dictionary of models depending on task type.

    Parameters:
        task_type (str): "classification" or "regression"

    Returns:
        dict: model_name -> model_object
    """

    # ======================================================
    # CLASSIFICATION MODELS
    # ======================================================

    if task_type == "classification":

        models = {

            # Handles imbalance automatically
            "RandomForest": RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            ),

            # Handles imbalance
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

            # KNN does NOT support class_weight
            "KNN": KNeighborsClassifier(),

            # SVC with imbalance handling + probability support
            "SVC": SVC(
                probability=True,
                class_weight="balanced"
            ),

            # Handles imbalance
            "DecisionTree": DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            ),

            # GradientBoosting does NOT support class_weight
            "GradientBoosting": GradientBoostingClassifier(
                random_state=42
            ),

            # 🔥 XGBoost Classifier
            # scale_pos_weight helps with imbalance
            "XGBoost": XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        }

    # ======================================================
    # REGRESSION MODELS
    # ======================================================

    elif task_type == "regression":

        models = {

            "RandomForest": RandomForestRegressor(
                random_state=42
            ),

            "LinearRegression": LinearRegression(),

            "KNN": KNeighborsRegressor(),

            "SVR": SVR(),

            "DecisionTree": DecisionTreeRegressor(
                random_state=42
            ),

            "GradientBoosting": GradientBoostingRegressor(
                random_state=42
            ),

            # 🔥 XGBoost Regressor
            "XGBoost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        }

    else:
        raise ValueError("Invalid task type. Must be 'classification' or 'regression'.")

    return models