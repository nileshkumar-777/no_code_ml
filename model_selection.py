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


def get_models(task_type):
    """
    Returns dictionary of models depending on task type.

    Parameters:
        task_type (str): "classification" or "regression"

    Returns:
        dict: Dictionary of model_name -> model_object
    """

    # ======================================================
    # CLASSIFICATION MODELS
    # ======================================================
    if task_type == "classification":

        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "SVC": SVC(),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42)
        }

    # ======================================================
    # REGRESSION MODELS
    # ======================================================
    elif task_type == "regression":

        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42)
        }

    else:
        raise ValueError("Invalid task type. Must be 'classification' or 'regression'.")

    return models