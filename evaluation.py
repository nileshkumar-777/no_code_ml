# ==========================================================
#                EVALUATION MODULE
# ==========================================================

# -------- Classification Metrics --------
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# -------- Regression Metrics --------
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np   # Used for square root (RMSE)


def evaluate_model(task_type, y_test, preds):
    """
    Evaluate model performance based on task type.

    Parameters:
        task_type (str): "classification" or "regression"
        y_test: true values
        preds: predicted values

    Returns:
        score (float): Used for model comparison
                       (Higher is better for both tasks)
    """

    # ======================================================
    # CLASSIFICATION EVALUATION
    # ======================================================
    if task_type == "classification":

        # Calculate accuracy
        acc = accuracy_score(y_test, preds)
        print(f"\nAccuracy: {acc:.4f}")

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        # Calculate weighted F1 score
        weighted_f1 = f1_score(y_test, preds, average="weighted")
        print(f"Weighted F1 Score: {weighted_f1:.4f}")

        # Return accuracy for model comparison (higher is better)
        return acc

    # ======================================================
    # REGRESSION EVALUATION
    # ======================================================
    elif task_type == "regression":

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"\nRMSE: {rmse:.4f}")

        # Calculate R² score
        r2 = r2_score(y_test, preds)
        print(f"R2 Score: {r2:.4f}")

        # Return R² for model comparison (higher is better)
        return r2

    else:
        raise ValueError("Invalid task type. Must be 'classification' or 'regression'.")