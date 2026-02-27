# ==========================================================
#                    MAIN ENTRY POINT
# ==========================================================

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score

# Import custom modules
from config import MODEL_FILE
from preprocessing import detect_columns, build_pipeline
from model_selection import get_models
from evaluation import evaluate_model
from inference import run_inference


def train(csv_path, target_col):

    # ======================================================
    # LOAD DATA
    # ======================================================

    df_full = pd.read_csv(csv_path)

    if target_col not in df_full.columns:
        raise ValueError("Target column not found")

    print(f"\nDataset Loaded: {df_full.shape[0]} rows")

    # ======================================================
    # DATA MODE SELECTION
    # ======================================================

    print("\nChoose Data Mode:")
    print("1. Create 80/20 train-test CSV files")
    print("2. Train directly (internal split only)")

    data_mode = input("Enter option: ")

    if data_mode == "1":

        train_df, test_df = train_test_split(
            df_full,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        train_df.to_csv("train_data.csv", index=False)
        test_df.to_csv("test_data.csv", index=False)

        print("\nCreated:")
        print("train_data.csv (80%)")
        print("test_data.csv (20%)")

        df_train = train_df
        df_test = test_df

    elif data_mode == "2":

        df_train, df_test = train_test_split(
            df_full,
            test_size=0.2,
            random_state=42
        )

    else:
        print("Invalid option.")
        return

    # ======================================================
    # SPLIT FEATURES & TARGET
    # ======================================================

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # ======================================================
    # TASK TYPE DETECTION
    # ======================================================

    if y_train.dtype in ["int64", "float64"] and y_train.nunique() > 20:
        task_type = "regression"
        scoring_metric = "r2"
        print("\nDetected Task Type: REGRESSION")
    else:
        task_type = "classification"
        scoring_metric = "accuracy"
        print("\nDetected Task Type: CLASSIFICATION")
        print("\nClass Distribution:")
        print(y_train.value_counts())

    # ======================================================
    # BUILD PREPROCESSING PIPELINE
    # ======================================================

    num_cols, cat_cols = detect_columns(df_train, target_col)
    preprocessing_pipeline = build_pipeline(num_cols, cat_cols)

    # ======================================================
    # LOAD MODELS
    # ======================================================

    models = get_models(task_type)

    print("\nSelect Training Mode:")
    print("1. Auto Select Best Model (5-Fold CV)")
    print("2. Manually Choose Model")

    mode = input("Enter option: ")

    # ======================================================
    # AUTO MODEL SELECTION
    # ======================================================

    if mode == "1":

        best_model_name = None
        best_score = float("-inf")

        print("\nCross Validation Results (5-Fold):\n")

        for name, model in models.items():

            full_pipeline = Pipeline([
                ("preprocessing", preprocessing_pipeline),
                ("model", model)
            ])

            cv_scores = cross_val_score(
                full_pipeline,
                X_train,
                y_train,
                cv=5,
                scoring=scoring_metric
            )

            avg_score = np.mean(cv_scores)

            print(f"{name} → CV Mean {scoring_metric.upper()}: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_model_name = name

        print(f"\nBest Model (CV Selected): {best_model_name}")
        final_model = models[best_model_name]

    # ======================================================
    # MANUAL MODEL SELECTION
    # ======================================================

    elif mode == "2":

        print("\nAvailable Models:")
        for i, name in enumerate(models.keys(), start=1):
            print(f"{i}. {name}")

        selection = int(input("Select model number: "))
        model_names = list(models.keys())

        if selection < 1 or selection > len(model_names):
            print("Invalid selection.")
            return

        chosen_model_name = model_names[selection - 1]
        final_model = models[chosen_model_name]

        print(f"\nYou Selected: {chosen_model_name}")

    else:
        print("Invalid mode selected.")
        return

    # ======================================================
    # HYPERPARAMETER TUNING
    # ======================================================

    print("\nEnable Hyperparameter Tuning? (Y/N)")
    tuning_choice = input("Enter option: ").lower()

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("model", final_model)
    ])

    if tuning_choice == "y":

        print("\nRunning GridSearchCV...\n")

        param_grid = {}
        model_name = final_model.__class__.__name__

        # ---------------- XGBoost Regression ----------------
        if model_name == "XGBRegressor":

            param_grid = {
                "model__n_estimators": [150, 250],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__gamma": [0, 0.1],
                "model__reg_lambda": [1, 3]
            }

        # ---------------- XGBoost Classification ----------------
        elif model_name == "XGBClassifier":

            param_grid = {
                "model__n_estimators": [150, 250],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__gamma": [0, 0.1],
                "model__reg_lambda": [1, 3]
            }

        # ---------------- RandomForest ----------------
        elif "RandomForest" in model_name:

            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20]
            }

        if param_grid:

            grid_search = GridSearchCV(
                full_pipeline,
                param_grid,
                cv=5,
                scoring=scoring_metric,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            full_pipeline = grid_search.best_estimator_

            print("\nBest Parameters Found:")
            print(grid_search.best_params_)

        else:
            print("No tuning grid defined for this model. Skipping.")
            full_pipeline.fit(X_train, y_train)

    else:
        full_pipeline.fit(X_train, y_train)

    # ======================================================
    # OVERFITTING CHECK
    # ======================================================

    print("\nChecking for Overfitting...\n")

    train_preds = full_pipeline.predict(X_train)
    test_preds = full_pipeline.predict(X_test)

    if task_type == "regression":
        train_score = r2_score(y_train, train_preds)
        test_score = r2_score(y_test, test_preds)
        print(f"Train R2: {train_score:.4f}")
        print(f"Test  R2: {test_score:.4f}")
    else:
        train_score = accuracy_score(y_train, train_preds)
        test_score = accuracy_score(y_test, test_preds)
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test  Accuracy: {test_score:.4f}")

    gap = train_score - test_score
    print(f"Generalization Gap: {gap:.4f}")

    if gap > 0.10:
        print("⚠ Warning: Possible Overfitting Detected!")
    else:
        print("Model Generalization Looks Good.")

    # ======================================================
    # FINAL EVALUATION
    # ======================================================

    print("\nFinal Evaluation on Test Set:\n")
    evaluate_model(task_type, y_test, test_preds)

    # ======================================================
    # SAVE MODEL
    # ======================================================

    joblib.dump(full_pipeline, MODEL_FILE)

    print("\nFull pipeline trained and saved successfully.")


# ==========================================================
# CLI MENU
# ==========================================================

if __name__ == "__main__":

    print("1. Train Model")
    print("2. Run Inference")

    choice = input("Select option: ")

    if choice == "1":
        csv_path = input("Enter CSV path: ")
        target_col = input("Enter target column: ")
        train(csv_path, target_col)

    elif choice == "2":
        csv_path = input("Enter inference CSV path: ")
        run_inference(csv_path)

    else:
        print("Invalid option.")