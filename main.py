# ==========================================================
#                    MAIN ENTRY POINT
# ==========================================================

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier, VotingRegressor

from explainability import generate_shap_explanations

from config import MODEL_FILE
from preprocessing import detect_columns, build_pipeline
from model_selection import get_models
from evaluation import evaluate_model
from inference import run_inference


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def export_feature_importance(trained_pipeline):

    try:
        model = trained_pipeline.named_steps["model"]
        preprocessing = trained_pipeline.named_steps["preprocessing"]

        feature_names = preprocessing.get_feature_names_out()

        if hasattr(model, "feature_importances_"):

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            importance_df.to_csv("feature_importance.csv", index=False)

            print("\nTop 10 Important Features:")
            print(importance_df.head(10))

        else:
            print("\nFeature importance not supported.")

    except Exception as e:
        print(f"\nFeature importance failed: {e}")


# ==========================================================
# RUN CROSS VALIDATION
# ==========================================================

def run_cross_validation(models, preprocessing, X, y, task_type, scoring):

    scores_dict = {}

    for name, model in models.items():

        temp_pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("model", model)
        ])

        cv_strategy = (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            if task_type == "classification"
            else 5
        )

        scores = cross_val_score(
            temp_pipeline,
            X,
            y,
            cv=cv_strategy,
            scoring=scoring
        )

        mean_score = np.mean(scores)
        scores_dict[name] = mean_score

        print(f"{name} → CV Mean: {mean_score:.4f}")

    return scores_dict


# ==========================================================
# BUILD TOP 3 ENSEMBLE
# ==========================================================

def build_top3_ensemble(models, scores_dict, task_type):

    sorted_models = sorted(
        scores_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top3 = sorted_models[:3]

    print("\nTop 3 Models Selected for Ensemble:")
    for name, score in top3:
        print(f"{name} ({score:.4f})")

    estimators = [(name, models[name]) for name, _ in top3]

    if task_type == "classification":
        ensemble_model = VotingClassifier(
            estimators=estimators,
            voting="soft"
        )
    else:
        ensemble_model = VotingRegressor(
            estimators=estimators
        )

    return ensemble_model


# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train(csv_path, target_col):

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError("Target column not found")

    print(f"\nDataset Loaded: {df.shape[0]} rows")

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # ---------------- TASK TYPE ----------------

    if y_train.dtype in ["int64", "float64"] and y_train.nunique() > 20:
        task_type = "regression"
        scoring_metric = "r2"
    else:
        task_type = "classification"
        scoring_metric = "accuracy"

    print(f"\nDetected Task Type: {task_type.upper()}")

    # ---------------- PREPROCESSING ----------------

    num_cols, cat_cols = detect_columns(df_train, target_col)
    preprocessing_pipeline = build_pipeline(num_cols, cat_cols)

    models = get_models(task_type)

    # ---------------- TRAINING MODE ----------------

    print("\nSelect Training Mode:")
    print("1. Auto Select Best Model")
    print("2. Manual Model")
    print("3. Auto Ensemble (Top 3)")

    mode = input("Enter option: ")

    final_model = None

    # ---------------- AUTO BEST ----------------

    if mode == "1":

        scores_dict = run_cross_validation(
            models,
            preprocessing_pipeline,
            X_train,
            y_train,
            task_type,
            scoring_metric
        )

        best_model_name = max(scores_dict, key=scores_dict.get)
        final_model = models[best_model_name]

        print(f"\nBest Model Selected: {best_model_name}")

    # ---------------- MANUAL ----------------

    elif mode == "2":

        model_list = list(models.keys())

        for i, name in enumerate(model_list, 1):
            print(f"{i}. {name}")

        selection = int(input("Select model number: "))
        final_model = models[model_list[selection - 1]]

    # ---------------- AUTO ENSEMBLE ----------------

    elif mode == "3":

        scores_dict = run_cross_validation(
            models,
            preprocessing_pipeline,
            X_train,
            y_train,
            task_type,
            scoring_metric
        )

        final_model = build_top3_ensemble(
            models,
            scores_dict,
            task_type
        )

    else:
        print("Invalid option.")
        return

    # ---------------- FINAL PIPELINE ----------------

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("model", final_model)
    ])

    full_pipeline.fit(X_train, y_train)

    # ---------------- EVALUATION ----------------

    train_preds = full_pipeline.predict(X_train)
    test_preds = full_pipeline.predict(X_test)

    if task_type == "regression":
        train_score = r2_score(y_train, train_preds)
        test_score = r2_score(y_test, test_preds)
    else:
        train_score = accuracy_score(y_train, train_preds)
        test_score = accuracy_score(y_test, test_preds)

    print(f"\nTrain Score: {train_score:.4f}")
    print(f"Test Score:  {test_score:.4f}")
    print(f"Gap: {train_score - test_score:.4f}")

    print("\nFinal Evaluation:")
    evaluate_model(task_type, y_test, test_preds)

    # ---------------- FEATURE IMPORTANCE ----------------

    export_feature_importance(full_pipeline)

    # ---------------- SHAP ----------------

    print("\nGenerate SHAP Explainability? (Y/N)")
    shap_choice = input().lower()

    if shap_choice == "y":
        sample_data = X_test.sample(min(200, len(X_test)), random_state=42)
        generate_shap_explanations(full_pipeline, sample_data, task_type)

    # ---------------- SAVE MODEL ----------------

    joblib.dump(full_pipeline, MODEL_FILE)
    print("\nModel saved successfully.")


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    print("1. Train Model")
    print("2. Run Inference")

    choice = input()

    if choice == "1":
        csv_path = input("Enter CSV path: ")
        target_col = input("Enter target column: ")
        train(csv_path, target_col)

    elif choice == "2":
        csv_path = input("Enter inference CSV path: ")
        run_inference(csv_path)

    else:
        print("Invalid option.")