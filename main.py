# ==========================================================
#                    MAIN ENTRY POINT
# ==========================================================

import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier, VotingRegressor

from explainability import generate_shap_explanations

from config import MODEL_DIR
from preprocessing import detect_columns, build_pipeline
from model_selection import get_models
from evaluation import evaluate_model
from inference import run_inference


# ==========================================================
# DATA VALIDATION LAYER
# ==========================================================

def validate_dataset(df, target_col):

    print("\n================ DATA VALIDATION REPORT ================\n")

    total_rows = df.shape[0]
    total_cols = df.shape[1]

    print(f"Total Rows: {total_rows}")
    print(f"Total Columns: {total_cols}")

    # 1️⃣ Missing Values
    missing_percent = df.isnull().mean() * 100
    high_missing = missing_percent[missing_percent > 20]

    if not high_missing.empty:
        print("\n⚠ Warning: Columns with >20% missing values:")
        print(high_missing.sort_values(ascending=False))
    else:
        print("\nNo major missing value issues detected.")

    # 2️⃣ Duplicate Rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\n⚠ Warning: {duplicate_count} duplicate rows found.")
    else:
        print("\nNo duplicate rows detected.")

    # 3️⃣ Constant Columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print("\n⚠ Warning: Constant columns detected:")
        print(constant_cols)
    else:
        print("\nNo constant columns detected.")

    # 4️⃣ Small Dataset Warning
    if total_rows < 200:
        print("\n⚠ Warning: Dataset is small. Risk of overfitting.")

    # 5️⃣ Class Imbalance (classification only)
    if df[target_col].nunique() < 20:
        class_distribution = df[target_col].value_counts(normalize=True) * 100
        if class_distribution.min() < 10:
            print("\n⚠ Warning: Class imbalance detected:")
            print(class_distribution.round(2))

    print("\n=========================================================\n")


# ==========================================================
# MODEL VERSIONING SYSTEM
# ==========================================================

def get_next_model_version(dataset_name):

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    existing_files = os.listdir(MODEL_DIR)
    versions = []

    for file in existing_files:
        if file.startswith(dataset_name + "_v") and file.endswith(".pkl"):
            try:
                version = int(file.split("_v")[-1].replace(".pkl", ""))
                versions.append(version)
            except:
                pass

    next_version = 1 if not versions else max(versions) + 1
    model_filename = f"{dataset_name}_v{next_version}.pkl"

    return os.path.join(MODEL_DIR, model_filename)


# ==========================================================
# LEADERBOARD SYSTEM
# ==========================================================

def update_leaderboard(
    dataset_name,
    model_name,
    training_mode,
    cv_mean,
    cv_std,
    train_score,
    test_score,
    gap
):

    leaderboard_file = "leaderboard.csv"

    if not os.path.exists(leaderboard_file):
        df = pd.DataFrame(columns=[
            "experiment_id",
            "timestamp",
            "dataset_name",
            "model_name",
            "training_mode",
            "cv_mean",
            "cv_std",
            "train_score",
            "test_score",
            "gap"
        ])
        df.to_csv(leaderboard_file, index=False)

    df = pd.read_csv(leaderboard_file)
    experiment_id = 1 if df.empty else df["experiment_id"].max() + 1

    new_entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "training_mode": training_mode,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "train_score": train_score,
        "test_score": test_score,
        "gap": gap
    }

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(leaderboard_file, index=False)

    print(f"\nExperiment Logged Successfully (ID: {experiment_id})")


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

        scores_dict[name] = {
            "mean": np.mean(scores),
            "std": np.std(scores)
        }

        print(f"{name} → CV Mean: {np.mean(scores):.4f} | Std: {np.std(scores):.4f}")

    return scores_dict


# ==========================================================
# BUILD TOP 3 ENSEMBLE
# ==========================================================

def build_top3_ensemble(models, scores_dict, task_type):

    sorted_models = sorted(
        scores_dict.items(),
        key=lambda x: x[1]["mean"],
        reverse=True
    )

    top3 = sorted_models[:3]

    print("\nTop 3 Models Selected for Ensemble:")
    for name, stats in top3:
        print(f"{name} ({stats['mean']:.4f})")

    estimators = [(name, models[name]) for name, _ in top3]

    if task_type == "classification":
        return VotingClassifier(estimators=estimators, voting="soft")
    else:
        return VotingRegressor(estimators=estimators)


# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train(csv_path, target_col):

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError("Target column not found")

    # 🔥 VALIDATION ADDED HERE
    validate_dataset(df, target_col)

    print(f"\nDataset Loaded: {df.shape[0]} rows")

    dataset_name = os.path.basename(csv_path).replace(".csv", "")

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

    num_cols, cat_cols = detect_columns(df_train, target_col)
    preprocessing_pipeline = build_pipeline(num_cols, cat_cols)

    models = get_models(task_type)

    print("\nSelect Training Mode:")
    print("1. Auto Select Best Model")
    print("2. Manual Model")
    print("3. Auto Ensemble (Top 3)")

    mode = input("Enter option: ")

    final_model = None
    training_mode = ""
    cv_mean = None
    cv_std = None

    if mode == "1":

        scores_dict = run_cross_validation(
            models, preprocessing_pipeline,
            X_train, y_train,
            task_type, scoring_metric
        )

        best_model_name = max(scores_dict, key=lambda x: scores_dict[x]["mean"])

        final_model = models[best_model_name]
        training_mode = "Auto Best"
        cv_mean = scores_dict[best_model_name]["mean"]
        cv_std = scores_dict[best_model_name]["std"]

        print(f"\nBest Model Selected: {best_model_name}")

    elif mode == "2":

        model_list = list(models.keys())

        for i, name in enumerate(model_list, 1):
            print(f"{i}. {name}")

        selection = int(input("Select model number: "))
        final_model = models[model_list[selection - 1]]
        training_mode = "Manual"

    elif mode == "3":

        scores_dict = run_cross_validation(
            models, preprocessing_pipeline,
            X_train, y_train,
            task_type, scoring_metric
        )

        final_model = build_top3_ensemble(
            models, scores_dict, task_type
        )

        best_model_name = max(scores_dict, key=lambda x: scores_dict[x]["mean"])

        training_mode = "Auto Ensemble"
        cv_mean = scores_dict[best_model_name]["mean"]
        cv_std = scores_dict[best_model_name]["std"]

    else:
        print("Invalid option.")
        return

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("model", final_model)
    ])

    full_pipeline.fit(X_train, y_train)

    train_preds = full_pipeline.predict(X_train)
    test_preds = full_pipeline.predict(X_test)

    if task_type == "regression":
        train_score = r2_score(y_train, train_preds)
        test_score = r2_score(y_test, test_preds)
    else:
        train_score = accuracy_score(y_train, train_preds)
        test_score = accuracy_score(y_test, test_preds)

    gap = train_score - test_score

    print(f"\nTrain Score: {train_score:.4f}")
    print(f"Test Score:  {test_score:.4f}")
    print(f"Gap: {gap:.4f}")

    print("\nFinal Evaluation:")
    evaluate_model(task_type, y_test, test_preds)

    update_leaderboard(
        dataset_name,
        final_model.__class__.__name__,
        training_mode,
        cv_mean,
        cv_std,
        train_score,
        test_score,
        gap
    )

    export_feature_importance(full_pipeline)

    print("\nGenerate SHAP Explainability? (Y/N)")
    shap_choice = input().lower()

    if shap_choice == "y":
        sample_data = X_test.sample(min(200, len(X_test)), random_state=42)
        generate_shap_explanations(full_pipeline, sample_data, task_type)

    model_path = get_next_model_version(dataset_name)
    joblib.dump(full_pipeline, model_path)

    print(f"\nModel saved successfully at: {model_path}")


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