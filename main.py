import pandas as pd
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

from config import MODEL_DIR
from preprocessing import detect_columns, build_pipeline
from model_selection import get_models
from evaluation import evaluate_model
from inference import run_inference, list_available_models
from explainability import generate_shap_explanations
from validation import validate_dataset
from versioning import get_next_model_version
from leaderboard import update_leaderboard
from training_utils import (
    export_feature_importance,
    run_cross_validation,
    build_top3_ensemble
)
from hyperparameter import tune_model


# ==========================================================
# TRAINING REPORT GENERATOR
# ==========================================================

def generate_training_report(
    dataset_name,
    task_type,
    model_name,
    training_mode,
    cv_mean,
    train_score,
    test_score,
    model_path
):

    report_file = f"{dataset_name}_training_report.txt"

    with open(report_file, "w") as f:

        f.write("=========== TRAINING REPORT ===========\n\n")

        f.write(f"Timestamp: {datetime.now()}\n\n")

        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Task Type: {task_type}\n\n")

        f.write(f"Model Used: {model_name}\n")
        f.write(f"Training Mode: {training_mode}\n\n")

        if cv_mean is not None:
            f.write(f"Cross Validation Mean: {cv_mean:.4f}\n")

        if train_score is not None:
            f.write(f"Train Score: {train_score:.4f}\n")

        if test_score is not None:
            f.write(f"Test Score: {test_score:.4f}\n")

        f.write(f"\nSaved Model Path: {model_path}\n")

        f.write("\n=======================================\n")

    print(f"\nTraining report saved: {report_file}")


# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train(csv_path, target_col):

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError("Target column not found")

    validate_dataset(df, target_col)

    dataset_name = os.path.basename(csv_path).replace(".csv", "")

    # ======================================================
    # DATA TRAINING MODE
    # ======================================================

    print("\nSelect Data Training Mode:")
    print("1. Train using Train/Test Split")
    print("2. Train using FULL Dataset")

    data_mode = input("Enter option: ")

    if data_mode == "1":

        df_train, df_test = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        train_file = f"{dataset_name}_train.csv"
        test_file = f"{dataset_name}_test.csv"

        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)

        print(f"\nTrain CSV saved: {train_file}")
        print(f"Test CSV saved: {test_file}")

        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]

        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        print(f"\nTrain Size: {len(X_train)}")
        print(f"Test Size: {len(X_test)}")

    elif data_mode == "2":

        X_train = df.drop(columns=[target_col])
        y_train = df[target_col]

        X_test = None
        y_test = None

        print("\nUsing FULL dataset for training")

    else:
        print("Invalid option")
        return

    # ======================================================
    # TASK TYPE DETECTION
    # ======================================================

    if y_train.dtype in ["int64", "float64"] and y_train.nunique() > 20:
        task_type = "regression"
        scoring_metric = "r2"
    else:
        task_type = "classification"
        scoring_metric = "accuracy"

    print(f"\nDetected Task Type: {task_type.upper()}")

    # ======================================================
    # PREPROCESSING
    # ======================================================

    num_cols, cat_cols = detect_columns(X_train, target_col)

    preprocessing_pipeline = build_pipeline(num_cols, cat_cols)

    # ======================================================
    # FEATURE SELECTION
    # ======================================================

    if task_type == "regression":
        feature_selector = SelectKBest(
            score_func=f_regression,
            k=min(10, X_train.shape[1])
        )
    else:
        feature_selector = SelectKBest(
            score_func=f_classif,
            k=min(10, X_train.shape[1])
        )

    models = get_models(task_type)

    # ======================================================
    # TRAINING MODE
    # ======================================================

    print("\nSelect Training Mode:")
    print("1. Auto Select Best Model")
    print("2. Manual Model")
    print("3. Auto Ensemble (Top 3)")

    mode = input("Enter option: ")

    # ======================================================
    # HYPERPARAMETER OPTION
    # ======================================================

    print("\nEnable Hyperparameter Tuning?")
    print("1. Yes")
    print("2. No")

    tuning_choice = input("Enter option: ")

    final_model = None
    training_mode = ""
    cv_mean = None
    cv_std = None

    # ======================================================
    # AUTO MODEL SELECTION
    # ======================================================

    if mode == "1":

        scores_dict = run_cross_validation(
            models,
            preprocessing_pipeline,
            X_train,
            y_train,
            task_type,
            scoring_metric
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

        if selection < 1 or selection > len(model_list):
            print("Invalid selection")
            return

        final_model = models[model_list[selection - 1]]

        training_mode = "Manual"

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

        best_model_name = max(scores_dict, key=lambda x: scores_dict[x]["mean"])

        training_mode = "Auto Ensemble"

        cv_mean = scores_dict[best_model_name]["mean"]
        cv_std = scores_dict[best_model_name]["std"]

    else:
        print("Invalid option.")
        return

    # ======================================================
    # BUILD PIPELINE
    # ======================================================

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("feature_selection", feature_selector),
        ("model", final_model)
    ])

    # ======================================================
    # HYPERPARAMETER TUNING
    # ======================================================

    if tuning_choice == "1":

        full_pipeline = tune_model(
            full_pipeline,
            final_model.__class__.__name__,
            X_train,
            y_train,
            scoring_metric
        )

    else:
        full_pipeline.fit(X_train, y_train)

    # ======================================================
    # EVALUATION
    # ======================================================

    if X_test is not None:

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
        print(f"Test Score: {test_score:.4f}")

        evaluate_model(task_type, y_test, test_preds)

    else:
        train_score = None
        test_score = None
        gap = None

    # ======================================================
    # LEADERBOARD
    # ======================================================

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

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    export_feature_importance(full_pipeline)

    # ======================================================
    # DISPLAY SELECTED FEATURES
    # ======================================================

    try:

        selector = full_pipeline.named_steps["feature_selection"]
        mask = selector.get_support()

        feature_names = preprocessing_pipeline.get_feature_names_out()

        selected_features = [
            name for name, keep in zip(feature_names, mask) if keep
        ]

        print("\nSelected Features:")

        for f in selected_features:
            print(f)

    except Exception:
        pass

    # ======================================================
    # SHAP EXPLAINABILITY
    # ======================================================

    print("\nGenerate SHAP Explainability? (Y/N)")
    shap_choice = input().lower()

    if shap_choice == "y":

        sample_data = X_train.sample(
            min(200, len(X_train)),
            random_state=42
        )

        generate_shap_explanations(
            full_pipeline,
            sample_data,
            task_type
        )

    # ======================================================
    # SAVE MODEL
    # ======================================================

    model_path = get_next_model_version(dataset_name)

    joblib.dump(full_pipeline, model_path)

    print(f"\nModel saved successfully at: {model_path}")

    # ======================================================
    # GENERATE TRAINING REPORT
    # ======================================================

    generate_training_report(
        dataset_name,
        task_type,
        final_model.__class__.__name__,
        training_mode,
        cv_mean,
        train_score,
        test_score,
        model_path
    )


# ==========================================================
# CLI MENU
# ==========================================================

if __name__ == "__main__":

    while True:

        print("\n=========== ML ENGINE ===========")
        print("1. Train Model")
        print("2. Run Inference")
        print("3. List Available Models")
        print("4. Exit")

        choice = input("Select option: ")

        if choice == "1":

            csv_path = input("Enter CSV path: ")
            target_col = input("Enter target column: ")

            train(csv_path, target_col)

        elif choice == "2":

            csv_path = input("Enter inference CSV path:")
            run_inference(csv_path)

        elif choice == "3":

            list_available_models()

        elif choice == "4":

            print("Exiting ML Engine.")
            break

        else:
            print("Invalid option.")