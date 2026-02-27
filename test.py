# ==========================================================
#                    MAIN ENTRY POINT
# ==========================================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Import custom modules
from config import MODEL_FILE, PIPELINE_FILE
from preprocessing import detect_columns, build_pipeline
from model_selection import get_models
from evaluation import evaluate_model
from inference import run_inference


def train(csv_path, target_col):

    # ---------------- LOAD DATA ----------------

    df_full = pd.read_csv(csv_path)

    if target_col not in df_full.columns:
        raise ValueError("Target column not found")

    print(f"\nDataset Loaded: {df_full.shape[0]} rows")

    # ---------------- SPLIT FEATURES & TARGET ----------------

    X_full = df_full.drop(columns=[target_col])
    y_full = df_full[target_col]

    # ---------------- PRINT TARGET STATS ----------------

    print("\n========== TARGET STATISTICS ==========")
    print(y_full.describe())

    # ---------------- TASK TYPE DETECTION ----------------

    if y_full.dtype in ["int64", "float64"] and y_full.nunique() > 20:
        task_type = "regression"
        print("\nDetected Task Type: REGRESSION")
    else:
        task_type = "classification"
        print("\nDetected Task Type: CLASSIFICATION")
        print("\nClass Distribution:")
        print(y_full.value_counts())

    # ---------------- CORRELATION CHECK (REGRESSION ONLY) ----------------

    if task_type == "regression":
        print("\n========== CORRELATION WITH TARGET ==========")
        correlations = (
            df_full.corr(numeric_only=True)[target_col]
            .sort_values(ascending=False)
        )
        print(correlations)

    # ---------------- BUILD PREPROCESSING ----------------

    num_cols, cat_cols = detect_columns(df_full, target_col)
    pipeline = build_pipeline(num_cols, cat_cols)

    pipeline.fit(X_full)

    # ---------------- USE FULL DATA FOR MODEL SELECTION ----------------

    df_sample = df_full   # using full dataset for debugging

    X_sample = df_sample.drop(columns=[target_col])
    y_sample = df_sample[target_col]

    # ---------------- TRAIN-TEST SPLIT ----------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        test_size=0.2,
        random_state=42
    )

    X_train_prepared = pipeline.transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    # ---------------- LOAD MODELS ----------------

    models = get_models(task_type)

    # ======================================================
    #            SELECT TRAINING MODE
    # ======================================================

    print("\nSelect Training Mode:")
    print("1. Auto Select Best Model")
    print("2. Manually Choose Model")

    mode = input("Enter option: ")

    # ------------------------------------------------------
    # AUTO MODE
    # ------------------------------------------------------
    if mode == "1":

        best_model_name = None
        best_score = float("-inf")

        print("\nModel Comparison:\n")

        for name, model in models.items():

            model.fit(X_train_prepared, y_train)
            preds = model.predict(X_test_prepared)

            score = evaluate_model(task_type, y_test, preds)

            # Print prediction range for debugging
            print("\nPrediction Range:")
            print("Min prediction:", preds.min())
            print("Max prediction:", preds.max())
            print("Mean prediction:", preds.mean())

            if score > best_score:
                best_score = score
                best_model_name = name

        print(f"\nBest Model (Auto Selected): {best_model_name}")
        final_model = models[best_model_name]

    # ------------------------------------------------------
    # MANUAL MODE
    # ------------------------------------------------------
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

        final_model.fit(X_train_prepared, y_train)
        preds = final_model.predict(X_test_prepared)

        evaluate_model(task_type, y_test, preds)

        print("\nPrediction Range:")
        print("Min prediction:", preds.min())
        print("Max prediction:", preds.max())
        print("Mean prediction:", preds.mean())

    else:
        print("Invalid mode selected.")
        return

    # ======================================================
    #           FINAL TRAINING ON FULL DATASET
    # ======================================================

    print("\nRetraining selected model on FULL dataset...")

    X_full_prepared = pipeline.transform(X_full)
    final_model.fit(X_full_prepared, y_full)

    # ---------------- SAVE MODEL ----------------

    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Final model saved successfully.")


# ==========================================================
#                     CLI MENU
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