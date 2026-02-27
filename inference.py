# ==========================================================
#                INFERENCE MODULE
# ==========================================================

import os                          # Used to check if model file exists
import joblib                      # Used to load trained pipeline
import pandas as pd                # Used to read CSV files
from config import MODEL_FILE      # Import saved pipeline file name


def run_inference(csv_path, output_path="predictions.csv"):
    """
    Perform inference using saved FULL pipeline
    (Preprocessing + Model together).
    """

    # ------------------------------------------------------
    # CHECK IF TRAINED MODEL EXISTS
    # ------------------------------------------------------

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Train the model first")

    # ------------------------------------------------------
    # LOAD FULL PIPELINE
    # ------------------------------------------------------

    # This pipeline already contains:
    # 1. Preprocessing
    # 2. Trained model
    full_pipeline = joblib.load(MODEL_FILE)

    # ------------------------------------------------------
    # LOAD INPUT DATA
    # ------------------------------------------------------

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------
    # GENERATE PREDICTIONS
    # ------------------------------------------------------

    # No manual preprocessing needed
    # Pipeline handles everything internally
    predictions = full_pipeline.predict(df)

    # ------------------------------------------------------
    # ADD PREDICTIONS COLUMN
    # ------------------------------------------------------

    df["predicted_value"] = predictions

    # ------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------

    df.to_csv(output_path, index=False)

    print(f"Inference Complete. Predictions saved to: {output_path}")