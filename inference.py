# ==========================================================
#                INFERENCE MODULE
# ==========================================================

import os
import joblib
import pandas as pd

from config import MODEL_DIR


# ==========================================================
# LIST AVAILABLE MODELS
# ==========================================================

def list_available_models():

    if not os.path.exists(MODEL_DIR):
        print("No models folder found. Train a model first.")
        return []

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

    if not model_files:
        print("No trained models found.")
        return []

    print("\nAvailable Models:")

    for idx, model in enumerate(model_files, 1):
        print(f"{idx}. {model}")

    return model_files


# ==========================================================
# RUN INFERENCE
# ==========================================================

def run_inference(csv_path):

    model_files = list_available_models()

    if not model_files:
        return

    selection = int(input("\nSelect model number to use: "))

    if selection < 1 or selection > len(model_files):
        print("Invalid selection.")
        return

    selected_model_file = model_files[selection - 1]
    model_path = os.path.join(MODEL_DIR, selected_model_file)

    print(f"\nLoading model: {selected_model_file}")

    # Load full pipeline (preprocessing + model)
    full_pipeline = joblib.load(model_path)

    # Load inference data
    df = pd.read_csv(csv_path)

    # Generate predictions
    predictions = full_pipeline.predict(df)

    df["predicted_value"] = predictions

    # Save output
    output_filename = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(output_filename, index=False)

    print(f"\nInference Complete.")
    print(f"Predictions saved to: {output_filename}")