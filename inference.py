# ==========================================================
#                INFERENCE MODULE
# ==========================================================

import os                          # Used to check file existence
import joblib                      # Used to load saved model
import pandas as pd                # Used to read CSV
from config import MODEL_FILE, PIPELINE_FILE  # Import file paths


def run_inference(csv_path, output_path="predictions.csv"):
    """
    Perform inference using saved model and pipeline.
    """

    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Train the model first")

    # Load trained model
    model = joblib.load(MODEL_FILE)

    # Load preprocessing pipeline
    pipeline = joblib.load(PIPELINE_FILE)

    # Load new data
    df = pd.read_csv(csv_path)

    # Apply preprocessing
    transformed = pipeline.transform(df)

    # Generate predictions
    predictions = model.predict(transformed)

    # Add predictions column only
    df["predicted_value"] = predictions

    # Save results
    df.to_csv(output_path, index=False)

    print(f"Inference Complete. Predictions saved to: {output_path}")