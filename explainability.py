import shap
import pandas as pd
import numpy as np


# ==========================================================
# SHAP EXPLAINABILITY
# ==========================================================

def generate_shap_explanations(pipeline, X_sample, task_type):

    try:

        model = pipeline.named_steps["model"]
        preprocessing = pipeline.named_steps["preprocessing"]

        # --------------------------------------------------
        # Transform sample data
        # --------------------------------------------------

        X_processed = preprocessing.transform(X_sample)

        # --------------------------------------------------
        # Try extracting feature names
        # --------------------------------------------------

        try:
            feature_names = preprocessing.get_feature_names_out()

        except Exception:
            print("Could not extract feature names from preprocessing pipeline.")

            # fallback feature names
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

        # --------------------------------------------------
        # SHAP only supported for tree models
        # --------------------------------------------------

        if hasattr(model, "feature_importances_"):

            print("\nGenerating SHAP values...")

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_processed)

            shap.summary_plot(
                shap_values,
                X_processed,
                feature_names=feature_names,
                show=False
            )

            print("SHAP summary plot generated.")

        else:
            print("SHAP not supported for this model.")

    except Exception as e:
        print(f"SHAP generation failed: {e}")