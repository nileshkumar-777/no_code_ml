import shap
import pandas as pd
import numpy as np


def generate_shap_explanations(pipeline, X_sample, task_type):

    try:
        model = pipeline.named_steps["model"]
        preprocessing = pipeline.named_steps["preprocessing"]

        # Transform data
        X_processed = preprocessing.transform(X_sample)

        feature_names = preprocessing.get_feature_names_out()

        # Only for tree models
        if hasattr(model, "feature_importances_"):

            print("\nGenerating SHAP values...")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)

            # Global importance
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