import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier, VotingRegressor


# ==========================================================
# FEATURE IMPORTANCE EXPORT
# ==========================================================

def export_feature_importance(trained_pipeline):

    try:

        model = trained_pipeline.named_steps["model"]
        preprocessing = trained_pipeline.named_steps["preprocessing"]

        # --------------------------------------------------
        # Get feature names from preprocessing
        # --------------------------------------------------

        try:
            feature_names = preprocessing.get_feature_names_out()
        except Exception:
            feature_names = None

        # --------------------------------------------------
        # Handle feature selection if present
        # --------------------------------------------------

        if "feature_selection" in trained_pipeline.named_steps:

            selector = trained_pipeline.named_steps["feature_selection"]

            if feature_names is not None:
                mask = selector.get_support()
                feature_names = [f for f, keep in zip(feature_names, mask) if keep]

        # Fallback if feature names cannot be extracted
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

        # --------------------------------------------------
        # Extract feature importance
        # --------------------------------------------------

        if hasattr(model, "feature_importances_"):

            importance_values = model.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance_values
            })

            # Convert to percentage
            importance_df["Importance (%)"] = (
                importance_df["Importance"] /
                importance_df["Importance"].sum()
            ) * 100

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            importance_df.to_csv("feature_importance.csv", index=False)

            print("\nTop 10 Important Features:")
            print(
                importance_df[["Feature", "Importance (%)"]]
                .head(10)
                .round(2)
            )

        else:
            print("\nFeature importance not supported for this model.")

    except Exception as e:
        print(f"\nFeature importance failed: {e}")


# ==========================================================
# CROSS VALIDATION
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