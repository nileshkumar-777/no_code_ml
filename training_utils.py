import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier, VotingRegressor


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