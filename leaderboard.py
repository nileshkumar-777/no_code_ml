import pandas as pd
import os
from datetime import datetime


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