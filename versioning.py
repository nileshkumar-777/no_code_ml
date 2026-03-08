import os
from config import MODEL_DIR


def get_next_model_version(dataset_name):

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    existing_files = os.listdir(MODEL_DIR)
    versions = []

    for file in existing_files:
        if file.startswith(dataset_name + "_v") and file.endswith(".pkl"):
            try:
                version = int(file.split("_v")[-1].replace(".pkl", ""))
                versions.append(version)
            except:
                pass

    next_version = 1 if not versions else max(versions) + 1

    model_filename = f"{dataset_name}_v{next_version}.pkl"

    return os.path.join(MODEL_DIR, model_filename)