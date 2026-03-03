# ==========================================================
#                    CONFIGURATION FILE
# ==========================================================

import os

# Base directory to store trained models
MODEL_DIR = "models"

# Create models folder if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)