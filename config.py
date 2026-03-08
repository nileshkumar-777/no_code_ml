# ==========================================================
#                    CONFIGURATION FILE
# ==========================================================

import os

# Absolute project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models directory
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)