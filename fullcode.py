# ==========================================================
#                    IMPORT SECTION
# ==========================================================

import os                          # Used to check if model files exist
import joblib                      # Used to save and load trained models
import pandas as pd                # Used for reading CSV files
import numpy as np                 # Used for numerical operations


# ==========================================================
#                 SCIKIT-LEARN IMPORTS
# ==========================================================

from sklearn.model_selection import train_test_split   # Splits data into train/test
from sklearn.pipeline import Pipeline                  # Creates preprocessing pipelines
from sklearn.compose import ColumnTransformer          # Applies different preprocessing to different columns
from sklearn.impute import SimpleImputer               # Fills missing values
from sklearn.preprocessing import StandardScaler       # Scales numeric features
from sklearn.preprocessing import OneHotEncoder        # Converts categorical features to numeric


# -------- Classification Models --------
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# -------- Evaluation Metrics --------
from sklearn.metrics import accuracy_score             # Calculates accuracy
from sklearn.metrics import confusion_matrix           # Shows confusion matrix
from sklearn.metrics import classification_report      # Shows precision/recall/F1
from sklearn.metrics import f1_score                   # Calculates F1 score


# ==========================================================
#               FILE STORAGE CONFIGURATION
# ==========================================================

MODEL_FILE = "best_model.pkl"        # File to store trained model
PIPELINE_FILE = "pipeline.pkl"       # File to store preprocessing pipeline


# ==========================================================
#         FUNCTION: AUTOMATIC COLUMN TYPE DETECTION
# ==========================================================

def detect_columns(df, target_col):

    X = df.drop(columns=[target_col])  
    # Remove target column to isolate features

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Detect numeric feature columns automatically

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # Detect categorical feature columns automatically

    return num_cols, cat_cols
    # Return both lists


# ==========================================================
#         FUNCTION: BUILD PREPROCESSING PIPELINE
# ==========================================================

def build_pipeline(num_cols, cat_cols):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  
        # Replace missing numeric values with median

        ("scaler", StandardScaler())  
        # Scale numeric features (important for distance-based models)
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  
        # Replace missing categorical values with most common value

        ("onehot", OneHotEncoder(handle_unknown="ignore"))  
        # Convert categorical features into one-hot encoded columns
        # handle_unknown avoids crash during inference
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),  
        # Apply numeric pipeline to numeric columns

        ("cat", cat_pipeline, cat_cols)  
        # Apply categorical pipeline to categorical columns
    ])


# ==========================================================
#                    TRAIN FUNCTION
# ==========================================================

def train(csv_path, target_col):

    df_full = pd.read_csv(csv_path)  
    # Load dataset from CSV

    if target_col not in df_full.columns:
        raise ValueError("Target column not found")  
        # Safety check

    print(f"\nFull Dataset Loaded: {df_full.shape[0]} rows")

    X_full = df_full.drop(columns=[target_col])  
    # Separate features

    y_full = df_full[target_col]  
    # Separate target

    print("\nClass Distribution:")
    print(y_full.value_counts())  
    # Show how many samples per class
    # Helps detect class imbalance

    num_cols, cat_cols = detect_columns(df_full, target_col)
    # Detect feature types

    pipeline = build_pipeline(num_cols, cat_cols)
    # Build preprocessing pipeline

    pipeline.fit(X_full)  
    # Fit pipeline on FULL dataset
    # Captures all categories


    # ---------------- SAMPLE FOR SPEED ----------------

    sample_size = max(int(len(df_full) * 0.05), 2000)  
    # Use 5% of dataset or minimum 2000 rows

    sample_size = min(sample_size, len(df_full))  
    # Prevent sampling more rows than available

    df_sample = df_full.sample(n=sample_size, random_state=42)  
    # Random sampling for fair model comparison

    print(f"\nUsing {sample_size} rows for model comparison")

    X_sample = df_sample.drop(columns=[target_col])
    y_sample = df_sample[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample,
        test_size=0.2,
        random_state=42
    )
    # Split sample into 80% train and 20% test

    X_train_prepared = pipeline.transform(X_train)
    # Apply preprocessing to training data

    X_test_prepared = pipeline.transform(X_test)
    # Apply preprocessing to testing data


    # ---------------- MODEL LIST ----------------

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    best_model_name = None
    best_accuracy = 0

    print("\nModel Comparison (Higher Accuracy = Better)\n")

    for name, model in models.items():

        model.fit(X_train_prepared, y_train)  
        # Train model

        preds = model.predict(X_test_prepared)  
        # Predict on test set

        acc = accuracy_score(y_test, preds)  
        # Calculate accuracy

        print(f"{name} Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_preds = preds
            # Store best predictions


    print(f"\nBest Model: {best_model_name}")

    # ---------------- ADVANCED EVALUATION ----------------

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_preds))
    # Shows actual vs predicted counts

    print("\nClassification Report:")
    print(classification_report(y_test, best_preds))
    # Shows precision, recall, F1-score per class

    weighted_f1 = f1_score(y_test, best_preds, average="weighted")
    # Weighted F1 accounts for class imbalance

    print(f"Weighted F1 Score: {weighted_f1:.4f}")


    # ---------------- FINAL TRAINING ----------------

    print("\nRetraining best model on FULL dataset...\n")

    final_model = models[best_model_name]
    # Select best model

    X_full_prepared = pipeline.transform(X_full)
    # Transform full dataset

    final_model.fit(X_full_prepared, y_full)
    # Train on entire dataset

    joblib.dump(final_model, MODEL_FILE)
    # Save model

    joblib.dump(pipeline, PIPELINE_FILE)
    # Save preprocessing pipeline

    print("Final model saved successfully.")


# ==========================================================
#                   INFERENCE FUNCTION
# ==========================================================

def inference(csv_path, output_path="predictions.csv"):

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Train the model first")

    model = joblib.load(MODEL_FILE)
    # Load trained model

    pipeline = joblib.load(PIPELINE_FILE)
    # Load preprocessing pipeline

    df = pd.read_csv(csv_path)
    # Load new unseen data

    transformed = pipeline.transform(df)
    # Apply preprocessing

    predictions = model.predict(transformed)
    # Generate predictions

    df["predicted_value"] = predictions
    # Add predictions column

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)
        df["confidence"] = probabilities.max(axis=1)
        # Add confidence score

    df.to_csv(output_path, index=False)
    # Save results

    print(f"Inference Complete. Predictions saved to: {output_path}")


# ==========================================================
#                      CLI MENU
# ==========================================================

if __name__ == "__main__":

    print("1. Train Model")
    print("2. Run Inference")

    choice = input("Select option: ")

    if choice == "1":
        csv_path = input("Enter CSV path: ")
        target_col = input("Enter target column: ")
        train(csv_path, target_col)

    elif choice == "2":
        csv_path = input("Enter inference CSV path: ")
        inference(csv_path)

    else:
        print("Invalid option.")


# Add cross-validation

# Auto-detect imbalance and apply class_weight

# Plot confusion matrix graphically

# Add feature importance report