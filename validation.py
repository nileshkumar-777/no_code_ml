import pandas as pd


def validate_dataset(df, target_col):

    print("\n================ DATA VALIDATION REPORT ================\n")

    total_rows = df.shape[0]
    total_cols = df.shape[1]

    print(f"Total Rows: {total_rows}")
    print(f"Total Columns: {total_cols}")

    missing_percent = df.isnull().mean() * 100
    high_missing = missing_percent[missing_percent > 20]

    if not high_missing.empty:
        print("\n⚠ Warning: Columns with >20% missing values:")
        print(high_missing.sort_values(ascending=False))
    else:
        print("\nNo major missing value issues detected.")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\n⚠ Warning: {duplicate_count} duplicate rows found.")
    else:
        print("\nNo duplicate rows detected.")

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    if constant_cols:
        print("\n⚠ Warning: Constant columns detected:")
        print(constant_cols)
    else:
        print("\nNo constant columns detected.")

    if total_rows < 200:
        print("\n⚠ Warning: Dataset is small. Risk of overfitting.")

    if df[target_col].nunique() < 20:
        class_distribution = df[target_col].value_counts(normalize=True) * 100

        if class_distribution.min() < 10:
            print("\n⚠ Warning: Class imbalance detected:")
            print(class_distribution.round(2))

    print("\n=========================================================\n")