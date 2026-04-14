# src/data_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(data_path):
    print(f"[📂] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Optional: forward fill missing values
    df.fillna(method='ffill', inplace=True)

    # Check for expected target column
    if 'loan_approved' not in df.columns:
        raise ValueError("[❌] Target column 'loan_approved' not found in dataset." \
        "\n[ℹ️] Make sure your dataset has a 'loan_approved' column as the target.")

    # Normalize target values: Approved → 1, Denied → 0
    if df['loan_approved'].dtype == object:
        df['loan_approved'] = df['loan_approved'].str.strip().map({
            "Approved": 1,
            "Denied": 0
        })

    # Final check to ensure binary format
    if df['loan_approved'].isnull().any():
        raise ValueError("[❌] Invalid values in 'loan_approved'. Must be 'Approved' or 'Denied'.")

    # Required modeling features
    required_features = [
        'age', 'income', 'loan_amount', 'credit_score',
        'gender', 'race', 'zip_code_group'
    ]
    for col in required_features:
        if col not in df.columns:
            raise ValueError(f"[❌] Missing required feature column: '{col}'")

    # Select features + target
    X = df[required_features]
    y = df['loan_approved']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=False)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[✅] Data loaded, processed, and split successfully.")
    return (X_train, X_test, y_train, y_test), X.columns.tolist()
