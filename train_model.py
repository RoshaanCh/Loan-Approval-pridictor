import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_approval_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")


def load_data():
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    df.columns = df.columns.str.strip()

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype(str).str.strip()

    df = df.drop(columns=["loan_id"])
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
    df = df.dropna(subset=["loan_status"])

    numeric_columns = [
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()
    return df


def prepare_features(df, feature_columns=None):
    df = pd.get_dummies(df, columns=["education", "self_employed"], drop_first=True)

    if feature_columns is not None:
        df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def train_and_evaluate():
    df = clean_data(load_data())

    print("Rows:", len(df))
    print("Target counts:")
    print(df["loan_status"].value_counts())

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train = prepare_features(X_train)
    X_test = prepare_features(X_test, X_train.columns)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X_train.columns)}, MODEL_PATH)
    print("Model saved to", MODEL_PATH)


if __name__ == "__main__":
    train_and_evaluate()