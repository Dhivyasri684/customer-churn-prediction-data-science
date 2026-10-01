"""
Week 3: Python-Based Machine Learning Model Development and Evaluation Plan
Project: Customer Churn Prediction

This is a reusable conceptual implementation template. No real dataset is
included in this Week 3 task. Replace DATA_PATH with a real CSV when available.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_PATH = "customer_churn.csv"
TARGET = "Churn"


def load_data(path):
    """Load the dataset."""
    return pd.read_csv(path)


def prepare_features(df, target):
    """Separate target and features."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(X):
    """Create preprocessing for numerical and categorical columns."""
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])


def build_models(preprocessor):
    """Define candidate classification models."""
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ))
        ])
    }


def evaluate_model(model, X_test, y_test):
    """Calculate standard classification metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
    }


def run_pipeline(path=DATA_PATH):
    """Run the planned training and evaluation workflow."""
    df = load_data(path)

    # Basic quality preparation
    df = df.drop_duplicates()

    X, y = prepare_features(df, TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)
        print(f"\n{name}")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))
        print(results[name])

    return results


if __name__ == "__main__":
    # Uncomment after placing customer_churn.csv in this folder.
    # run_pipeline()
    print("Week 3 ML framework is ready. Add a dataset and uncomment run_pipeline().")
