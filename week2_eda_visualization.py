"""
Week 2: Exploratory Data Analysis (EDA) and Visualization Framework
Yuva Intern - Data Science

This script provides a reusable EDA framework. It can be adapted to any
CSV dataset by changing the DATA_PATH variable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "dataset.csv"


def load_data(path):
    """Load a CSV dataset."""
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    return df


def basic_overview(df):
    """Display basic information about the dataset."""
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Dataset Information ---")
    print(df.info())

    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="all"))


def data_quality_check(df):
    """Check missing values, duplicates and unique values."""
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Missing Value Percentage ---")
    print((df.isnull().mean() * 100).round(2))

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Unique Values ---")
    print(df.nunique())


def univariate_analysis(df):
    """Create basic distributions for numerical variables."""
    numerical_columns = df.select_dtypes(include=np.number).columns

    for column in numerical_columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


def categorical_analysis(df):
    """Create bar charts for categorical variables."""
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:
        counts = df[column].value_counts().head(10)

        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts.values, y=counts.index)
        plt.title(f"Top Categories - {column}")
        plt.xlabel("Count")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()


def correlation_analysis(df):
    """Display correlation matrix for numerical variables."""
    numerical_df = df.select_dtypes(include=np.number)

    if numerical_df.shape[1] >= 2:
        plt.figure(figsize=(9, 6))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numerical columns for correlation analysis.")


def outlier_check(df):
    """Create box plots for numerical variables."""
    numerical_columns = df.select_dtypes(include=np.number).columns

    for column in numerical_columns:
        plt.figure(figsize=(7, 3))
        sns.boxplot(x=df[column])
        plt.title(f"Outlier Check - {column}")
        plt.tight_layout()
        plt.show()


def run_eda(path):
    """Run the complete EDA workflow."""
    df = load_data(path)
    basic_overview(df)
    data_quality_check(df)
    univariate_analysis(df)
    categorical_analysis(df)
    correlation_analysis(df)
    outlier_check(df)

    print("\nEDA workflow completed successfully.")


if __name__ == "__main__":
    # Replace dataset.csv with the actual CSV filename when a dataset is available.
    run_eda(DATA_PATH)
