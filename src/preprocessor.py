"""
preprocessor.py
---------------
Data cleaning and preprocessing functions for the diabetes
dataset. All transformations are documented with the reasoning
behind each decision.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Columns where zero is physiologically impossible
# and must be treated as missing data
ZERO_NOT_VALID = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]


def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zero values in medical columns with NaN.

    Rationale: In the Pima Indians dataset, several columns
    contain zeros that are physiologically impossible. For
    example, a blood pressure of 0 or a BMI of 0 cannot exist
    in a living patient. These represent missing data encoded
    as zeros and must be treated accordingly.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with zero-encoded missing values.

    Returns
    -------
    pd.DataFrame
        Dataframe with invalid zeros replaced by NaN.
    """
    df = df.copy()
    for col in ZERO_NOT_VALID:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            df[col] = df[col].replace(0, np.nan)
            print(f"  {col}: {zero_count} zeros replaced with NaN")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median imputation per outcome group.

    Rationale: Median is preferred over mean because medical data
    often contains outliers that skew the mean. Group-wise imputation
    by Outcome (diabetic vs non-diabetic) preserves the statistical
    differences between groups rather than collapsing them into a
    single population median.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with NaN values to be imputed.

    Returns
    -------
    pd.DataFrame
        Dataframe with all missing values filled.
    """
    df = df.copy()
    for col in ZERO_NOT_VALID:
        if col in df.columns and df[col].isnull().any():
            group_medians = df.groupby('Outcome')[col].transform('median')
            df[col] = df[col].fillna(group_medians)
            print(f"  {col}: imputed with group-wise median")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns to improve
    model performance.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered features.
    """
    df = df.copy()

    # BMI categories based on WHO classification
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    ).astype(str)

    # Age groups for demographic segmentation
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 45, 60, 100],
        labels=['Young', 'Middle', 'Senior', 'Elderly']
    ).astype(str)

    # Glucose severity based on clinical thresholds
    df['Glucose_Level'] = pd.cut(
        df['Glucose'],
        bins=[0, 99, 125, 500],
        labels=['Normal', 'Prediabetic', 'Diabetic']
    ).astype(str)

    # Interaction feature: high glucose with high BMI
    df['HighRisk_Score'] = (
        (df['Glucose'] / df['Glucose'].max()) * 0.4 +
        (df['BMI'] / df['BMI'].max()) * 0.3 +
        (df['Age'] / df['Age'].max()) * 0.3
    ).round(4)

    print(f"Feature engineering complete. New shape: {df.shape}")
    return df


def split_and_scale(
    df: pd.DataFrame,
    target_col: str = 'Outcome',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split data into train and test sets and apply standard scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed dataframe.
    target_col : str
        Name of the target column.
    test_size : float
        Proportion of data to hold out for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Select only numeric features for modeling
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Train set: {X_train_scaled.shape[0]} samples")
    print(f"Test set:  {X_test_scaled.shape[0]} samples")
    print(f"Features:  {len(feature_cols)}")

    return (X_train_scaled, X_test_scaled,
            y_train, y_test, scaler, feature_cols)