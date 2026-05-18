"""
data_loader.py
--------------
Utility functions for loading and performing initial validation
of the diabetes dataset. Separates data access logic from
analysis notebooks.
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset from the given filepath.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with no modifications applied.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully.")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def validate_schema(df: pd.DataFrame, expected_columns: list) -> bool:
    """
    Validate that the dataframe contains the expected columns.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataframe to validate.
    expected_columns : list
        List of column names that must be present.

    Returns
    -------
    bool
        True if all expected columns are present, False otherwise.
    """
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return False
    print("Schema validation passed.")
    return True


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive summary of the dataframe including
    data types, missing values, and basic statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to summarize.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with one row per column.
    """
    summary = pd.DataFrame({
        'dtype'       : df.dtypes,
        'non_null'    : df.notnull().sum(),
        'null_count'  : df.isnull().sum(),
        'null_pct'    : (df.isnull().sum() / len(df) * 100).round(2),
        'unique'      : df.nunique(),
        'min'         : df.min(numeric_only=True),
        'max'         : df.max(numeric_only=True),
        'mean'        : df.mean(numeric_only=True).round(2)
    })
    return summary