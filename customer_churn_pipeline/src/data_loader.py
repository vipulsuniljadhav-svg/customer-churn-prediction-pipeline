"""Data loading, target encoding, and stratified dataset splitting."""

import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COLUMN,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)


def load_raw_dataset(filepath: str = None) -> pd.DataFrame:
    """Load the raw Telco customer churn dataset from CSV."""
    path = filepath or RAW_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    df = pd.read_csv(path)
    return df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target column, encode it to binary integer, and remove ID column."""
    df_clean = df.copy()

    if TARGET_COLUMN in df_clean.columns:
        # Convert "Yes"/"No" or binary to 1/0
        y = (
            df_clean[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        )
        if y.isnull().any():
            y = y.fillna(0).astype(int)
        else:
            y = y.astype(int)
        X = df_clean.drop(columns=[TARGET_COLUMN])
    else:
        y = None
        X = df_clean

    if ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])

    return X, y


def split_and_save_data(
    filepath: str = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, split into stratified train/test sets, and cache to processed directory."""
    raw_df = load_raw_dataset(filepath)
    X, y = prepare_features_and_target(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Save processed splits for reproducibility
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_export = X_train.copy()
    train_export[TARGET_COLUMN] = y_train
    train_export.to_csv(TRAIN_DATA_PATH, index=False)

    test_export = X_test.copy()
    test_export[TARGET_COLUMN] = y_test
    test_export.to_csv(TEST_DATA_PATH, index=False)

    return X_train, X_test, y_train, y_test
