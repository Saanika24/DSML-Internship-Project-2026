
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(data):
    """
    Build a preprocessing pipeline for numerical
    and categorical features.

    Parameters:
        data: Pandas DataFrame containing input features.

    Returns:
        A Scikit-learn ColumnTransformer.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if data.empty or data.shape[1] == 0:
        raise ValueError("Input dataset cannot be empty.")

    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")

    # Identify numerical columns
    numerical_columns = data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identify categorical and Boolean columns
    categorical_columns = data.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Datetime columns need explicit feature engineering
    unsupported_columns = [
        column for column in data.columns
        if column not in numerical_columns
        and column not in categorical_columns
    ]

    if unsupported_columns:
        raise ValueError(
            "Unsupported columns found. Convert datetime or "
            f"other columns before preprocessing: {unsupported_columns}"
        )

    transformers = []

    # Numerical preprocessing
    if numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    # Categorical preprocessing
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="missing"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def prepare_training_data(X_train, X_test):
    """
    Fit preprocessing only on training data and
    transform both training and test data.

    Returns:
        preprocessor, transformed training data,
        transformed test data.
    """

    if not X_train.columns.equals(X_test.columns):
        raise ValueError(
            "Training and test datasets must have identical columns "
            "in the same order."
        )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    return (
        preprocessor,
        X_train_processed,
        X_test_processed
    )
