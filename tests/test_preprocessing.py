
import numpy as np
import pandas as pd
import pytest

from src.prediction.preprocessing import (
    build_preprocessor,
    prepare_training_data
)


@pytest.fixture
def sample_data():

    return pd.DataFrame({
        "supplier_id": [101, 102, 103, 104, 105],
        "shipment_cost": [500, 700, np.nan, 600, 800],
        "delay_days": [2, 0, 4, np.nan, 3],
        "transport_mode": [
            "Road", "Air", "Road", None, "Sea"
        ]
    })


def test_preprocessor_creation(sample_data):

    preprocessor = build_preprocessor(sample_data)

    assert preprocessor is not None


def test_preprocessing_handles_missing_values(sample_data):

    preprocessor = build_preprocessor(sample_data)

    transformed = preprocessor.fit_transform(sample_data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert np.isfinite(transformed).all()


def test_preprocessing_preserves_all_rows(sample_data):

    preprocessor = build_preprocessor(sample_data)

    transformed = preprocessor.fit_transform(sample_data)

    assert transformed.shape[0] == len(sample_data)


def test_preprocessing_handles_unknown_categories():

    train_data = pd.DataFrame({
        "shipment_cost": [500, 600, 700],
        "transport_mode": ["Road", "Air", "Sea"]
    })

    test_data = pd.DataFrame({
        "shipment_cost": [650],
        "transport_mode": ["Rail"]
    })

    preprocessor = build_preprocessor(train_data)

    preprocessor.fit(train_data)

    transformed = preprocessor.transform(test_data)

    assert transformed.shape[0] == 1


def test_preprocessing_rejects_empty_data():

    empty_data = pd.DataFrame()

    with pytest.raises(ValueError):
        build_preprocessor(empty_data)


def test_preprocessing_rejects_non_dataframe():

    with pytest.raises(TypeError):
        build_preprocessor([[1, 2], [3, 4]])


def test_training_and_test_data_preparation(sample_data):

    X_train = sample_data.iloc[:4].copy()
    X_test = sample_data.iloc[4:].copy()

    preprocessor, train_processed, test_processed = (
        prepare_training_data(X_train, X_test)
    )

    assert preprocessor is not None
    assert train_processed.shape[0] == len(X_train)
    assert test_processed.shape[0] == len(X_test)


def test_preparation_rejects_different_columns():

    X_train = pd.DataFrame({
        "shipment_cost": [500, 600],
        "transport_mode": ["Road", "Air"]
    })

    X_test = pd.DataFrame({
        "shipment_cost": [700],
        "carrier": ["Company A"]
    })

    with pytest.raises(ValueError):
        prepare_training_data(X_train, X_test)


def test_preprocessing_rejects_datetime_columns():

    data = pd.DataFrame({
        "shipment_date": pd.to_datetime([
            "2026-01-01", "2026-01-02"
        ]),
        "shipment_cost": [500, 600]
    })

    with pytest.raises(ValueError, match="Unsupported columns"):
        build_preprocessor(data)
