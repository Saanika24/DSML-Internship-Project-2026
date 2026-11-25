import numpy as np
import pandas as pd
import pytest

from src.prediction import predict


class DummyClassifier:
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict_proba(self, df):
        return np.array([
            [0.20, 0.80],
            [0.75, 0.25],
        ])[:len(df)]


class DummyPipeline:
    def __init__(self):
        self.named_steps = {"model": DummyClassifier()}

    def predict_proba(self, df):
        return self.named_steps["model"].predict_proba(df)


@pytest.fixture
def sample_features():
    return pd.DataFrame({
        "supplier_id": ["SUP-001", "SUP-002"],
        "quantity_ordered": [100, 200],
    })


@pytest.fixture
def sample_artifacts():
    clf = {
        "model": DummyPipeline(),
        "feature_columns": ["supplier_id", "quantity_ordered"],
        "target": "is_delayed",
        "model_type": "classification",
        "threshold": 0.5,
    }
    reg = {
        "model": None,
        "feature_columns": ["supplier_id", "quantity_ordered"],
        "target": "delay_days",
        "model_type": "regression",
        "constant_prediction": 2.0,
    }
    return clf, reg


@pytest.fixture
def mock_models(monkeypatch, sample_artifacts):
    monkeypatch.setattr(predict, "load_prediction_models", lambda: sample_artifacts)
    return sample_artifacts


def test_prediction_returns_expected_results(sample_features, mock_models):
    results = predict.predict_purchase_order_delays(sample_features)

    assert len(results) == 2
    assert list(results.columns) == ["delay_probability", "predicted_status", "estimated_delay_days"]
    assert results["predicted_status"].tolist() == ["Delayed", "On time"]
    assert results["estimated_delay_days"].tolist() == [2.0, 2.0]


def test_prediction_rejects_missing_features(mock_models):
    incomplete_df = pd.DataFrame({"supplier_id": ["SUP-001"]})

    with pytest.raises(ValueError, match="Missing required features"):
        predict.predict_purchase_order_delays(incomplete_df)


def test_prediction_rejects_empty_data(sample_features, mock_models):
    empty_df = sample_features.iloc[0:0]

    with pytest.raises(ValueError, match="cannot be empty"):
        predict.predict_purchase_order_delays(empty_df)


def test_prediction_rejects_non_dataframe(mock_models):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        predict.predict_purchase_order_delays({"supplier_id": "SUP-001"})


def test_prediction_rejects_missing_values(sample_features, mock_models):
    bad_df = sample_features.copy()
    bad_df.loc[0, "quantity_ordered"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        predict.predict_purchase_order_delays(bad_df)


def test_prediction_rejects_mismatched_model_features(sample_features, sample_artifacts, monkeypatch):
    clf, reg = sample_artifacts
    reg["feature_columns"] = ["supplier_id"]
    monkeypatch.setattr(predict, "load_prediction_models", lambda: (clf, reg))

    with pytest.raises(ValueError, match="do not match"):
        predict.predict_purchase_order_delays(sample_features)