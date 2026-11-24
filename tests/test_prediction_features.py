
import pandas as pd
import pytest

from src.prediction.feature_engineering import (
    build_purchase_order_features,
    prepare_prediction_inputs,
    validate_prediction_columns,
)


@pytest.fixture
def sample_orders():
    return pd.DataFrame({
        "po_id": ["PO1", "PO2", "PO3", "PO4"],
        "supplier_id": ["S1", "S1", "S1", "S2"],
        "sku_id": ["SKU1", "SKU2", "SKU1", "SKU3"],
        "warehouse": ["W1", "W1", "W2", "W1"],
        "order_date": [
            "2025-01-01",
            "2025-01-01",
            "2025-01-05",
            "2025-01-03",
        ],
        "promised_delivery_date": [
            "2025-01-06",
            "2025-01-07",
            "2025-01-10",
            "2025-01-08",
        ],
        "actual_delivery_date": [
            "2025-01-06",
            "2025-01-09",
            "2025-01-12",
            "2025-01-07",
        ],
        "quantity_ordered": [100, 150, 200, 120],
        "quantity_received": [100, 150, 200, 120],
        "unit_cost": [20.0, 25.0, 22.0, 18.0],
        "is_on_time": [1, 0, 0, 1],
        "days_late": [0, 2, 2, 0],
        "po_status": ["completed"] * 4,
        "quality_accepted": [1, 1, 1, 1],
    })


def test_missing_columns_are_detected(sample_orders):
    invalid_data = sample_orders.drop(columns=["supplier_id"])

    with pytest.raises(ValueError, match="supplier_id"):
        validate_prediction_columns(invalid_data)


def test_feature_engineering_creates_expected_columns(sample_orders):
    result = build_purchase_order_features(sample_orders)

    expected_columns = [
        "planned_lead_time_days",
        "order_month",
        "order_day_of_week",
        "supplier_prior_order_count",
        "supplier_prior_on_time_rate",
        "supplier_prior_mean_days_late",
        "is_delayed",
        "delay_days",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_same_day_orders_do_not_leak_into_each_other(sample_orders):
    result = build_purchase_order_features(sample_orders)

    first_day_orders = result[
        result["order_date"] == pd.Timestamp("2025-01-01")
    ]

    assert (
        first_day_orders["supplier_prior_order_count"] == 0
    ).all()


def test_later_order_uses_only_previous_supplier_history(sample_orders):
    result = build_purchase_order_features(sample_orders)

    later_order = result.loc[result["po_id"] == "PO3"].iloc[0]

    assert later_order["supplier_prior_order_count"] == 2
    assert later_order["supplier_prior_on_time_rate"] == 0.5
    assert later_order["supplier_prior_mean_days_late"] == 1.0


def test_prediction_inputs_exclude_leakage_columns(sample_orders):
    result = build_purchase_order_features(sample_orders)

    X, y_class, y_reg = prepare_prediction_inputs(result)

    assert "actual_delivery_date" not in X.columns
    assert "days_late" not in X.columns
    assert "is_on_time" not in X.columns
    assert "quality_accepted" not in X.columns

    assert len(X) == len(y_class) == len(y_reg)


def test_classification_target_is_correct(sample_orders):
    result = build_purchase_order_features(sample_orders)

    assert result["is_delayed"].tolist() == [0, 1, 1, 0]


def test_invalid_delay_values_are_rejected(sample_orders):
    sample_orders.loc[0, "days_late"] = -2

    with pytest.raises(ValueError, match="negative"):
        build_purchase_order_features(sample_orders)


def test_inconsistent_target_values_are_rejected(sample_orders):
    sample_orders.loc[0, "is_on_time"] = 0

    with pytest.raises(ValueError, match="inconsistent"):
        build_purchase_order_features(sample_orders)
