
import pandas as pd
import pytest

from src.validation import validate_required_columns


def test_all_required_columns_exist():

    data = pd.DataFrame({
        "supplier_id": [1, 2, 3],
        "delay_days": [2, 0, 5],
        "shipment_cost": [100, 200, 150]
    })

    required_columns = [
        "supplier_id",
        "delay_days",
        "shipment_cost"
    ]

    assert validate_required_columns(
        data, required_columns
    ) is True


def test_missing_required_column():

    data = pd.DataFrame({
        "supplier_id": [1, 2],
        "delay_days": [2, 4]
    })

    required_columns = [
        "supplier_id",
        "delay_days",
        "shipment_cost"
    ]

    with pytest.raises(
        ValueError,
        match="Missing required columns"
    ):
        validate_required_columns(
            data, required_columns
        )


def test_empty_required_columns():

    data = pd.DataFrame({
        "supplier_id": [1, 2]
    })

    assert validate_required_columns(
        data, []
    ) is True


def test_missing_multiple_columns():

    data = pd.DataFrame({
        "supplier_id": [1, 2]
    })

    required_columns = [
        "delay_days",
        "shipment_cost"
    ]

    with pytest.raises(ValueError) as error:
        validate_required_columns(
            data, required_columns
        )

    assert "delay_days" in str(error.value)
    assert "shipment_cost" in str(error.value)
