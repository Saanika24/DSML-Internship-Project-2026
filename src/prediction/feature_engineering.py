
"""Feature engineering for supply-chain delivery predictions.

Creates order-time features and leakage-safe supplier history features.
"""

from typing import Tuple

import pandas as pd


REQUIRED_COLUMNS = [
    "supplier_id",
    "sku_id",
    "warehouse",
    "order_date",
    "promised_delivery_date",
    "quantity_ordered",
    "unit_cost",
    "is_on_time",
    "days_late",
]

# Only features that can be known when the purchase order is placed.
MODEL_FEATURE_COLUMNS = [
    "supplier_id",
    "sku_id",
    "warehouse",
    "quantity_ordered",
    "unit_cost",
    "order_month",
    "order_day_of_week",
    "order_quarter",
    "order_is_weekend",
    "planned_lead_time_days",
    "supplier_prior_order_count",
    "supplier_prior_on_time_rate",
    "supplier_prior_mean_days_late",
    "supplier_prior_late_order_count",
]


def validate_prediction_columns(data: pd.DataFrame) -> None:
    """Check that the purchase-order dataset has required columns."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required prediction columns: {missing}"
        )

    if data.empty:
        raise ValueError("Purchase-order dataset is empty.")


def build_purchase_order_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create prediction features without using future information.

    Supplier historical metrics are calculated using strictly earlier
    order dates. Orders from the same supplier on the same date do not
    use each other's delivery outcomes.

    Actual delivery dates and final delivery outcomes are retained
    only as source/target information, never as model input features.
    """

    validate_prediction_columns(data)

    result = data.copy().reset_index(drop=True)

    # Convert date columns into a consistent format.
    result["order_date"] = pd.to_datetime(
        result["order_date"], errors="coerce"
    )
    result["promised_delivery_date"] = pd.to_datetime(
        result["promised_delivery_date"], errors="coerce"
    )

    if result["order_date"].isna().any():
        raise ValueError("Invalid or missing order_date values found.")

    if result["promised_delivery_date"].isna().any():
        raise ValueError(
            "Invalid or missing promised_delivery_date values found."
        )

    if result["supplier_id"].isna().any():
        raise ValueError("Missing supplier_id values found.")

    # Validate numerical fields.
    for column in ["quantity_ordered", "unit_cost", "days_late"]:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

        if result[column].isna().any():
            raise ValueError(
                f"Invalid or missing values in {column}."
            )

    result["is_on_time"] = pd.to_numeric(
        result["is_on_time"], errors="coerce"
    )

    if result["is_on_time"].isna().any():
        raise ValueError("Invalid or missing is_on_time values.")

    if not result["is_on_time"].isin([0, 1]).all():
        raise ValueError("is_on_time must contain only 0 and 1.")

    if (result["days_late"] < 0).any():
        raise ValueError("days_late cannot contain negative values.")

    # Check whether the two target columns agree.
    expected_on_time = (result["days_late"] == 0).astype(int)

    if not result["is_on_time"].eq(expected_on_time).all():
        raise ValueError(
            "is_on_time and days_late contain inconsistent labels."
        )

    # --------------------------------------------------
    # 1. Calendar-based features
    # --------------------------------------------------

    result["order_month"] = result["order_date"].dt.month
    result["order_day_of_week"] = (
        result["order_date"].dt.dayofweek
    )
    result["order_quarter"] = (
        result["order_date"].dt.quarter
    )
    result["order_is_weekend"] = (
        result["order_date"].dt.dayofweek >= 5
    ).astype(int)

    # Planned lead time is known at order placement.
    result["planned_lead_time_days"] = (
        result["promised_delivery_date"] - result["order_date"]
    ).dt.days

    if (result["planned_lead_time_days"] < 0).any():
        raise ValueError(
            "Promised delivery date cannot precede order date."
        )

    # --------------------------------------------------
    # 2. Leakage-safe supplier history
    # --------------------------------------------------

    result["_order_day"] = result["order_date"].dt.normalize()

    daily_history = (
        result.groupby(
            ["supplier_id", "_order_day"],
            as_index=False,
            sort=False,
        )
        .agg(
            daily_order_count=("supplier_id", "size"),
            daily_on_time_count=("is_on_time", "sum"),
            daily_days_late_sum=("days_late", "sum"),
            daily_late_order_count=(
                "days_late",
                lambda values: (values > 0).sum(),
            ),
        )
        .sort_values(["supplier_id", "_order_day"])
        .reset_index(drop=True)
    )

    history_columns = [
        "daily_order_count",
        "daily_on_time_count",
        "daily_days_late_sum",
        "daily_late_order_count",
    ]

    # Cumulative totals minus current-day totals ensure that
    # only strictly earlier supplier orders are considered.
    for column in history_columns:
        daily_history[f"prior_{column}"] = (
            daily_history.groupby("supplier_id")[column].cumsum()
            - daily_history[column]
        )

    daily_history["supplier_prior_order_count"] = (
        daily_history["prior_daily_order_count"]
    )

    prior_count = daily_history[
        "supplier_prior_order_count"
    ].replace(0, pd.NA)

    daily_history["supplier_prior_on_time_rate"] = (
        daily_history["prior_daily_on_time_count"] / prior_count
    ).fillna(0.0)

    daily_history["supplier_prior_mean_days_late"] = (
        daily_history["prior_daily_days_late_sum"] / prior_count
    ).fillna(0.0)

    daily_history["supplier_prior_late_order_count"] = (
        daily_history["prior_daily_late_order_count"]
    )

    # Merge the historical metrics back into the original rows.
    history_features = daily_history[
        [
            "supplier_id",
            "_order_day",
            "supplier_prior_order_count",
            "supplier_prior_on_time_rate",
            "supplier_prior_mean_days_late",
            "supplier_prior_late_order_count",
        ]
    ]

    result = result.merge(
        history_features,
        on=["supplier_id", "_order_day"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    result = result.drop(columns=["_order_day"])

    # --------------------------------------------------
    # 3. Prediction targets
    # --------------------------------------------------

    result["is_delayed"] = (
        result["days_late"] > 0
    ).astype(int)

    result["delay_days"] = result["days_late"]

    return result


def prepare_prediction_inputs(
    engineered_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate safe model features from prediction targets.

    Returns:
        X: Model input features.
        y_classification: Delayed/on-time target.
        y_regression: Number of delay days.
    """

    missing = [
        column for column in MODEL_FEATURE_COLUMNS
        if column not in engineered_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing engineered features: {missing}"
        )

    required_targets = ["is_delayed", "delay_days"]

    if not all(
        column in engineered_data.columns
        for column in required_targets
    ):
        raise ValueError("Prediction target columns are missing.")

    X = engineered_data[MODEL_FEATURE_COLUMNS].copy()

    y_classification = engineered_data["is_delayed"].copy()
    y_regression = engineered_data["delay_days"].copy()

    return X, y_classification, y_regression
