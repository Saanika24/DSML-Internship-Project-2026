import pandas as pd


REQUIRED_COLUMNS = [
    "supplier_id",
    "sku_id",
    "warehouse",
    "order_date",
    "promised_delivery_date",
    "actual_delivery_date",
    "quantity_ordered",
    "unit_cost",
    "is_on_time",
    "days_late",
]

# Features strictly available at the time a purchase order is placed
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


def validate_prediction_columns(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    if df.empty:
        raise ValueError("Purchase-order dataset is empty.")


def build_purchase_order_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_prediction_columns(df)
    out = df.copy().reset_index(drop=True)

    # Parse core dates
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    out["promised_delivery_date"] = pd.to_datetime(out["promised_delivery_date"], errors="coerce")

    if out["order_date"].isna().any():
        raise ValueError("Invalid or missing order_date values found.")

    if out["promised_delivery_date"].isna().any():
        raise ValueError("Invalid or missing promised_delivery_date values found.")

    # Missing actual_delivery_date is allowed for open orders, but malformed strings should fail
    raw_actual = out["actual_delivery_date"]
    out["actual_delivery_date"] = pd.to_datetime(raw_actual, errors="coerce")

    invalid_actual = (
        raw_actual.notna()
        & raw_actual.astype(str).str.strip().ne("")
        & out["actual_delivery_date"].isna()
    )
    if invalid_actual.any():
        raise ValueError("Invalid actual_delivery_date values found.")

    if out["supplier_id"].isna().any():
        raise ValueError("Missing supplier_id values found.")

    # Validate numeric fields and target consistency
    for col in ["quantity_ordered", "unit_cost", "days_late"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"Invalid or missing values in {col}.")

    out["is_on_time"] = pd.to_numeric(out["is_on_time"], errors="coerce")
    if out["is_on_time"].isna().any():
        raise ValueError("Invalid or missing is_on_time values.")

    if not out["is_on_time"].isin([0, 1]).all():
        raise ValueError("is_on_time must contain only 0 and 1.")

    if (out["days_late"] < 0).any():
        raise ValueError("days_late cannot contain negative values.")

    expected_on_time = (out["days_late"] == 0).astype(int)
    if not out["is_on_time"].eq(expected_on_time).all():
        raise ValueError("is_on_time and days_late contain inconsistent labels.")

    # Order-time calendar & lead-time features
    out["order_month"] = out["order_date"].dt.month
    out["order_day_of_week"] = out["order_date"].dt.day_of_week
    out["order_quarter"] = out["order_date"].dt.quarter
    out["order_is_weekend"] = (out["order_date"].dt.day_of_week >= 5).astype(int)

    out["planned_lead_time_days"] = (out["promised_delivery_date"] - out["order_date"]).dt.days
    if (out["planned_lead_time_days"] < 0).any():
        raise ValueError("Promised delivery date cannot precede order date.")

    # Point-in-time supplier history: only count orders placed AND delivered strictly
    # before the current order date to avoid leakage and same-day ambiguity.
    history_cols = [
        "supplier_prior_order_count",
        "supplier_prior_on_time_rate",
        "supplier_prior_mean_days_late",
        "supplier_prior_late_order_count",
    ]
    for col in history_cols:
        out[col] = 0.0

    for _, group in out.groupby("supplier_id", sort=False):
        order_dates = group["order_date"]
        delivery_dates = group["actual_delivery_date"]

        for idx, row in group.iterrows():
            curr_date = row["order_date"]
            prior_mask = (
                (order_dates < curr_date)
                & (delivery_dates < curr_date)
                & delivery_dates.notna()
            )
            prior_orders = group.loc[prior_mask]

            if prior_orders.empty:
                continue

            out.at[idx, "supplier_prior_order_count"] = len(prior_orders)
            out.at[idx, "supplier_prior_on_time_rate"] = prior_orders["is_on_time"].mean()
            out.at[idx, "supplier_prior_mean_days_late"] = prior_orders["days_late"].mean()
            out.at[idx, "supplier_prior_late_order_count"] = (prior_orders["days_late"] > 0).sum()

    # Targets
    out["is_delayed"] = (out["days_late"] > 0).astype(int)
    out["delay_days"] = out["days_late"]

    return out


def prepare_prediction_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    missing = [col for col in MODEL_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")

    if not {"is_delayed", "delay_days"}.issubset(df.columns):
        raise ValueError("Prediction target columns are missing.")

    X = df[MODEL_FEATURE_COLUMNS].copy()
    y_clf = df["is_delayed"].copy()
    y_reg = df["delay_days"].copy()

    return X, y_clf, y_reg