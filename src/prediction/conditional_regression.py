from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.prediction.evaluation import regression_metrics
from src.prediction.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    build_purchase_order_features,
    prepare_prediction_inputs,
)
from src.prediction.model_utils import save_model
from src.prediction.train import build_pipeline, split_by_date


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "purchase_orders.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "models_conditional_regression"


def train_conditional_regressor(
    data_path=DEFAULT_DATA_PATH,
    model_dir=DEFAULT_MODEL_DIR,
    random_state=42,
):
    data_path = Path(data_path)
    model_dir = Path(model_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print("Loading data...")
    raw_df = pd.read_csv(data_path)

    print("Engineering features...")
    df = build_purchase_order_features(raw_df)
    X, _, y_reg = prepare_prediction_inputs(df)

    feature_columns = MODEL_FEATURE_COLUMNS.copy()
    X = X[feature_columns]

    # Preserve the same chronological split as the existing pipeline.
    train_df, val_df, test_df = split_by_date(df)

    X_train = X.loc[train_df.index]
    X_val = X.loc[val_df.index]
    X_test = X.loc[test_df.index]

    delay_train = y_reg.loc[train_df.index]
    delay_val = y_reg.loc[val_df.index]
    delay_test = y_reg.loc[test_df.index]

    # Conditional regression: train only on delayed training orders.
    train_mask = delay_train > 0
    val_mask = delay_val > 0
    test_mask = delay_test > 0

    X_train_delayed = X_train.loc[train_mask]
    y_train_delayed = delay_train.loc[train_mask]

    X_val_delayed = X_val.loc[val_mask]
    y_val_delayed = delay_val.loc[val_mask]

    X_test_delayed = X_test.loc[test_mask]
    y_test_delayed = delay_test.loc[test_mask]

    if len(X_train_delayed) < 2:
        raise ValueError("Not enough delayed training orders.")

    if X_val_delayed.empty:
        raise ValueError("No delayed validation orders available.")

    if X_test_delayed.empty:
        raise ValueError("No delayed test orders available.")

    print("\nDataset sizes:")
    print(f"Delayed training orders:   {len(X_train_delayed)}")
    print(f"Delayed validation orders: {len(X_val_delayed)}")
    print(f"Delayed test orders:       {len(X_test_delayed)}")

    # Baseline: median delay among delayed training orders only.
    median_delay = float(y_train_delayed.median())

    baseline_val_preds = np.full(
        len(y_val_delayed),
        median_delay,
    )

    baseline_val_mae = regression_metrics(
        y_val_delayed,
        baseline_val_preds,
    )["mae"]

    print(f"\nTraining median baseline: {median_delay:.2f} days")
    print(f"Baseline validation MAE:  {baseline_val_mae:.4f}")

    # Train XGBoost only on delayed training orders.
    print("\nTraining conditional XGBoost regressor...")

    regressor = build_pipeline(
        XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),
        X_train_delayed,
    )

    regressor.fit(X_train_delayed, y_train_delayed)

    val_predictions = np.maximum(
        regressor.predict(X_val_delayed),
        0,
    )

    xgb_val_mae = regression_metrics(
        y_val_delayed,
        val_predictions,
    )["mae"]

    print(f"XGBoost validation MAE:    {xgb_val_mae:.4f}")
    print(f"Median baseline MAE:       {baseline_val_mae:.4f}")

    # Select using validation MAE only.
    if xgb_val_mae <= baseline_val_mae:
        selected_model = regressor
        selected_name = "Conditional XGBoost"
        artifact = {
            "model": regressor,
            "feature_columns": feature_columns,
            "target": "delay_days",
            "model_type": "conditional_regression",
            "model_name": selected_name,
            "conditional_on_delay": True,
        }
        test_predictions = np.maximum(
            selected_model.predict(X_test_delayed),
            0,
        )
    else:
        selected_model = None
        selected_name = "Conditional median baseline"
        artifact = {
            "model": None,
            "feature_columns": feature_columns,
            "target": "delay_days",
            "model_type": "conditional_regression",
            "model_name": selected_name,
            "conditional_on_delay": True,
            "constant_prediction": median_delay,
        }
        test_predictions = np.full(
            len(X_test_delayed),
            median_delay,
        )

    # Final evaluation on delayed test orders only.
    test_results = regression_metrics(
        y_test_delayed,
        test_predictions,
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_model(
        artifact,
        model_dir / "conditional_delay_regressor.joblib",
    )

    print(f"\nSelected model: {selected_name}")
    print("\nConditional regression test results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value}")

    print(f"\nSaved experimental model to: {model_path}")

    return {
        "model_name": selected_name,
        "validation_xgboost_mae": xgb_val_mae,
        "validation_baseline_mae": baseline_val_mae,
        "test_metrics": test_results,
        "model_path": model_path,
        "delayed_train_records": len(X_train_delayed),
        "delayed_validation_records": len(X_val_delayed),
        "delayed_test_records": len(X_test_delayed),
    }


if __name__ == "__main__":
    train_conditional_regressor()
