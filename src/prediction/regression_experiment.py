from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.prediction.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    build_purchase_order_features,
    prepare_prediction_inputs,
)
from src.prediction.preprocessing import build_preprocessor
from src.prediction.train import split_by_date
from src.prediction.model_utils import save_model


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "purchase_orders.csv"
OUTPUT_DIR = ROOT / "models_regression_experiment"

FEATURES = MODEL_FEATURE_COLUMNS.copy()

CONFIGURATIONS = {
    "current_baseline": {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
    "shallow": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
    },
    "regularized": {
        "n_estimators": 350,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
    },
    "more_trees_lower_rate": {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
    },
    "moderate_depth": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.04,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
    },
}


def build_regressor(X_train, config):
    model = XGBRegressor(
        **config,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", model),
    ])


def calculate_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
    }


def run_regression_experiment():
    print("Loading dataset...")
    raw = pd.read_csv(DATA_PATH)

    print("Engineering features...")
    df = build_purchase_order_features(raw)

    X, _, y_delay = prepare_prediction_inputs(df)

    train_df, val_df, test_df = split_by_date(df)

    # Only previously completed delayed orders are used
    # to train the conditional delay-duration model.
    train_mask = y_delay.loc[train_df.index] > 0
    val_mask = y_delay.loc[val_df.index] > 0

    X_train = X.loc[train_df.index].loc[train_mask, FEATURES]
    y_train = y_delay.loc[train_df.index].loc[train_mask]

    X_val = X.loc[val_df.index].loc[val_mask, FEATURES]
    y_val = y_delay.loc[val_df.index].loc[val_mask]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("No delayed orders available for training or validation.")

    print("\nDataset:")
    print("Training delayed orders:", len(X_train))
    print("Validation delayed orders:", len(X_val))
    print("Test records reserved:", len(test_df))

    baseline_prediction = float(y_train.median())
    baseline_predictions = np.full(
        len(y_val),
        baseline_prediction,
    )

    baseline_metrics = calculate_metrics(
        y_val.to_numpy(),
        baseline_predictions,
    )

    print("\n========== MEDIAN BASELINE ==========")
    print("Training median:", round(baseline_prediction, 4))
    print("Validation MAE:", round(baseline_metrics["mae"], 4))
    print("Validation RMSE:", round(baseline_metrics["rmse"], 4))
    print("Validation R2:", round(baseline_metrics["r2"], 4))

    results = []
    best_model = None
    best_name = None
    best_metrics = None
    best_config = None

    for name, config in CONFIGURATIONS.items():
        print(f"\nTraining configuration: {name}")

        model = build_regressor(X_train, config)
        model.fit(X_train, y_train)

        predictions = np.maximum(
            model.predict(X_val),
            0,
        )

        metrics = calculate_metrics(
            y_val.to_numpy(),
            predictions,
        )

        print("Validation MAE:", round(metrics["mae"], 4))
        print("Validation RMSE:", round(metrics["rmse"], 4))
        print("Validation R2:", round(metrics["r2"], 4))

        results.append({
            "configuration": name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
        })

        if (
            best_metrics is None
            or metrics["mae"] < best_metrics["mae"]
            or (
                np.isclose(metrics["mae"], best_metrics["mae"])
                and metrics["rmse"] < best_metrics["rmse"]
            )
        ):
            best_model = model
            best_name = name
            best_metrics = metrics
            best_config = config

    comparison = pd.DataFrame(results).sort_values(
        ["mae", "rmse"],
        ascending=[True, True],
    )

    print("\n========== VALIDATION COMPARISON ==========")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n========== BEST MODEL ==========")
    print("Configuration:", best_name)
    print("Validation MAE:", round(best_metrics["mae"], 4))
    print("Validation RMSE:", round(best_metrics["rmse"], 4))
    print("Validation R2:", round(best_metrics["r2"], 4))

    artifact = {
        "model": best_model,
        "feature_columns": FEATURES,
        "target": "delay_days",
        "model_type": "regression",
        "model_name": f"Conditional XGBoost - {best_name}",
        "conditional_on_delay": True,
        "training_config": best_config,
        "validation_mae": best_metrics["mae"],
        "validation_rmse": best_metrics["rmse"],
        "validation_r2": best_metrics["r2"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = save_model(
        artifact,
        OUTPUT_DIR / "conditional_delay_regressor.joblib",
    )

    comparison.to_csv(
        OUTPUT_DIR / "validation_comparison.csv",
        index=False,
    )

    print("\nSaved experiment model:", model_path)
    print("Saved comparison:", OUTPUT_DIR / "validation_comparison.csv")
    print("\nExisting conditional regressor was not modified.")
    print("Test data was not used for model selection.")


if __name__ == "__main__":
    run_regression_experiment()
