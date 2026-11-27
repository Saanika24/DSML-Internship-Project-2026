from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.prediction.feature_engineering import (
    build_purchase_order_features,
    prepare_prediction_inputs,
)
from src.prediction.train import split_by_date
from src.prediction.predict import predict_purchase_order_delays


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "raw" / "purchase_orders.csv"
OUTPUT_DIR = ROOT_DIR / "models_error_analysis"

COMBINATIONS = {
    "old_classifier_old_regressor": {
        "classifier_dir": "models",
        "regressor_dir": "models_conditional_regression",
    },
    "old_classifier_new_regressor": {
        "classifier_dir": "models",
        "regressor_dir": "models_regression_experiment",
    },
    "new_classifier_old_regressor": {
        "classifier_dir": "models_classifier_experiment",
        "regressor_dir": "models_conditional_regression",
    },
    "new_classifier_new_regressor": {
        "classifier_dir": "models_classifier_experiment",
        "regressor_dir": "models_regression_experiment",
    },
}


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_actual_delay": float(np.mean(y_true)),
        "mean_predicted_delay": float(np.mean(y_pred)),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    raw_df = pd.read_csv(DATA_PATH)

    print("Engineering point-in-time features...")
    df = build_purchase_order_features(raw_df)

    X, y_classification, y_delay = prepare_prediction_inputs(df)

    train_df, val_df, test_df = split_by_date(df)

    X_train = X.loc[train_df.index]
    X_val = X.loc[val_df.index]

    y_class_train = y_classification.loc[train_df.index].astype(int)
    y_class_val = y_classification.loc[val_df.index].astype(int)

    y_delay_train = y_delay.loc[train_df.index].astype(float)
    y_delay_val = y_delay.loc[val_df.index].astype(float)

    if not np.isfinite(y_delay_train).all():
        raise ValueError("Training delay targets contain non-finite values.")

    if not np.isfinite(y_delay_val).all():
        raise ValueError("Validation delay targets contain non-finite values.")

    print("\n========== DATA SPLIT ==========")
    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(val_df)}")
    print(f"Reserved test records: {len(test_df)}")
    print(f"Training delayed orders: {int(y_class_train.sum())}")
    print(f"Validation delayed orders: {int(y_class_val.sum())}")
    print(f"Training mean delay across all orders: {y_delay_train.mean():.4f}")
    print(f"Training median delay across all orders: {y_delay_train.median():.4f}")

    # ---------------------------------------------------------
    # PART A: Simple baselines calculated from training only
    # ---------------------------------------------------------

    baseline_values = {
        "always_predict_zero": 0.0,
        "training_mean_delay": float(y_delay_train.mean()),
        "training_median_delay": float(y_delay_train.median()),
    }

    baseline_rows = []

    print("\n========== SIMPLE BASELINE COMPARISON ==========")

    for name, constant in baseline_values.items():
        predictions = np.full(len(y_delay_val), constant, dtype=float)
        metrics = regression_metrics(y_delay_val.to_numpy(), predictions)

        row = {
            "method": name,
            "constant_prediction_days": constant,
            **metrics,
        }
        baseline_rows.append(row)

        print(f"\n{name}")
        print(f"Constant prediction: {constant:.4f} days")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R2: {metrics['r2']:.4f}")
        print(f"Mean signed error: {metrics['mean_signed_error']:.4f}")

    # ---------------------------------------------------------
    # PART B: Integrated model combinations
    # ---------------------------------------------------------

    pipeline_rows = []
    detail_frames = []

    print("\n========== INTEGRATED MODEL ANALYSIS ==========")

    for name, paths in COMBINATIONS.items():
        print(f"\nAnalyzing: {name}")

        predictions = predict_purchase_order_delays(
            X_val,
            model_dir=paths["classifier_dir"],
            regressor_dir=paths["regressor_dir"],
        )

        predicted_class = (
            predictions["predicted_status"] == "Delayed"
        ).astype(int)

        predicted_delay = predictions[
            "estimated_delay_days"
        ].astype(float)

        delay_probability = predictions[
            "delay_probability"
        ].astype(float)

        if not np.isfinite(predicted_delay).all():
            raise ValueError(f"{name} produced non-finite delay predictions.")

        if not np.isfinite(delay_probability).all():
            raise ValueError(f"{name} produced non-finite probabilities.")

        actual_class = y_class_val
        actual_delay = y_delay_val

        tn = int(((actual_class == 0) & (predicted_class == 0)).sum())
        fp = int(((actual_class == 0) & (predicted_class == 1)).sum())
        fn = int(((actual_class == 1) & (predicted_class == 0)).sum())
        tp = int(((actual_class == 1) & (predicted_class == 1)).sum())

        metrics = regression_metrics(
            actual_delay.to_numpy(),
            predicted_delay.to_numpy(),
        )

        detected_mask = (actual_class == 1) & (predicted_class == 1)
        actual_delayed_mask = actual_class == 1

        detected_severity_mae = (
            mean_absolute_error(
                actual_delay.loc[detected_mask],
                predicted_delay.loc[detected_mask],
            )
            if detected_mask.any()
            else np.nan
        )

        actual_delayed_pipeline_mae = mean_absolute_error(
            actual_delay.loc[actual_delayed_mask],
            predicted_delay.loc[actual_delayed_mask],
        )

        row = {
            "combination": name,
            **metrics,
            "accuracy": accuracy_score(actual_class, predicted_class),
            "precision": precision_score(
                actual_class, predicted_class, zero_division=0
            ),
            "recall": recall_score(
                actual_class, predicted_class, zero_division=0
            ),
            "f1": f1_score(
                actual_class, predicted_class, zero_division=0
            ),
            "actual_delayed_pipeline_mae": actual_delayed_pipeline_mae,
            "detected_severity_mae": detected_severity_mae,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "predicted_delayed_orders": int(predicted_class.sum()),
        }

        pipeline_rows.append(row)

        print(f"MAE: {metrics['mae']:.4f} days")
        print(f"RMSE: {metrics['rmse']:.4f} days")
        print(f"R2: {metrics['r2']:.4f}")
        print(f"Mean signed error: {metrics['mean_signed_error']:.4f}")
        print(f"Precision: {row['precision']:.4f}")
        print(f"Recall: {row['recall']:.4f}")
        print(f"F1: {row['f1']:.4f}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"True positives: {tp}")
        print(f"True negatives: {tn}")

        # Build detailed validation predictions.
        details = pd.DataFrame(index=X_val.index)

        details["combination"] = name
        details["actual_delayed"] = actual_class
        details["predicted_delayed"] = predicted_class
        details["delay_probability"] = delay_probability
        details["actual_delay_days"] = actual_delay
        details["predicted_delay_days"] = predicted_delay

        details["signed_error_days"] = (
            details["predicted_delay_days"] -
            details["actual_delay_days"]
        )

        details["absolute_error_days"] = (
            details["signed_error_days"].abs()
        )

        details["error_type"] = np.select(
            [
                (actual_class == 0) & (predicted_class == 0),
                (actual_class == 0) & (predicted_class == 1),
                (actual_class == 1) & (predicted_class == 0),
                (actual_class == 1) & (predicted_class == 1),
            ],
            ["true_negative", "false_positive", "false_negative", "true_positive"],
            default="unknown",
        )

        details["actual_delay_bucket"] = pd.cut(
            details["actual_delay_days"],
            bins=[0, 2, 5, 10, np.inf],
            labels=["1-2 days", "3-5 days", "6-10 days", "11+ days"],
            right=True,
        )

        details["order_date"] = val_df["order_date"]

        if "po_id" in val_df.columns:
            details["po_id"] = val_df["po_id"]

        detail_frames.append(details.reset_index(names="row_index"))

    # ---------------------------------------------------------
    # PART C: Save baseline and pipeline comparison
    # ---------------------------------------------------------

    baseline_comparison = pd.DataFrame(baseline_rows)
    pipeline_comparison = pd.DataFrame(pipeline_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = OUTPUT_DIR / "baseline_comparison.csv"
    pipeline_path = OUTPUT_DIR / "pipeline_comparison.csv"

    baseline_comparison.to_csv(baseline_path, index=False)
    pipeline_comparison.to_csv(pipeline_path, index=False)

    print("\n========== BASELINE RESULTS ==========")
    print(
        baseline_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n========== PIPELINE RESULTS ==========")
    print(
        pipeline_comparison.sort_values(
            ["mae", "f1"],
            ascending=[True, False],
        ).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ---------------------------------------------------------
    # PART D: Error breakdown by classification outcome
    # ---------------------------------------------------------

    all_details = pd.concat(detail_frames, ignore_index=True)

    error_group = (
        all_details.groupby(
            ["combination", "error_type"],
            observed=False,
        )
        .agg(
            order_count=("actual_delay_days", "size"),
            mean_actual_delay=("actual_delay_days", "mean"),
            mean_predicted_delay=("predicted_delay_days", "mean"),
            mae=("absolute_error_days", "mean"),
            mean_signed_error=("signed_error_days", "mean"),
        )
        .reset_index()
    )

    error_group_path = OUTPUT_DIR / "error_by_classification_group.csv"
    error_group.to_csv(error_group_path, index=False)

    print("\n========== ERROR BY CLASSIFICATION GROUP ==========")
    print(
        error_group.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ---------------------------------------------------------
    # PART E: Performance by actual delay severity
    # ---------------------------------------------------------

    delayed_details = all_details[
        all_details["actual_delayed"] == 1
    ].copy()

    severity_rows = []

    for (combination, bucket), group in delayed_details.groupby(
        ["combination", "actual_delay_bucket"],
        observed=False,
    ):
        if group.empty:
            continue

        detected = group[group["predicted_delayed"] == 1]

        severity_rows.append({
            "combination": combination,
            "actual_delay_bucket": str(bucket),
            "actual_orders": len(group),
            "detected_orders": len(detected),
            "missed_orders": int((group["predicted_delayed"] == 0).sum()),
            "recall": float(group["predicted_delayed"].mean()),
            "end_to_end_mae": mean_absolute_error(
                group["actual_delay_days"],
                group["predicted_delay_days"],
            ),
            "detected_only_severity_mae": (
                mean_absolute_error(
                    detected["actual_delay_days"],
                    detected["predicted_delay_days"],
                )
                if not detected.empty
                else np.nan
            ),
        })

    severity_comparison = pd.DataFrame(severity_rows)

    severity_path = OUTPUT_DIR / "delay_severity_analysis.csv"
    severity_comparison.to_csv(severity_path, index=False)

    print("\n========== PERFORMANCE BY DELAY SEVERITY ==========")
    print(
        severity_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ---------------------------------------------------------
    # PART F: Save every validation prediction for inspection
    # ---------------------------------------------------------

    details_path = OUTPUT_DIR / "pipeline_error_details.csv"
    all_details.to_csv(details_path, index=False)

    print("\n========== SAVED FILES ==========")
    print(baseline_path)
    print(pipeline_path)
    print(error_group_path)
    print(severity_path)
    print(details_path)

    print("\nAnalysis completed successfully.")
    print("No model was trained or overwritten.")
    print("Reserved test data was not used.")
    print("All baseline constants were calculated from training data only.")


if __name__ == "__main__":
    main()
