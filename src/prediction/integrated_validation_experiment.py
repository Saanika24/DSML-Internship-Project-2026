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
OUTPUT_DIR = ROOT_DIR / "models_integrated_validation_experiment"

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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    raw_df = pd.read_csv(DATA_PATH)

    print("Engineering features...")
    df = build_purchase_order_features(raw_df)

    X, y_classification, y_delay = prepare_prediction_inputs(df)

    train_df, val_df, test_df = split_by_date(df)

    X_val = X.loc[val_df.index]
    y_class_val = y_classification.loc[val_df.index].astype(int)
    y_delay_val = y_delay.loc[val_df.index].astype(float)

    print("\n========== VALIDATION DATA ==========")
    print(f"Training records:   {len(train_df)}")
    print(f"Validation records: {len(val_df)}")
    print(f"Reserved test records: {len(test_df)}")
    print(f"Actual delayed validation orders: {int(y_class_val.sum())}")

    results = []

    for name, paths in COMBINATIONS.items():
        print(f"\n\n========== {name} ==========")

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

        tn = int(((y_class_val == 0) & (predicted_class == 0)).sum())
        fp = int(((y_class_val == 0) & (predicted_class == 1)).sum())
        fn = int(((y_class_val == 1) & (predicted_class == 0)).sum())
        tp = int(((y_class_val == 1) & (predicted_class == 1)).sum())

        accuracy = accuracy_score(y_class_val, predicted_class)
        precision = precision_score(
            y_class_val, predicted_class, zero_division=0
        )
        recall = recall_score(
            y_class_val, predicted_class, zero_division=0
        )
        f1 = f1_score(
            y_class_val, predicted_class, zero_division=0
        )

        end_to_end_mae = mean_absolute_error(
            y_delay_val, predicted_delay
        )
        end_to_end_rmse = np.sqrt(
            mean_squared_error(y_delay_val, predicted_delay)
        )
        end_to_end_r2 = r2_score(
            y_delay_val, predicted_delay
        )

        actual_delayed_mask = y_class_val == 1
        actual_delayed_mae = mean_absolute_error(
            y_delay_val.loc[actual_delayed_mask],
            predicted_delay.loc[actual_delayed_mask],
        )

        correctly_detected_delayed_mask = (
            (y_class_val == 1) & (predicted_class == 1)
        )

        if correctly_detected_delayed_mask.any():
            detected_delayed_mae = mean_absolute_error(
                y_delay_val.loc[correctly_detected_delayed_mask],
                predicted_delay.loc[correctly_detected_delayed_mask],
            )
        else:
            detected_delayed_mae = np.nan

        row = {
            "combination": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "end_to_end_mae": end_to_end_mae,
            "end_to_end_rmse": end_to_end_rmse,
            "end_to_end_r2": end_to_end_r2,
            "actual_delayed_mae_in_pipeline": actual_delayed_mae,
            "detected_delayed_severity_mae": detected_delayed_mae,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "predicted_delayed_orders": int(predicted_class.sum()),
        }

        results.append(row)

        print(f"Accuracy:                 {accuracy:.4f}")
        print(f"Precision:                {precision:.4f}")
        print(f"Recall:                   {recall:.4f}")
        print(f"F1:                       {f1:.4f}")
        print(f"End-to-end MAE:           {end_to_end_mae:.4f} days")
        print(f"End-to-end RMSE:          {end_to_end_rmse:.4f} days")
        print(f"End-to-end R2:            {end_to_end_r2:.4f}")
        print(f"MAE on all actual delays: {actual_delayed_mae:.4f} days")
        print(f"MAE on detected delays:   {detected_delayed_mae:.4f} days")
        print(f"True negatives:           {tn}")
        print(f"False positives:          {fp}")
        print(f"False negatives:          {fn}")
        print(f"True positives:           {tp}")
        print(f"Predicted delayed:        {int(predicted_class.sum())}")

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        ["end_to_end_mae", "f1"],
        ascending=[True, False],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "integrated_validation_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\n\n========== FINAL VALIDATION COMPARISON ==========")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved comparison: {output_path}")
    print("\nNo models were trained or overwritten.")
    print("The reserved test set was not used.")


if __name__ == "__main__":
    main()
