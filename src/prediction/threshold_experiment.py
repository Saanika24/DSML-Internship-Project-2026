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
from src.prediction.model_utils import load_model


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "raw" / "purchase_orders.csv"
OUTPUT_DIR = ROOT_DIR / "models_threshold_experiment"

CLASSIFIERS = {
    "old_classifier": "models",
    "new_classifier": "models_classifier_experiment",
}

REGRESSORS = {
    "old_regressor": "models_conditional_regression",
    "new_regressor": "models_regression_experiment",
}

THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)


def calculate_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
        "mean_predicted_delay": float(np.mean(y_pred)),
    }


def load_artifact(directory, filename):
    path = ROOT_DIR / directory / filename

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    return load_model(path)


def get_delay_probabilities(artifact, X):
    feature_columns = artifact["feature_columns"]

    missing = [
        column for column in feature_columns
        if column not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Classifier is missing features: {missing}"
        )

    model = artifact["model"]
    model_features = X[feature_columns].copy()

    if model_features.isna().any().any():
        raise ValueError("Classifier input contains missing values.")

    classes = list(model.named_steps["model"].classes_)

    if 1 not in classes:
        raise ValueError("Classifier does not contain delayed class 1.")

    positive_index = classes.index(1)

    return model.predict_proba(model_features)[:, positive_index]


def predict_conditional_delay(artifact, X, delayed_mask):
    estimates = np.zeros(len(X), dtype=float)

    if not delayed_mask.any():
        return estimates

    feature_columns = artifact["feature_columns"]

    missing = [
        column for column in feature_columns
        if column not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Regressor is missing features: {missing}"
        )

    selected_features = X.loc[delayed_mask, feature_columns].copy()

    if selected_features.isna().any().any():
        raise ValueError("Regressor input contains missing values.")

    model = artifact.get("model")

    if model is None:
        if "constant_prediction" not in artifact:
            raise ValueError("Regressor model or baseline is missing.")

        delayed_estimates = np.full(
            len(selected_features),
            float(artifact["constant_prediction"]),
        )
    else:
        delayed_estimates = model.predict(selected_features)

    estimates[delayed_mask] = np.maximum(
        np.asarray(delayed_estimates, dtype=float),
        0,
    )

    return estimates


def evaluate_combination(
    classifier_name,
    regressor_name,
    probabilities,
    regressor_artifact,
    X_val,
    y_class,
    y_delay,
):
    rows = []

    for threshold in THRESHOLDS:
        predicted_class = (probabilities >= threshold).astype(int)

        delayed_mask = predicted_class == 1

        predicted_delay = predict_conditional_delay(
            regressor_artifact,
            X_val,
            delayed_mask,
        )

        tn = int(((y_class == 0) & (predicted_class == 0)).sum())
        fp = int(((y_class == 0) & (predicted_class == 1)).sum())
        fn = int(((y_class == 1) & (predicted_class == 0)).sum())
        tp = int(((y_class == 1) & (predicted_class == 1)).sum())

        regression = calculate_metrics(
            y_delay.to_numpy(),
            predicted_delay,
        )

        rows.append({
            "classifier": classifier_name,
            "regressor": regressor_name,
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_class, predicted_class),
            "precision": precision_score(
                y_class, predicted_class, zero_division=0
            ),
            "recall": recall_score(
                y_class, predicted_class, zero_division=0
            ),
            "f1": f1_score(
                y_class, predicted_class, zero_division=0
            ),
            **regression,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "predicted_delayed_orders": int(predicted_class.sum()),
        })

    return rows


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    raw_df = pd.read_csv(DATA_PATH)

    print("Engineering point-in-time features...")
    df = build_purchase_order_features(raw_df)

    X, y_classification, y_delay = prepare_prediction_inputs(df)

    train_df, val_df, test_df = split_by_date(df)

    X_val = X.loc[val_df.index].copy()
    y_class = y_classification.loc[val_df.index].astype(int)
    y_delay_val = y_delay.loc[val_df.index].astype(float)

    print("\n========== DATA ==========")
    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(val_df)}")
    print(f"Reserved test records: {len(test_df)}")
    print(f"Actual delayed validation orders: {int(y_class.sum())}")

    # Baseline: zero estimated delay for every validation order.
    zero_predictions = np.zeros(len(y_delay_val), dtype=float)
    zero_baseline_mae = mean_absolute_error(
        y_delay_val,
        zero_predictions,
    )
    zero_baseline_rmse = np.sqrt(
        mean_squared_error(y_delay_val, zero_predictions)
    )

    print("\n========== ZERO-DELAY BASELINE ==========")
    print(f"MAE: {zero_baseline_mae:.4f}")
    print(f"RMSE: {zero_baseline_rmse:.4f}")

    all_rows = []

    for classifier_name, classifier_dir in CLASSIFIERS.items():
        print(f"\nLoading {classifier_name}...")
        classifier_artifact = load_artifact(
            classifier_dir,
            "purchase_order_classifier.joblib",
        )

        probabilities = get_delay_probabilities(
            classifier_artifact,
            X_val,
        )

        print(
            f"Probability range: {probabilities.min():.4f} "
            f"to {probabilities.max():.4f}"
        )

        for regressor_name, regressor_dir in REGRESSORS.items():
            print(f"\nEvaluating {classifier_name} + {regressor_name}")

            regressor_artifact = load_artifact(
                regressor_dir,
                "conditional_delay_regressor.joblib",
            )

            rows = evaluate_combination(
                classifier_name,
                regressor_name,
                probabilities,
                regressor_artifact,
                X_val,
                y_class,
                y_delay_val,
            )

            all_rows.extend(rows)

    results = pd.DataFrame(all_rows)

    # Selection criteria are calculated only from validation results.
    best_mae = (
        results.sort_values(
            ["mae", "f1"],
            ascending=[True, False],
        )
        .groupby(["classifier", "regressor"], as_index=False)
        .head(1)
        .copy()
    )
    best_mae["selection_rule"] = "lowest_validation_mae"

    best_f1 = (
        results.sort_values(
            ["f1", "precision"],
            ascending=[False, False],
        )
        .groupby(["classifier", "regressor"], as_index=False)
        .head(1)
        .copy()
    )
    best_f1["selection_rule"] = "highest_validation_f1"

    high_recall = results[results["recall"] >= 0.95].copy()

    if not high_recall.empty:
        best_high_recall = (
            high_recall.sort_values(
                ["mae", "f1"],
                ascending=[True, False],
            )
            .groupby(["classifier", "regressor"], as_index=False)
            .head(1)
            .copy()
        )
        best_high_recall["selection_rule"] = (
            "lowest_mae_with_recall_at_least_0.95"
        )
    else:
        best_high_recall = pd.DataFrame()

    selected = pd.concat(
        [best_mae, best_f1, best_high_recall],
        ignore_index=True,
    )

    # Flag thresholds whose MAE beats the zero-delay baseline.
    results["beats_zero_baseline_mae"] = (
        results["mae"] < zero_baseline_mae
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_path = OUTPUT_DIR / "threshold_grid_comparison.csv"
    selected_path = OUTPUT_DIR / "selected_thresholds.csv"
    baseline_path = OUTPUT_DIR / "baseline_reference.csv"

    results.to_csv(grid_path, index=False)
    selected.to_csv(selected_path, index=False)

    pd.DataFrame([{
        "baseline": "always_predict_zero",
        "mae": zero_baseline_mae,
        "rmse": zero_baseline_rmse,
    }]).to_csv(baseline_path, index=False)

    print("\n========== ALL THRESHOLD RESULTS ==========")
    print(
        results.sort_values(
            ["mae", "f1"],
            ascending=[True, False],
        ).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n========== SELECTED THRESHOLDS ==========")
    columns = [
        "selection_rule",
        "classifier",
        "regressor",
        "threshold",
        "mae",
        "rmse",
        "r2",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
        "predicted_delayed_orders",
    ]

    print(
        selected[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n========== BASELINE CHECK ==========")
    print(f"Zero-delay baseline MAE: {zero_baseline_mae:.4f}")
    print(
        "Threshold rows beating zero baseline:",
        int(results["beats_zero_baseline_mae"].sum()),
        "out of",
        len(results),
    )

    print("\n========== SAVED FILES ==========")
    print(grid_path)
    print(selected_path)
    print(baseline_path)

    print("\nThreshold analysis completed.")
    print("No models were trained or overwritten.")
    print("Reserved test data was not used.")
    print(
        "Thresholds selected on this validation set are exploratory; "
        "they are not final unbiased performance estimates."
    )


if __name__ == "__main__":
    main()
