from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from src.prediction.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    build_purchase_order_features,
    prepare_prediction_inputs,
)
from src.prediction.preprocessing import build_preprocessor
from src.prediction.train import split_by_date
from src.prediction.model_utils import save_model


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "raw" / "purchase_orders.csv"
OUTPUT_DIR = ROOT_DIR / "models_classifier_experiment"

BASE_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS[:10]
FULL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS.copy()


def build_classifier(X_train, config, random_state=42):
    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        reg_lambda=config["reg_lambda"],
        scale_pos_weight=config["scale_pos_weight"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", model),
    ])


def evaluate_thresholds(y_true, probabilities):
    rows = []

    for threshold in np.arange(0.10, 0.81, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "predicted_delayed": int(predictions.sum()),
        })

    return pd.DataFrame(rows)


def run_classifier_experiment(random_state=42):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    raw_df = pd.read_csv(DATA_PATH)

    print("Engineering features...")
    df = build_purchase_order_features(raw_df)
    X, y_classification, _ = prepare_prediction_inputs(df)

    train_df, val_df, test_df = split_by_date(df)

    X_train_full = X.loc[train_df.index]
    X_val_full = X.loc[val_df.index]

    y_train = y_classification.loc[train_df.index]
    y_val = y_classification.loc[val_df.index]

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError("Training and validation need both classes.")

    print("\nChronological dataset split:")
    print(f"Training:   {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")
    print("Test data will not be used in model selection.")

    configurations = {
        "baseline": {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
        },
        "shallow": {
            "n_estimators": 350,
            "max_depth": 3,
            "learning_rate": 0.04,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "scale_pos_weight": 1.0,
        },
        "regularized": {
            "n_estimators": 350,
            "max_depth": 3,
            "learning_rate": 0.04,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 5.0,
            "scale_pos_weight": 1.0,
        },
        "class_balanced": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.04,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
            "scale_pos_weight": (
                (y_train == 0).sum() / (y_train == 1).sum()
            ),
        },
    }

    feature_sets = {
        "10_features": BASE_FEATURE_COLUMNS,
        "14_features": FULL_FEATURE_COLUMNS,
    }

    results = []
    best_model = None
    best_artifact = None
    best_score = -1.0
    best_precision = -1.0

    for feature_name, feature_columns in feature_sets.items():
        X_train = X_train_full[feature_columns]
        X_val = X_val_full[feature_columns]

        for config_name, config in configurations.items():
            experiment_name = f"{feature_name}_{config_name}"

            print(f"\nTraining: {experiment_name}")

            model = build_classifier(
                X_train,
                config,
                random_state,
            )

            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_val)[:, 1]

            threshold_table = evaluate_thresholds(
                y_val,
                probabilities,
            )

            best_row = threshold_table.sort_values(
                ["f1", "precision"],
                ascending=[False, False],
            ).iloc[0]

            threshold = float(best_row["threshold"])
            val_f1 = float(best_row["f1"])
            val_precision = float(best_row["precision"])
            val_recall = float(best_row["recall"])

            print(
                f"Best threshold: {threshold:.2f} | "
                f"Precision: {val_precision:.4f} | "
                f"Recall: {val_recall:.4f} | "
                f"F1: {val_f1:.4f}"
            )

            results.append({
                "experiment": experiment_name,
                "feature_count": len(feature_columns),
                "threshold": threshold,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1,
                "model": model,
                "feature_columns": feature_columns,
                "threshold_table": threshold_table,
                "config": config,
            })

            if (
                val_f1 > best_score
                or (
                    np.isclose(val_f1, best_score)
                    and val_precision > best_precision
                )
            ):
                best_score = val_f1
                best_precision = val_precision
                best_model = model
                best_artifact = {
                    "model": model,
                    "feature_columns": feature_columns,
                    "target": "is_delayed",
                    "model_type": "classification",
                    "model_name": experiment_name,
                    "threshold": threshold,
                    "validation_precision": val_precision,
                    "validation_recall": val_recall,
                    "validation_f1": val_f1,
                    "training_config": config,
                }

    summary = pd.DataFrame([
        {
            "experiment": item["experiment"],
            "features": item["feature_count"],
            "threshold": item["threshold"],
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1"],
        }
        for item in results
    ])

    summary = summary.sort_values(
        ["f1", "precision"],
        ascending=[False, False],
    )

    print("\n========== VALIDATION MODEL COMPARISON ==========")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_result = next(
        item for item in results
        if item["experiment"] == best_artifact["model_name"]
    )

    print("\n========== BEST MODEL THRESHOLD ANALYSIS ==========")
    print(
        best_result["threshold_table"].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = save_model(
        best_artifact,
        OUTPUT_DIR / "purchase_order_classifier.joblib",
    )

    summary.drop(columns=[], errors="ignore").to_csv(
        OUTPUT_DIR / "validation_comparison.csv",
        index=False,
    )

    print("\n========== SELECTED EXPERIMENT ==========")
    print("Model:", best_artifact["model_name"])
    print("Features:", len(best_artifact["feature_columns"]))
    print("Threshold:", best_artifact["threshold"])
    print("Validation F1:", best_artifact["validation_f1"])
    print("Validation precision:", best_artifact["validation_precision"])
    print("Validation recall:", best_artifact["validation_recall"])
    print("Saved model:", model_path)
    print("\nExisting production models were not modified.")

    return best_artifact


if __name__ == "__main__":
    run_classifier_experiment()
