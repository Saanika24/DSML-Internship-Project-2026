from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.prediction.evaluation import classification_metrics, regression_metrics
from src.prediction.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    build_purchase_order_features,
    prepare_prediction_inputs,
)
from src.prediction.model_utils import save_model
from src.prediction.preprocessing import build_preprocessor


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "purchase_orders.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "models"


def split_by_date(df, train_size=0.7, val_size=0.1):
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Training and validation sizes must be positive.")

    if train_size + val_size >= 1:
        raise ValueError("Training and validation sizes must add up to less than 1.")

    order_days = pd.to_datetime(df["order_date"]).dt.normalize()
    unique_days = np.sort(order_days.unique())

    if len(unique_days) < 3:
        raise ValueError("Need at least three distinct order dates to split.")

    train_end = int(len(unique_days) * train_size)
    val_end = int(len(unique_days) * (train_size + val_size))

    train_end = max(1, min(train_end, len(unique_days) - 2))
    val_end = max(train_end + 1, min(val_end, len(unique_days) - 1))

    train_cutoff = unique_days[train_end - 1]
    val_cutoff = unique_days[val_end - 1]

    train_df = df[order_days <= train_cutoff].copy()
    val_df = df[(order_days > train_cutoff) & (order_days <= val_cutoff)].copy()
    test_df = df[order_days > val_cutoff].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the chronological splits is empty.")

    return train_df, val_df, test_df


def build_pipeline(model, X_train):
    return Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", model),
    ])


def find_best_threshold(y_true, probs):
    best_thresh = 0.5
    best_f1 = -1.0
    best_prec = -1.0

    for thresh in np.arange(0.10, 0.91, 0.05):
        preds = (probs >= thresh).astype(int)
        metrics = classification_metrics(y_true, preds, positive_label=1)

        f1 = metrics.get("delayed_f1", 0.0)
        prec = metrics.get("delayed_precision", 0.0)

        # Break F1 ties using higher precision
        if f1 > best_f1 or (np.isclose(f1, best_f1) and prec > best_prec):
            best_f1 = f1
            best_prec = prec
            best_thresh = float(round(thresh, 2))

    return best_thresh


def train_models(
    data_path=DEFAULT_DATA_PATH,
    model_dir=DEFAULT_MODEL_DIR,
    random_state=42,
):
    data_path = Path(data_path)
    model_dir = Path(model_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    print("Loading data...")
    raw_df = pd.read_csv(data_path)

    print("Engineering features...")
    df = build_purchase_order_features(raw_df)
    X, y_clf, y_reg = prepare_prediction_inputs(df)

    feature_cols = MODEL_FEATURE_COLUMNS.copy()
    X = X[feature_cols]

    # Split by date so same-day orders never span across train/val/test
    train_df, val_df, test_df = split_by_date(df)

    X_train, X_val, X_test = X.loc[train_df.index], X.loc[val_df.index], X.loc[test_df.index]
    y_train, y_val, y_test = y_clf.loc[train_df.index], y_clf.loc[val_df.index], y_clf.loc[test_df.index]
    delay_train, delay_val, delay_test = y_reg.loc[train_df.index], y_reg.loc[val_df.index], y_reg.loc[test_df.index]

    if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Train, validation, and test sets must each contain both on-time and delayed orders.")

    print(f"\nSplit sizes: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")
    print(f"Train ends: {train_df['order_date'].max().date()}")
    print(f"Val range:  {val_df['order_date'].min().date()} to {val_df['order_date'].max().date()}")
    print(f"Test starts: {test_df['order_date'].min().date()}")

    # 1. Classification model selection
    print("\nTraining classifiers...")
    candidates = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
            X_train,
        ),
        "XGBoost": build_pipeline(
            XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            X_train,
        ),
    }

    best_clf_name = None
    best_clf = None
    best_thresh = 0.5
    best_val_f1 = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        thresh = find_best_threshold(y_val, val_probs)

        val_preds = (val_probs >= thresh).astype(int)
        val_metrics = classification_metrics(y_val, val_preds, positive_label=1)
        val_f1 = val_metrics.get("delayed_f1", 0.0)

        print(f"  {name}: threshold={thresh}, val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_clf_name = name
            best_clf = model
            best_thresh = thresh

    print(f"Selected classifier: {best_clf_name}")

    test_probs = best_clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)
    clf_results = classification_metrics(y_test, test_preds, test_probs, positive_label=1)

    # 2. Regression model selection (compare against median baseline)
    print("\nTraining regressor...")
    reg_pipe = build_pipeline(
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
        X_train,
    )
    reg_pipe.fit(X_train, delay_train)

    val_reg_preds = np.maximum(reg_pipe.predict(X_val), 0)
    median_delay = float(delay_train.median())
    val_baseline_preds = np.full(len(delay_val), median_delay)

    xgb_val_mae = regression_metrics(delay_val, val_reg_preds)["mae"]
    base_val_mae = regression_metrics(delay_val, val_baseline_preds)["mae"]

    print(f"  XGBoost val MAE:  {xgb_val_mae:.4f}")
    print(f"  Baseline val MAE: {base_val_mae:.4f}")

    if xgb_val_mae <= base_val_mae:
        best_reg = reg_pipe
        best_reg_name = "XGBoost"
        reg_artifact = {
            "model": reg_pipe,
            "feature_columns": feature_cols,
            "target": "delay_days",
            "model_type": "regression",
            "model_name": best_reg_name,
        }
        test_reg_preds = np.maximum(best_reg.predict(X_test), 0)
    else:
        best_reg_name = "Median baseline"
        reg_artifact = {
            "model": None,
            "feature_columns": feature_cols,
            "target": "delay_days",
            "model_type": "regression",
            "model_name": best_reg_name,
            "constant_prediction": median_delay,
        }
        test_reg_preds = np.full(len(X_test), median_delay)

    reg_results = regression_metrics(delay_test, test_reg_preds)

    # 3. Save models
    model_dir.mkdir(parents=True, exist_ok=True)

    clf_path = save_model(
        {
            "model": best_clf,
            "feature_columns": feature_cols,
            "target": "is_delayed",
            "model_type": "classification",
            "model_name": best_clf_name,
            "threshold": best_thresh,
        },
        model_dir / "purchase_order_classifier.joblib",
    )
    reg_path = save_model(reg_artifact, model_dir / "purchase_order_regressor.joblib")

    # 4. Report test metrics
    print(f"\n--- Classification Test Results ({best_clf_name}, threshold={best_thresh}) ---")
    for k, v in clf_results.items():
        if k != "classification_report":
            print(f"{k}: {v}")

    print(f"\n--- Regression Test Results ({best_reg_name}) ---")
    for k, v in reg_results.items():
        print(f"{k}: {v}")

    print(f"\nSaved classifier to: {clf_path}")
    print(f"Saved regressor to:  {reg_path}")

    return {
        "classification": clf_results,
        "regression": reg_results,
        "classifier_name": best_clf_name,
        "regressor_name": best_reg_name,
        "classification_threshold": best_thresh,
        "classifier_path": clf_path,
        "regressor_path": reg_path,
        "train_records": len(X_train),
        "validation_records": len(X_val),
        "test_records": len(X_test),
        "feature_columns": feature_cols,
    }


if __name__ == "__main__":
    train_models()