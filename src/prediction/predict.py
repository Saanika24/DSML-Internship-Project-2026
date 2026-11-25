from pathlib import Path
import numpy as np
import pandas as pd

from src.prediction.model_utils import load_model


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"

CLASSIFIER_PATH = MODEL_DIR / "purchase_order_classifier.joblib"
REGRESSOR_PATH = MODEL_DIR / "purchase_order_regressor.joblib"


def load_prediction_models():
    return load_model(CLASSIFIER_PATH), load_model(REGRESSOR_PATH)


def validate_prediction_data(df, feature_cols):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    features = df[feature_cols].copy()
    if features.isna().any().any():
        raise ValueError("Input features contain missing values.")

    return features


def predict_purchase_order_delays(df):
    clf_artifact, reg_artifact = load_prediction_models()
    feature_cols = clf_artifact["feature_columns"]

    if feature_cols != reg_artifact["feature_columns"]:
        raise ValueError("Classifier and regressor feature columns do not match.")

    features = validate_prediction_data(df, feature_cols)

    clf = clf_artifact["model"]
    threshold = clf_artifact.get("threshold", 0.5)

    classes = list(clf.named_steps["model"].classes_)
    if 1 not in classes:
        raise ValueError("Classifier is missing the delayed class (1).")

    pos_idx = classes.index(1)
    probs = clf.predict_proba(features)[:, pos_idx]
    status = np.where(probs >= threshold, "Delayed", "On time")

    reg = reg_artifact.get("model")
    if reg is None:
        # Fall back to the saved baseline median when no regression model beat the baseline
        if "constant_prediction" not in reg_artifact:
            raise ValueError("Baseline median delay value is missing from regressor artifact.")
        est_days = np.full(len(features), float(reg_artifact["constant_prediction"]))
    else:
        est_days = np.maximum(reg.predict(features), 0)

    return pd.DataFrame(
        {
            "delay_probability": probs,
            "predicted_status": status,
            "estimated_delay_days": est_days,
        },
        index=df.index,
    )


if __name__ == "__main__":
    print("Ready. Import predict_purchase_order_delays() with engineered features to run predictions.")