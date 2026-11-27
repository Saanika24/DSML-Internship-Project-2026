from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.model_utils import load_model


ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_DIR = ROOT_DIR / "models"
DEFAULT_CONDITIONAL_REGRESSOR_DIR = ROOT_DIR / "models_conditional_regression"

CLASSIFIER_FILENAME = "purchase_order_classifier.joblib"
REGRESSOR_FILENAME = "conditional_delay_regressor.joblib"


def load_prediction_models(model_dir=None, regressor_dir=None):
    """Load classifier and conditional regressor independently."""

    if model_dir is None:
        classifier_dir = DEFAULT_MODEL_DIR
    else:
        classifier_dir = Path(model_dir)
        if not classifier_dir.is_absolute():
            classifier_dir = ROOT_DIR / classifier_dir

    if regressor_dir is not None:
        selected_regressor_dir = Path(regressor_dir)
        if not selected_regressor_dir.is_absolute():
            selected_regressor_dir = ROOT_DIR / selected_regressor_dir
    elif model_dir is None:
        selected_regressor_dir = DEFAULT_CONDITIONAL_REGRESSOR_DIR
    else:
        selected_regressor_dir = classifier_dir

    clf_artifact = load_model(
        classifier_dir / CLASSIFIER_FILENAME
    )
    reg_artifact = load_model(
        selected_regressor_dir / REGRESSOR_FILENAME
    )

    return clf_artifact, reg_artifact


def validate_prediction_data(df, feature_cols):
    """Validate and select the features required by one model."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    missing = [
        col for col in feature_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    features = df[feature_cols].copy()

    if features.isna().any().any():
        raise ValueError("Input features contain missing values.")

    return features


def predict_purchase_order_delays(
    df,
    model_dir=None,
    regressor_dir=None,
):
    """Predict delay probability and conditional delay severity."""

    if model_dir is None and regressor_dir is None:
        clf_artifact, reg_artifact = load_prediction_models()
    else:
        clf_artifact, reg_artifact = load_prediction_models(
            model_dir=model_dir,
            regressor_dir=regressor_dir,
        )

    # Each model selects its own required feature columns.
    classifier_features = validate_prediction_data(
        df,
        clf_artifact["feature_columns"],
    )

    # Stage 1: classify whether each order is delayed.
    clf = clf_artifact["model"]
    threshold = clf_artifact.get("threshold", 0.5)

    classes = list(clf.named_steps["model"].classes_)

    if 1 not in classes:
        raise ValueError("Classifier is missing the delayed class (1).")

    delayed_idx = classes.index(1)

    delay_probs = clf.predict_proba(
        classifier_features
    )[:, delayed_idx]

    predicted_status = np.where(
        delay_probs >= threshold,
        "Delayed",
        "On time",
    )

    # Stage 2: estimate delay severity only for predicted delayed orders.
    estimated_days = np.zeros(len(df), dtype=float)
    delayed_mask = predicted_status == "Delayed"

    if delayed_mask.any():
        regressor_features = validate_prediction_data(
            df.loc[delayed_mask],
            reg_artifact["feature_columns"],
        )

        reg = reg_artifact.get("model")

        if reg is None:
            if "constant_prediction" not in reg_artifact:
                raise ValueError(
                    "Conditional regression model or baseline is missing."
                )

            delayed_predictions = np.full(
                len(regressor_features),
                float(reg_artifact["constant_prediction"]),
            )
        else:
            delayed_predictions = reg.predict(regressor_features)

        estimated_days[delayed_mask] = np.maximum(
            delayed_predictions,
            0,
        )

    return pd.DataFrame(
        {
            "delay_probability": delay_probs,
            "predicted_status": predicted_status,
            "estimated_delay_days": estimated_days,
        },
        index=df.index,
    )


if __name__ == "__main__":
    print(
        "Ready. Call predict_purchase_order_delays(df) "
        "with engineered features."
    )


def predict_new_purchase_orders(
    new_orders,
    historical_orders,
    model_dir=None,
    regressor_dir=None,
):
    """Build inference features and predict delays for new purchase orders."""

    from src.prediction.feature_engineering import build_inference_features

    features = build_inference_features(
        new_orders=new_orders,
        historical_orders=historical_orders,
    )

    return predict_purchase_order_delays(
        features,
        model_dir=model_dir,
        regressor_dir=regressor_dir,
    )
