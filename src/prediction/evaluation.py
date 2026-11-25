
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
    positive_class_index=1,
    positive_label=1,
):
    """Calculate classification metrics, including delayed-order metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Target arrays must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(
            "Actual and predicted labels must have equal, "
            "non-zero lengths."
        )

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    # Additional metrics for binary delayed-order classification.
    if len(np.unique(y_true)) == 2 and positive_label in np.unique(y_true):
        results["delayed_precision"] = precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        )
        results["delayed_recall"] = recall_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        )
        results["delayed_f1"] = f1_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        )

        other_label = next(
            label for label in np.unique(y_true)
            if label != positive_label
        )

        results["on_time_recall"] = recall_score(
            y_true, y_pred, pos_label=other_label, zero_division=0
        )

    # Calculate ROC-AUC when binary probabilities are provided.
    if y_prob is not None and len(np.unique(y_true)) == 2:
        probabilities = np.asarray(y_prob)

        if probabilities.ndim == 2:
            if not 0 <= positive_class_index < probabilities.shape[1]:
                raise ValueError("Invalid positive class index.")

            probabilities = probabilities[:, positive_class_index]

        if probabilities.ndim != 1 or len(probabilities) != len(y_true):
            raise ValueError(
                "Probability values must match the number of samples."
            )

        if not np.isfinite(probabilities).all():
            raise ValueError("Probability values must be finite.")

        results["roc_auc"] = roc_auc_score(
            (y_true == positive_label).astype(int),
            probabilities,
        )

    return results


def regression_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Target arrays must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(
            "Actual and predicted values must have equal, "
            "non-zero lengths."
        )

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Regression values must be finite.")

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2_score": (
            r2_score(y_true, y_pred)
            if len(y_true) >= 2 else None
        ),
    }
